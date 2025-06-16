import argparse
import os
import sys
import time
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, pdb, math, copy
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist
from target_datasets import FileListDataset, Imagenet_Dataset
from target_datasets import my_Dataset as Dataset
from loss import infoNCE, SupConLoss
from test import val_office, val_pclass
from PIL import Image
from net.pytorch_pretrained_vit import ViT
from augmentation import get_augmentation_versions, test_augmentation
from torchvision.transforms import ToPILImage
from itertools import combinations
import logging


def arg_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_epoches', default=15, type=int)
    parser.add_argument('--source_model', default='./model_source/20240717-1602-single_gpu_cal256_ce_vit_B_16_best.pkl')
    parser.add_argument('--source_centers', default='./model_source/20240717-1602vit_B_16_cal256_source_centers_mean.pkl')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')
    parser.add_argument('--aug_versions', default='tw')
    parser.add_argument('--aug_type', default='moco-v2')
    parser.add_argument('--aug_nums', default=2, type=int)
    parser.add_argument('--con_coeff', default=0.5, type=int)
    parser.add_argument('--nav_t', default=1, type=float, help='temperature for the navigator')
    parser.add_argument('--s_par', default=0.5, type=float, help='s_par')
    
    args = parser.parse_args()
    return args


def setup_logging(dataset_name, source_domain, target_domain):
    """Setup logging with custom filename"""
    log_dir = f"./log/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{dataset_name}_{source_domain}_to_{target_domain}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    return logger


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_full_model(model_path, n_gpus):
    """
    Load complete model instance, adapting to different GPU configurations
    """
    model = torch.load(model_path)
    if isinstance(model, nn.Module):
        is_data_parallel = any([k.startswith('module.') for k, _ in model.state_dict().items()])
        if n_gpus > 1 and not is_data_parallel:
            model = nn.DataParallel(model)
        elif n_gpus <= 1 and is_data_parallel:
            model = model.module
    return model


class reply_dataset(torch.utils.data.Dataset):
    """Dataset class for reply buffer"""
    def __init__(self, images, labels, buffer_per_class, soft_predictions, return_id=False):
        super(reply_dataset, self).__init__()
        self.images = torch.cat([torch.tensor(img).unsqueeze(0) for imgs in images for img in imgs], dim=0)
        self.labels = torch.cat([torch.tensor(label).repeat(buffer_per_class, 1) for label in labels], dim=0)[:len(self.images)]
        self.batch_soft_pred = torch.cat([soft_pred.unsqueeze(0) for soft_preds in soft_predictions for soft_pred in soft_preds], dim=0)
        self.images = self.images.cpu()
        self.labels = self.labels.cpu()
        self.batch_soft_pred = self.batch_soft_pred.cpu()
        self.return_id = return_id

    def __getitem__(self, index):
        if index >= len(self.images) or index < 0:
            raise IndexError(f"Index {index} is out of bounds.")
        else:
            if self.return_id:
                return index, self.images[index], self.labels[index], self.batch_soft_pred[index]
            else:
                return self.images[index], self.labels[index], self.batch_soft_pred[index]

    def __len__(self):
        return len(self.labels)


def cosine_similarity(feature, pairs):
    """
    Calculate cosine similarity between features and pairs
    """
    feature = F.normalize(feature)
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())
    return similarity


def obtain_label(total_sample_nums, source_weight, dataloader, model, confi_class_idx, args, epoch):
    """
    Obtain pseudo labels for target domain samples
    """
    model.eval()
    emd_feat_stack = []
    cls_out_stack = []
    gt_label_stack = []
    idx_stack = []
    logits1 = []
    logits2 = []
    feats1 = []
    feats2 = []
    prediction_bank = torch.zeros(1, 256).cuda()
    
    with torch.no_grad():
        for train_data in tqdm(dataloader, ncols=60):
            data = torch.cat(train_data[0], dim=0).cuda()
            label = train_data[1].repeat(len(train_data[0])).cuda()
            original_idx = train_data[2].cuda()
            augmented_idx = original_idx + total_sample_nums // args.aug_nums
            idx = torch.cat([original_idx, augmented_idx], dim=0).cuda()

            cls_out, embed_feat = model(data)
            output_prob = F.softmax(cls_out, dim=1).data
            cls_out = cls_out[:, confi_class_idx]
            embed_feat = embed_feat[:, 0, :]

            split_nums = int(data.shape[0] // args.aug_nums)
            original_feats, augmented_feats = embed_feat[:split_nums], embed_feat[split_nums:]
            original_logits, augmented_logits = cls_out[:split_nums], cls_out[split_nums:]
            
            # Calculate cumulative probability for each class
            batch_prob_sum = torch.sum(output_prob, dim=0)
            prediction_bank += batch_prob_sum.unsqueeze(0)

            feats1.append(original_feats)
            feats2.append(augmented_feats)
            logits1.append(original_logits)
            logits2.append(augmented_logits)
            emd_feat_stack.append(embed_feat)
            cls_out_stack.append(cls_out)
            gt_label_stack.append(label)
            idx_stack.append(idx)

    # Normalize prediction probabilities
    prediction_bank = prediction_bank.squeeze(0)
    max_cls = torch.max(prediction_bank)
    min_cls = torch.min(prediction_bank)
    prediction_bank = (prediction_bank - min_cls) / (max_cls - min_cls)
    
    all_idx = torch.cat(idx_stack).int()
    all_gt_label = torch.cat(gt_label_stack)
    all_emd_feat = torch.cat(emd_feat_stack, dim=0)
    all_emd_feat = all_emd_feat / torch.norm(all_emd_feat, p=2, dim=1, keepdim=True)
    all_cls_out = torch.cat(cls_out_stack, dim=0)
    all_cls_out = torch.softmax(all_cls_out, dim=1)
    all_logits_max = all_cls_out.max(1)[0]
    _, all_psd_label = torch.max(all_cls_out, dim=1)
    all_psd_label = torch.tensor([confi_class_idx[i] for i in all_psd_label], device='cuda:0')

    model_accuracy = torch.sum(all_gt_label == all_psd_label) / len(all_gt_label)
    logger.info(f"Initial accuracy: {model_accuracy:.4f}")

    feats1 = torch.cat(feats1, dim=0)
    feats2 = torch.cat(feats2, dim=0)
    features = torch.cat((feats1.unsqueeze(0), feats2.unsqueeze(0)), dim=0)
    softmax_logits1 = torch.softmax(torch.cat(logits1, dim=0), dim=1)
    softmax_logits2 = torch.softmax(torch.cat(logits2, dim=0), dim=1)
    _, psd_labels1_idx = torch.max(softmax_logits1, dim=1)
    _, psd_labels2_idx = torch.max(softmax_logits2, dim=1)
    psd_labels1 = torch.tensor([confi_class_idx[i] for i in psd_labels1_idx], device='cuda:0')
    psd_labels2 = torch.tensor([confi_class_idx[i] for i in psd_labels2_idx], device='cuda:0')

    logits1 = torch.cat(logits1, dim=0)
    logits2 = torch.cat(logits2, dim=0)
    logits = torch.cat((logits1.unsqueeze(0), logits2.unsqueeze(0)), dim=0)
    pred_start = torch.softmax(logits, dim=2).max(2)[0]

    # 1. Confidence-based selection
    pred_con = pred_start
    conf_thres = pred_con.mean()
    confidence_sel = pred_con.mean(0) > conf_thres

    # 2. Uncertainty-based selection
    pred_std = pred_start.std(0)
    uncertainty_threshold = pred_std.mean(0)
    uncertainty_sel = pred_std < uncertainty_threshold

    # 3. NC1-based selection
    class_centers = {}
    for i in confi_class_idx:
        class_feats = torch.cat((feats1[psd_labels1 == i], feats2[psd_labels2 == i]), dim=0)
        if class_feats.shape[0] > 0:
            class_centers[i] = class_feats.mean(dim=0).unsqueeze(0)

    distance_sel = torch.zeros(len(psd_labels1), dtype=torch.bool, device='cuda:0')
    for i in confi_class_idx:
        if i in class_centers:
            class_center = class_centers[i]
            class_feats = features[:, psd_labels1 == i, :]
            
            # Calculate dot product
            dot_product = torch.matmul(class_feats, class_center.T)
            # Calculate norms and ensure shape matching
            class_feats_norm = torch.norm(class_feats, dim=2, keepdim=True)
            class_center_norm = torch.norm(class_center, dim=1, keepdim=True)
            norm_product = class_feats_norm * class_center_norm
            
            # Ensure shape matching for broadcasting
            cosine_similarity = dot_product / (norm_product + 1e-8)
            cosine_distance = 1 - cosine_similarity.squeeze()
            mean_distance = cosine_distance.mean(dim=0)
            distance_sel[psd_labels1 == i] = mean_distance < mean_distance.mean()

    # 4. Combine three selection methods
    truth_array = torch.logical_or(torch.logical_and(confidence_sel, uncertainty_sel), distance_sel)
    ind_keep = truth_array.nonzero()
    if epoch < 5:
        # 5. Address class imbalance issue
        unique_labels, counts = all_psd_label[ind_keep].unique(return_counts = True)
        logger.info(f"Unique labels before 5th iteration: {unique_labels}")
        min_count = min(counts)
        if len(unique_labels) < len(confi_class_idx):
            missing_classes = [ii for ii in confi_class_idx if ii not in unique_labels]
            for i in missing_classes:
                indices = (all_psd_label == i).nonzero(as_tuple=True)[0]
                if indices.numel() > 0 and ind_keep.numel()>0:
                    probs = all_logits_max[indices]
                    _ , index_miss = probs.sort(descending=True)
                    sorted_indices = indices[index_miss]
                    try:
                        new_min_count = min(len(indices), min_count)
                        ind_keep = torch.cat((ind_keep, indices[index_miss[0 : new_min_count]]))         
                    except:
                        pass
        min_nums = None
        # Find indices with highest min_count confidence for each class
        top_indices_per_class = {}
        for class_idx in confi_class_idx:
            class_indices = ind_keep[all_psd_label[ind_keep] == class_idx]
            class_confidences = all_logits_max[class_indices]
            sorted_indices = class_indices[class_confidences.argsort(descending=True)]
            _, nums = all_psd_label[ind_keep].unique(return_counts = True)
            min_nums = min(nums)
            top_indices = sorted_indices[:min_nums]
            top_indices_per_class[int(class_idx)] = top_indices
        feat_sample_idx = torch.cat(list(top_indices_per_class.values()))
    else:
        ind_keep_cent = ind_keep.flatten()
        feat_sample_idx = ind_keep_cent
        unique_labels, counts = all_psd_label[ind_keep].unique(return_counts = True)

    feat_cls_sample = all_emd_feat[feat_sample_idx, :]
    source_weight_cent = source_weight[unique_labels, :]
    source_weight_cent = source_weight_cent.to(feat_cls_sample.device)
    feat_cls_sample = torch.cat([source_weight_cent, feat_cls_sample], dim=0)

    # Get class labels corresponding to each feature center
    multi_label_cent = all_psd_label[feat_sample_idx].cpu().numpy()
    unique_labels_repeated = np.tile(unique_labels.cpu().numpy(), 1)
    centers_labels = np.hstack((unique_labels_repeated, multi_label_cent))

    # Use cosine distance to calculate distances between all_fea and initc
    dd = cdist(all_emd_feat.cpu().numpy(), feat_cls_sample.cpu().numpy(), 'cosine')
    
    # Initialize matrix to save average distance for each class
    class_avg_distances = np.full((dd.shape[0], torch.max(torch.tensor(unique_labels)).item() + 1), np.inf)

    for i, label in enumerate(centers_labels):
        # Calculate average distance from each sample to current class
        class_avg_distances[:, label] = np.minimum(
            class_avg_distances[:, label], 
            np.mean(dd[:, centers_labels == label], axis=1)
        )
    # Select class with minimum average distance as new pseudo label for each sample
    new_psd_labels = np.argmin(class_avg_distances, axis=1)

    all_gt_label_np = all_gt_label.cpu().numpy()
    correct_predictions = np.sum(new_psd_labels == all_gt_label_np)
    accuracy = correct_predictions / len(all_gt_label_np)
    logger.info(f'Updated accuracy: {accuracy * 100:.2f}%')

    # Create empty dictionary to store mapping of indices and new pseudo labels
    label_dict = {}
    for idx, label in zip(all_idx.cpu().numpy(), new_psd_labels):
        label_dict[idx] = label
    
    new_class_centers = {}
    for i in unique_labels:
        class_feats = torch.cat((feats1[psd_labels1 == i], feats2[psd_labels2 == i]), dim=0)
        if class_feats.shape[0] > 0:
            new_class_centers[i] = class_feats.mean(dim=0).unsqueeze(0)
    
    unique_labels, counts = torch.from_numpy(new_psd_labels).unique(return_counts = True)
    confi_class_probs = {idx: prediction_bank[idx].item() for idx in unique_labels.cpu().numpy()}

    return label_dict, model_accuracy * 100, accuracy * 100, unique_labels, new_class_centers, confi_class_probs


def get_target_centers(target_train_loader, net, confi_class_idx, confi_label_dict):
    """
    Get target domain class centers for training on new target domain
    """
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(target_train_loader)
        for _ in range(len(target_train_loader)):
            data = next(iter_test)
            inputs = data[0]
            sample_idx = data[2]
            inputs = inputs.cuda()
            _, feas = net(inputs)
            feas = feas[:, 0, :]

            if start_test:
                all_fea = feas.float().cpu()
                all_idx = sample_idx.float().cpu()
                start_test = False
            else:
                all_idx = torch.cat((all_idx, sample_idx.float()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)

    for idx, cls in enumerate(confi_class_idx):
        cnt = 0
        for i, sam_idx in enumerate(all_idx):
            if confi_label_dict[sam_idx.item()] == cls:
                if cnt == 0:
                    target_cls_proto = all_fea[i]
                else:
                    target_cls_proto += all_fea[i]
                cnt += 1

        target_cls_proto = target_cls_proto / cnt

        if idx == 0:
            total_target_protos = target_cls_proto.unsqueeze(0)
        else:
            total_target_protos = torch.cat((total_target_protos, target_cls_proto.unsqueeze(0)), 0)

    return total_target_protos


def get_confi_classes(source_model, target_data_loader, n_gpus):
    """
    Shared class detection step - calculate cumulative probability for each class
    """
    source_model.eval()
    prediction_bank = torch.zeros(1, 256).cuda()
    all_data = []
    relation_matrices = []
    
    # Load shared class source domain prototypes
    source_centers = load_full_model(args.source_centers, n_gpus)
    source_centers = source_centers[:, 0, :]
    for j, (img_data, _, _) in enumerate(target_data_loader):  
        img_data = torch.cat(img_data, dim=0) 
        img_data = img_data.cuda()  
        all_data.append(img_data)
        
        # Extract features and calculate relation matrix
        with torch.no_grad():
            output, features = source_model(img_data)  
            features = features[:, 0, :]
            output_prob = F.softmax(output, dim=1).data
            relation_matrix = torch.matmul(features, source_centers.t())
            relation_matrices.append(relation_matrix)
        
        # Calculate cumulative probability for each class
        batch_prob_sum = torch.sum(output_prob, dim=0)  
        prediction_bank += batch_prob_sum  
    
    # Accumulate and normalize relation matrices
    relation_matrices = torch.cat(relation_matrices, dim=0)
    relation_matrices = F.softmax(relation_matrices)

    all_data = torch.cat(all_data, dim=0)
    confi_class_idx = []  
    
    # Apply min-max normalization to prediction memory bank
    prediction_bank = prediction_bank.squeeze(0)
    max_cls = torch.max(prediction_bank)
    min_cls = torch.min(prediction_bank)
    prediction_bank = (prediction_bank - min_cls) / (max_cls - min_cls)
    
    # Calculate average probability of normalized prediction memory bank
    avg_prob = torch.mean(prediction_bank)
    # Calculate average of relation matrix
    avg_relation_matrix = torch.mean(relation_matrices, dim=0)
    
    logger.info(f"Average relation matrix mean: {torch.mean(avg_relation_matrix):.4f}")
    
    prob_idx, relation_idx = [], []
    # Determine confident classes based on thresholds
    for idx, value in enumerate(prediction_bank):
        if value > avg_prob:
            prob_idx.append(idx)
        if torch.mean(relation_matrices[:, idx]) > torch.mean(avg_relation_matrix):
            relation_idx.append(idx)
        if value > avg_prob or torch.mean(relation_matrices[:, idx]) > torch.mean(avg_relation_matrix):
            confi_class_idx.append(idx)
    
    logger.info(f"Shared classes: {confi_class_idx}")
    logger.info(f"Classes meeting probability condition: {prob_idx}")
    logger.info(f"Classes meeting similarity condition: {relation_idx}")
    return confi_class_idx, prediction_bank[confi_class_idx], len(all_data)


def get_one_classes_imgs(target_train_loader, class_idx, confi_label_dict, total_sample_nums, aug_nums):
    """
    Extract and return all images belonging to specific class from target training data loader
    """
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(target_train_loader)
        for _ in range(len(target_train_loader)):
            data = next(iter_test)
            inputs = torch.cat(data[0], dim=0)
            labels = data[1].repeat(len(data[0]))
            original_sample_idx = data[2].cuda()
            augmented_sample_idx = original_sample_idx + total_sample_nums / aug_nums
            sample_idx = torch.cat([original_sample_idx, augmented_sample_idx], dim=0).int().cuda()

            if start_test:
                all_inputs = inputs.float().cpu()
                all_idx = sample_idx.int().cpu()
                all_label = labels.int()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)
                all_idx = torch.cat((all_idx, sample_idx.int().cpu()), 0)
                all_label = torch.cat((all_label, labels.int()), 0)

        logger.info(f'Constructing class {class_idx} exemplar')
        imgs_idx = []
        for cnt_idx, idx in enumerate(all_idx):
            if int(idx.item()) in confi_label_dict:
                if confi_label_dict[int(idx.item())] == class_idx:
                    imgs_idx.append(cnt_idx)
                    
        return all_inputs[imgs_idx]


class reply_buffer():  # memory bank
    """
    This class manages and updates a prototype collection (exemplar set) 
    for knowledge distillation and feature representation learning
    """
    def __init__(self, transform, imgs_per_class=20):
        super(reply_buffer, self).__init__()
        self.exemplar_indexes = []  # Initialize prototype image index memory bank list
        self.exemplar_set = []  # Initialize prototype image memory bank list
        self.soft_pred = []  # Initialize prototype soft prediction memory bank list
        self.target_center_set = []  # Initialize target center set
        self.transform = transform  # Save passed image transformation function
        self.m = imgs_per_class  # Number of images saved per class

    def Image_transform(self, images, transform):
        """Apply image transformation"""
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)  
        return data

    def compute_class_mean(self, model, images, transform):
        """Compute class mean features"""
        model.eval()
        with torch.no_grad():
            x = images.cuda()
            model = model
            output, feas = model(x)
            feas = feas[:, 0, :]
            feature_extractor_output = F.normalize(feas.detach()).cpu().numpy()
            class_mean = np.mean(feature_extractor_output, axis=0)
            class_center = np.mean(feas.detach().cpu().numpy(), axis=0)
            output = nn.Softmax(dim=1)(output)

        return class_mean, feature_extractor_output, output, class_center

    def construct_exemplar_set(self, images, model):
        """Construct exemplar set based on nearest neighbor method using target feature centers"""
        class_mean, feature_extractor_output, buffer_output, class_center = self.compute_class_mean(model, images, self.transform)
        
        exemplar_indexes = []  # Prototype image index set
        exemplar = []  # Prototype image set
        soft_predar = []  # Prototype soft label set
        feas_past = []  # Prototype normalized feature set
        now_class_mean = np.zeros((1, 768))

        for i in range(self.m):
            # Calculate distance from each image to current mean
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            # Select image closest to mean
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]

            exemplar_indexes.append(index)
            exemplar.append(images[index])
            soft_predar.append(buffer_output[index])
            feas_past.append(feature_extractor_output[index])
        
        self.exemplar_indexes.append(exemplar_indexes)
        self.exemplar_set.append(exemplar)
        self.soft_pred.append(soft_predar)

    def update_exemplar_set(self, images, model, history_idx):
        """Update existing exemplar set"""
        # Similar to construct exemplar set method, but updates existing set
        class_mean, feature_extractor_output, buffer_output, class_center = self.compute_class_mean(model, images, self.transform)
        
        exemplar_indexes = []  # Store prototype image indices
        exemplar = []
        soft_predar = []
        feas_past = []
        now_class_mean = np.zeros((1, 768))

        for i in range(self.m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar_indexes.append(index)
            exemplar.append(images[index])
            soft_predar.append(buffer_output[index])
            feas_past.append(feature_extractor_output[index])

        self.exemplar_indexes[history_idx] = exemplar_indexes  # Update prototype image index set
        self.exemplar_set[history_idx] = exemplar
        self.soft_pred[history_idx] = soft_predar

    def get_all_exemplars(self):
        """Get all exemplars from memory bank"""
        all_exemplars = torch.cat([torch.stack(exemplars) for exemplars in self.exemplar_set], dim=0)
        return all_exemplars

    def get_all_soft_preds(self):
        """Get all soft predictions from memory bank"""
        all_soft_preds = torch.cat([torch.stack(preds) for preds in self.soft_pred], dim=0)
        return all_soft_preds


def pairwise_cosine_dist(x, y):
    """Calculate pairwise cosine distance between two feature sets"""
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1 - torch.matmul(x, y.T)


def get_pos_logits(sim_mat, prop, eps, nav_t):
    """Calculate positive logits based on similarity matrix and class proportions"""
    log_prior = torch.log(prop + eps)
    return sim_mat/nav_t + log_prior


if __name__ == '__main__':
    args = arg_parser()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
    # dataset
    cal_dataset = Dataset(
        path='./dataset/ImageNet-Caltech',
        domains=['256_ObjectCategories'],
        files=[
            'caltech_list.txt',
        ],
        prefix='./dataset/ImageNet-Caltech')

    imgNet_dataset = Imagenet_Dataset(
        path='./dataset/ImageNet-Caltech',
        domains=['imageNet_84_val'],
        files=[
            'imagenet_84_list.txt',
        ],
        prefix='./dataset/ImageNet-Caltech')
    source = 0
    target = 0
    logger = setup_logging('ImageNet-Caltech', cal_dataset.domains[source], imgNet_dataset.domains[target])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    logger.info(f"Using {n_gpus} GPU(s)")
    logger.info(f"Source domain: {cal_dataset.domains[source]}, Target domain: {imgNet_dataset.domains[target]}")

    last_acc_stages = []
    best_acc_stages = []
    center_acc_stages = []
    top_ten_stages = []

    total_cls_nums = 256
    incre_cls_nums = 10
    
    reply_buffer_nums = 10
    batch_size = args.batchsize
    prototypes_update_interval = 10
    aug_nums = args.aug_nums
    con_coeff = args.con_coeff
    nav_t = args.nav_t
    s_par = args.s_par
    eps = 1e-6
    lr = args.lr
    weight_decay = 1e-6
    momentum = 0.9
    n_epoches = args.n_epoches
    target_file = imgNet_dataset.files[0]
    transform_train = get_augmentation_versions(args)
    transform_test = test_augmentation()
    margin = 0.3
    gamma = 0.07
    info_nce = infoNCE(class_num=total_cls_nums).cuda()
    nll = nn.NLLLoss()
    loss_ce = nn.CrossEntropyLoss()
    confi_cls_history = []
    confi_cls_value = np.zeros(total_cls_nums)
    current_proto_classes = {}
    reply_buffer = reply_buffer(transform_test, reply_buffer_nums)


    pretrained_net = load_full_model(args.source_model, n_gpus)
    pretrained_net = pretrained_net.cuda() if n_gpus > 0 else pretrained_net
    pretrained_net.eval()
    logger.info("Starting incremental learning...")

    cal_84_cls_list = [188, 0, 165, 145, 33, 94, 76, 86, 71, 128, 225, 108, 219, 115, 87, 44, 62, 163, 85, 177, 134,
                       229, 82, 60, 29, 172, 114, 160, 146, 228, 245, 75, 157, 209, 97, 106, 170, 198, 92, 230, 234, 40,
                       200, 151, 11, 27, 185, 47, 45, 9, 88, 37, 30, 2, 90, 196, 181, 28, 253, 112, 178, 109, 39, 96,
                       192, 110, 211, 237, 249, 89, 215, 116, 126, 133, 107, 123, 193, 68, 179, 227, 150, 141, 7, 50]
    cal_84_cls_list.sort()

    for incre_idx in range(len(cal_84_cls_list) // incre_cls_nums):
        logger.info(f"Starting incremental stage {incre_idx + 1}/{len(cal_84_cls_list) // incre_cls_nums}")
        target_train_classes = [cal_84_cls_list[i] for i in
                                range(incre_cls_nums * incre_idx, incre_cls_nums * incre_idx + incre_cls_nums)]
        target_test_classes = [cal_84_cls_list[i] for i in range(0, incre_cls_nums * incre_idx + incre_cls_nums)]
        target_train_ds = FileListDataset(list_path=target_file, path_prefix=imgNet_dataset.prefixes[target],
                                          return_id=True,
                                          transform=transform_train,
                                          filter=(lambda x: x in target_train_classes))
        target_train_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                                     num_workers=2 * 4)
        target_test_ds = FileListDataset(list_path=target_file, path_prefix=imgNet_dataset.prefixes[target],
                                         transform=transform_test,
                                         filter=(lambda x: x in target_test_classes),
                                         return_id=False)
        target_test_loader = torch.utils.data.DataLoader(target_test_ds, batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=2 * 4)
        # Get shared class indices and confidence values
        confi_class_idx, confi_class_values, total_sample_nums = get_confi_classes(pretrained_net, target_train_dl, n_gpus)

        source_weight = pretrained_net.fc.weight.data.clone()
        
        if incre_idx == 0:
            net = ViT('B_16_imagenet1k', pretrained=True)
            if hasattr(net, 'fc'):
                net.fc = nn.Linear(768, total_cls_nums)
                if n_gpus > 1:
                    net = nn.DataParallel(net, device_ids=list(range(n_gpus)))
        else:
            net = load_full_model('./model_source/target_c2i_{}_2_{}_ViT_DA_last_stage{}.pt'.format(source, target, incre_idx - 1), n_gpus)
        
        net = net.cuda() if n_gpus > 0 else net
        param_group = []
        for p in net.parameters():
            p.requires_grad = True

        model = net.module if hasattr(net, 'module') else net
        
        for k, v in model.named_parameters():
            if k[:2] == 'fc':
                param_group += [{'params': v, 'lr': lr}]
            else:
                param_group += [{'params': v, 'lr': lr}]
        optimizer = optim.SGD(param_group, momentum=momentum, weight_decay=weight_decay)

        best_tar_acc = 0.
        this_stage_save_imgs = True  # Flag for saving images in this stage
        logger.info(f"Starting training for stage {incre_idx + 1}")
        for epoch in range(n_epoches):
            logger.info(f"Epoch {epoch + 1}/{n_epoches}")
            if epoch < 5:
                pred_label_dict, _, best_acc, confi_class_idx, class_centers, confi_class_probs = obtain_label(total_sample_nums, source_weight, target_train_dl, pretrained_net, confi_class_idx, args, epoch)
            else:  # After warm-up stage, obtain pseudo labels with model
                pred_label_dict, _, best_acc, confi_class_idx, class_centers, confi_class_probs = obtain_label(total_sample_nums, source_weight, target_train_dl, model, confi_class_idx, args, epoch)
            
            # Calculate class proportions
            class_counts = {class_idx: 0 for class_idx in confi_class_idx.cpu().numpy()}
            for label in pred_label_dict.values():
                if label in class_counts:
                    class_counts[label] += 1
            total_count = sum(class_counts.values())
            class_proportions = [class_counts[class_idx] / total_count for class_idx in confi_class_idx.cpu().numpy()]
            prop = torch.tensor(class_proportions).unsqueeze(1).cuda()

            iter_target_train = iter(target_train_dl)
            if reply_buffer.exemplar_set:  # If there are prototype images in memory bank
                # Create memory bank dataset
                reply_ds = reply_dataset(images=reply_buffer.exemplar_set, labels=confi_cls_history,
                                         buffer_per_class=reply_buffer_nums, soft_predictions=reply_buffer.soft_pred, return_id = True)
                reply_loader = torch.utils.data.DataLoader(reply_ds, batch_size=batch_size, shuffle=True)
                iter_reply_buffer = iter(reply_loader)

            for iter_idx in range(len(target_train_dl)):
                optimizer.zero_grad()
                model.train()

                data = next(iter_target_train)
                inputs = torch.cat(data[0], dim=0).cuda()
                ground_truths = data[1].repeat(len(data[0]))
                original_sample_idx = data[2]
                augmented_sample_idx = original_sample_idx + total_sample_nums // aug_nums
                sample_idx = torch.cat([original_sample_idx, augmented_sample_idx], dim=0).int()
 
                cls_outputs, feas = model(inputs)
                feas = feas[:, 0, :]
                outputs = cls_outputs[:, confi_class_idx]
                outputs = F.softmax(outputs, dim=1)
                
                ce_sample_idx = sample_idx.numpy().tolist()
                ce_sample_idx = ce_sample_idx[:len(ce_sample_idx) // aug_nums]
                pseudo_labels = []
                for each_idx in ce_sample_idx:
                    pseudo_labels.append(pred_label_dict[each_idx])
                pseudo_labels = torch.tensor(pseudo_labels).cuda()
                ce_loss = loss_ce(cls_outputs[:len(ce_sample_idx), :], pseudo_labels)

                # Consistency contrastive loss
                nums = feas.size(0) // aug_nums
                feas1 = F.normalize(feas[:nums], dim = 1)
                feas2 = F.normalize(feas[nums:], dim = 1)
                features = torch.cat([feas1.unsqueeze(1), feas2.unsqueeze(1)], dim=1)
                contrastive_criterion = SupConLoss()
                con_loss = contrastive_criterion(features)
                con_coeff *=  np.exp(-0.0001)
                       
                ptd_loss = torch.tensor(0.).cuda()
                source_anchors = source_weight[confi_class_idx, :]
                target_anchors = model.fc.weight
                target_anchors = target_anchors[confi_class_idx, :]
                sim_mat = torch.matmul(source_anchors, target_anchors.T)
                new_logits = get_pos_logits(sim_mat.detach(), prop, eps, nav_t)
                s_dist = F.softmax(new_logits, dim=0)
                t_dist = F.softmax(sim_mat/nav_t, dim=1)
                cost_mat = pairwise_cosine_dist(source_anchors, target_anchors)
                com_loss = (s_par*cost_mat*s_dist).sum(0).mean()
                sep_loss = (((1-s_par)*cost_mat*t_dist).sum(1)*prop.squeeze(1)).sum()
                ptd_loss = com_loss + sep_loss
                
                # Target prototype contrastive alignment and reply buffer distillation
                rep_loss = torch.tensor(0.).cuda()
                # Check if there are samples in memory bank
                if reply_buffer.exemplar_set:
                    data_buffer = next(iter_reply_buffer, -1)
                
                    if data_buffer == -1:
                        data_target_iter = iter(reply_loader)
                        re_org_idx, re_org_img, re_org_label, re_org_sp = next(data_target_iter)
                    else:
                        re_org_idx, re_org_img, re_org_label, re_org_sp = data_buffer
    
                    re_org_img = re_org_img.cuda()
                    re_org_label = re_org_label.cuda()
                    re_org_sp = re_org_sp.cuda()
                    reply_outputs, reply_feas = model(re_org_img)
                    reply_feas = reply_feas[:, 0, :]
                    reply_outputs = nn.Softmax(dim=1)(reply_outputs)
                    reply_outputs = torch.log(reply_outputs)
                    soft_pred_loss = torch.sum(-1 * re_org_sp * reply_outputs, dim=1)
                    soft_pred_loss = torch.mean(soft_pred_loss)
                    rep_loss += soft_pred_loss
                    
                total_loss = 1 * ce_loss + con_coeff * con_loss + 1 * ptd_loss + 1 * rep_loss
                total_loss.backward()
                optimizer.step()

            # Check whether to update image prototypes
            if epoch == 5 or (epoch != 0 and epoch % prototypes_update_interval == 0):
                confi_class_idx = list(confi_class_probs.keys()) 
                confi_class_values = list(confi_class_probs.values())
                logger.info(f"Historical prototype classes: {confi_cls_history}")
                logger.info(f"Current prototype classes: {confi_class_idx}")
                
                for confi_idx, confi_class in enumerate(confi_class_idx):
                    if this_stage_save_imgs:  # Update image prototypes
                        if confi_class not in confi_cls_history:
                            confi_cls_history.append(confi_class)
                            confi_cls_value[confi_class] = confi_class_values[confi_idx]
                            imgs = get_one_classes_imgs(target_train_dl, confi_class, pred_label_dict, total_sample_nums, aug_nums)
                            reply_buffer.construct_exemplar_set(imgs, model)
                    
                    else:  # Update stage
                        # Compare prototype classes from previous stage with current confi_class_idx
                        missing_classes = list(set(current_proto_classes[incre_idx]) - set(confi_class_idx))
                        if missing_classes:
                            logger.info(f"Missing classes compared to historical: {missing_classes}")
                        
                        all_previous_classes = set()
                        for k in range(incre_idx):
                            all_previous_classes.update(current_proto_classes[k])
                
                        # Remove prototypes for missing classes
                        for missing_class in missing_classes:
                            if missing_class in confi_cls_history and missing_class not in all_previous_classes:
                                idx_to_remove = confi_cls_history.index(missing_class)
                                del reply_buffer.exemplar_set[idx_to_remove]
                                del reply_buffer.exemplar_indexes[idx_to_remove]
                                del reply_buffer.soft_pred[idx_to_remove]
                                confi_cls_history.remove(missing_class)
                                logger.info(f"Removed prototypes for missing class {missing_class}")
                        
                        history_idx = confi_cls_history.index(confi_class)
                        # Update if current class confidence is greater than or equal to previous record
                        if confi_class_values[confi_idx] >= confi_cls_value[confi_class]:
                            imgs = get_one_classes_imgs(target_train_dl, confi_class, pred_label_dict, total_sample_nums, aug_nums)
                            reply_buffer.update_exemplar_set(imgs, model, history_idx)
                
                logger.info(f"Updated prototype count: {len(reply_buffer.exemplar_set)}")
                current_proto_classes[incre_idx] = confi_class_idx.copy()
                this_stage_save_imgs = False
            
            # Evaluate model performance
            acc_list = val_pclass(model, target_test_loader, total_cls_nums, total_cls_nums, target_test_classes)

            if incre_idx < total_cls_nums // incre_cls_nums:
                acc_list = acc_list[:(incre_idx + 1) * incre_cls_nums]

            if incre_cls_nums >= 10:
                top_ten_classes_mean = acc_list[:10] * 100

            logger.info(f'Epoch {epoch + 1}: ce loss: {ce_loss.item():.3f}, con loss: {con_loss.item():.3f}, ptd loss: {ptd_loss.item():.3f}, rep loss: {rep_loss.item():.3f}')
            logger.info(f"Stage {incre_idx + 1}, Epoch {epoch + 1}: Accuracy list: {np.round(acc_list, 3)}")
            logger.info(f"Stage {incre_idx + 1}, Epoch {epoch + 1}: Mean accuracy: {acc_list.mean():.3f}")
            if incre_cls_nums >= 10:
                logger.info(f"Stage {incre_idx + 1}, Epoch {epoch + 1}: Top ten classes mean accuracy: {top_ten_classes_mean.mean():.3f}")

            _, source_only_acc, target_train_acc, _, _, _  = obtain_label(total_sample_nums, source_weight, target_train_dl, model, confi_class_idx, args, epoch)
            # Get overall loader accuracy
            total_mean_acc = val_office(model, target_test_loader)
            logger.info(f'Total mean accuracy: {total_mean_acc:.3f}')
            if total_mean_acc > best_tar_acc:
                best_tar_acc = total_mean_acc
            torch.save(net, './model_source/target_c2i_{}_2_{}_ViT_DA_last_stage{}.pt'.format(source, target, incre_idx))
            
        best_acc_stages.append(best_tar_acc)
        # Record last accuracy for each stage
        last_total_mean_acc = val_office(model, target_test_loader)
        last_acc_stages.append(last_total_mean_acc)
        top_ten_stages.append(top_ten_classes_mean.mean())
        logger.info(f"Stage {incre_idx + 1} completed. Best accuracy: {best_tar_acc:.3f}")

    logger.info(f'From {cal_dataset.domains[source]} to {imgNet_dataset.domains[target]}, best accuracy of different stages:')
    logger.info(np.round(best_acc_stages, 3))
    logger.info(f'From {cal_dataset.domains[source]} to {imgNet_dataset.domains[target]}, last accuracy of different stages:')
    logger.info(np.round(last_acc_stages, 3))
    logger.info(f'From {cal_dataset.domains[source]} to {imgNet_dataset.domains[target]}, top_ten accuracy of different stages:')
    logger.info(np.round(top_ten_stages, 3))
    logger.info("Training completed!")
