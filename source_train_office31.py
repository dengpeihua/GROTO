# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.transforms as transforms
import argparse
import time
import torch.nn.functional as F
from source_datasets import FileListDataset
from os.path import join
from net.pytorch_pretrained_vit import ViT

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--source', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

    args = parser.parse_args()

    return args


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


def val_net(net, test_loader):
    net.eval()

    correct = 0
    total = 0

    gt_list = []
    p_list = []

    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        gt_list.append(labels.cpu().numpy())
        with torch.no_grad():
            outputs = net(inputs, return_features = False)
        output_prob = F.softmax(outputs, dim=1).data
        p_list.append(output_prob[:, 1].detach().cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

    acc = 100. * correct.item() / total

    return acc


def get_source_centers(source_loader, net):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(source_loader)
        for _ in range(len(source_loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, feas = net(inputs)
            feas = feas[:, 0, :]

            # 初始化累积特征和输出
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = all_fea.float().cpu()


    for i in range(all_output.size(1)):
        cnt = 0
        for idx, label in enumerate(all_label):
            if label == i:
                if cnt == 0:
                    source_cls_proto = all_fea[idx]
                else:
                    source_cls_proto += all_fea[idx]
                cnt += 1
        source_cls_proto = source_cls_proto / cnt
        if i == 0:
            total_source_protos = source_cls_proto.unsqueeze(0)
        else:
            total_source_protos = torch.cat((total_source_protos, source_cls_proto.unsqueeze(0)), 0)
    return total_source_protos.cuda()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.MSE_loss = nn.MSELoss().cuda()
        self.loss = nn.CrossEntropyLoss().cuda()

    def train_half(self, model, optimizer, x_val, y_val):
        model.train()
        output, _ = model(x_val)
        hloss = self.loss(output, y_val)
        optimizer.zero_grad()
        hloss.backward()
        optimizer.step()
        return hloss.item()

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        batch_size = self.args.batchsize
        epochs = self.args.max_epoch

        best_acc = 0.
        cls_nums = 31

        source_model_root = './model_source'
        if not os.path.exists(source_model_root):
            os.mkdir(source_model_root)

        net = ViT('B_16_imagenet1k', pretrained=True)
        net.fc = nn.Linear(768, cls_nums)
        if n_gpus > 1:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)
        
        param_group = []
        for k, v in net.named_parameters():
            if k[:2] == 'fc':
                param_group += [{'params': v, 'lr': args.lr * 10}]
            else:
                param_group += [{'params': v, 'lr': args.lr}]

        optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4)
        source = args.source
        source_classes = [i for i in range(cls_nums)]
        my_dataset = Dataset(
            path='./dataset/office31',
            domains=['amazon', 'dslr', 'webcam'],
            files=[
                'amazon.txt',
                'dslr.txt',
                'webcam.txt'
            ],
            prefix='./dataset/office31')
        source_domain = my_dataset.domains[source]
        source_file = my_dataset.files[source]
        print(
            'office31_{}_all_train'.format(source_domain) + time_stamp_launch + 'model : vit  lr: %s' % args.lr)

        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        source_train_ds = FileListDataset(list_path=source_file, path_prefix=my_dataset.prefixes[source],
                                          transform=transform_train,
                                          filter=(lambda x: x in source_classes), return_id=False)
        train_loader = torch.utils.data.DataLoader(source_train_ds, batch_size=batch_size, shuffle=True,
                                                   num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        source_val_ds = FileListDataset(list_path=source_file, path_prefix=my_dataset.prefixes[source],
                                        transform=transform_test,
                                        filter=(lambda x: x in source_classes), return_id=False)
        val_loader = torch.utils.data.DataLoader(source_val_ds, batch_size=batch_size, shuffle=False,
                                                 num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        for i in range(epochs):
            running_loss = []
            net.train()

            for j, (img_data, img_label) in enumerate(train_loader):
                img_data = img_data.to(device)
                img_label = img_label.to(device)
                r_loss = self.train_half(net, optimizer, img_data, img_label)
                running_loss += [r_loss]
            avg_loss = np.mean(running_loss)
            if i % 3 == 0:
                acc = val_net(net, val_loader)
                print("Epoch %d running_loss=%.3f, acc=%.3f" % (i + 1, avg_loss, acc))

                if acc > best_acc:
                    best_acc = acc
                    best_model_path = './model_source/' + time_stamp_launch + 'source-free-OH_{}_ce_singe_gpu_vit_B_16_best.pkl'.format(my_dataset.domains[source])
                    torch.save(net, best_model_path)

        print(f"Finished Training. Best model saved to {best_model_path}")

        best_model = torch.load(best_model_path)
        source_loader = torch.utils.data.DataLoader(source_train_ds, batch_size=batch_size, shuffle=False, num_workers=2 * n_gpus if n_gpus <= 2 else 2)
        source_centers_mean = get_source_centers(source_loader, best_model)
        torch.save(source_centers_mean, './model_source/' + time_stamp_launch + 'source_centers_mean.pkl')

                               
if __name__ == '__main__':
    args = arg_parser()
    trainer = Trainer(args)
    trainer.train()