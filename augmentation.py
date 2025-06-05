import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x, seed=None):
        if seed is not None:
            random.seed(seed)
        q = self.base_transform(x)
        if seed is not None:
            random.seed(seed + 1)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x, seed=None):
        if seed is not None:
            random.seed(seed)
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_augmentation(aug_type, normalize=None):

    if not normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    if aug_type == "moco-v2":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8, 0.8, 0.5, 0.2)],
                    p=0.8,  
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees = [-2,2]),
                transforms.RandomPosterize(8, p=0.2),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2])], p=0.5),
                # transforms.AugMix(5,5),           ## While Applying Augmix, comment out the ColorJitter
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    
    elif aug_type == "moco-v1":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                # ImageNetPolicy(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    
    elif aug_type == "weak":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomGrayscale(p=0.05),           ## prob 0.1 works
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  ## all 0.1 works
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    
    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomGrayscale(p=0.05),           ## prob 0.1 works
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  ## all 0.1 works
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return None


class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        image, label, idx = x
        images = [transform(image) for transform in self.transform_list]
        return images, label, idx


def get_augmentation_versions(args):
    transform_list = []
    for version in args.aug_versions:    ## Change the value of augmented versions 
        if version == "s":
            transform_list.append(get_augmentation(args.aug_type))
        elif version == "w":
            transform_list.append(get_augmentation("weak"))
        elif version == "t":
            transform_list.append(get_augmentation("plain"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)
    return transform

class test_augmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        image, label, idx = x
        image = self.transform(image)
        return image, label, idx