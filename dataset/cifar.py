import logging

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch

from dataset.ds_utils import TransformFixMatch, x_u_split
from dataset.imbalance import get_img_num_per_cls, imbalance_cifar

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root, return_idxs=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    if args.imbalanced:
        base_dataset = imbalance_cifar(base_dataset, 10, args.imbalance_factor, args.seed)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split(
        args, base_dataset.targets, use_validation=args.validation_scoring)

    # Get dataset stats
    target_tensor = torch.Tensor(np.array(base_dataset.targets)[train_labeled_idxs]).int()
    bin_counts = torch.bincount(target_tensor)
    ds_stats = bin_counts / torch.sum(bin_counts)

    calib_dataset = None
    if "cp" in args and args.cp:
        np.random.seed(args.seed)
        np.random.shuffle(train_labeled_idxs)
        num_calib_instances = int(args.calibration_split * len(train_labeled_idxs))
        calib_idxs = train_labeled_idxs[:num_calib_instances]
        train_labeled_idxs = train_labeled_idxs[num_calib_instances:]

        # Apply same transform as for the weak part
        calib_idxs = np.unique(calib_idxs)
        calib_dataset = CIFAR10SSL(root, calib_idxs, train=True,
                                   # transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
                                   transform=transform_val
                                   )

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std), return_idxs=return_idxs)

    # Validation dataset
    if val_idxs is not None:
        test_dataset = CIFAR10SSL(root, val_idxs, train=True, transform=transform_val)
    else:
        test_dataset = datasets.CIFAR10(
            root, train=False, transform=transform_val, download=False)

    if "cp" in args and args.cp:
        return train_labeled_dataset, calib_dataset, train_unlabeled_dataset, test_dataset, ds_stats
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, ds_stats


def get_cifar100(args, root, return_idxs=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    if args.imbalanced:
        base_dataset = imbalance_cifar(base_dataset, 100, args.imbalance_factor, args.seed)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split(args, base_dataset.targets,
                                                                   use_validation=args.validation_scoring)

    # Get dataset stats
    target_tensor = torch.Tensor(np.array(base_dataset.targets)[train_labeled_idxs]).int()
    ds_stats = torch.bincount(target_tensor) / torch.sum(target_tensor)

    calib_dataset = None
    if "cp" in args and args.cp:
        np.random.seed(args.seed)
        np.random.shuffle(train_labeled_idxs)
        num_calib_instances = int(args.calibration_split * len(train_labeled_idxs))
        calib_idxs = train_labeled_idxs[:num_calib_instances]
        train_labeled_idxs = train_labeled_idxs[num_calib_instances:]

        # Apply same transform as for the weak part
        calib_idxs = np.unique(calib_idxs)
        calib_dataset = CIFAR100SSL(root, calib_idxs, train=True,
                                   # transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
                                   transform=transform_val
                                   )

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std),
        return_idxs=return_idxs)

    # Validation dataset
    if val_idxs is not None:
        test_dataset = CIFAR100SSL(root, val_idxs, train=True, transform=transform_val)
    else:
        test_dataset = datasets.CIFAR100(
            root, train=False, transform=transform_val, download=False)

    if "cp" in args and args.cp:
        return train_labeled_dataset, calib_dataset, train_unlabeled_dataset, test_dataset, ds_stats
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, ds_stats


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idxs=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idxs = return_idxs
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # Shape before: [w/h, w/h, n_channels]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Shape afterwards: [n_channels, w/h, w/h]
        if self.return_idxs:
            return img, target, index
        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idxs=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idxs = return_idxs
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idxs:
            return img, target, index
        return img, target
