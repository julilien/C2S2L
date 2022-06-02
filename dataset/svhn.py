import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch

from .ds_utils import TransformFixMatch, x_u_split

svhn_mean = (0.4914, 0.4822, 0.4465)
svhn_std = (0.2023, 0.1994, 0.2010)


def get_svhn(args, root, return_idxs=False):
    return load_svhn(args, root, with_extra=False, return_idxs=return_idxs)


def get_svhn_with_extra(args, root, return_idxs=False):
    return load_svhn(args, root, with_extra=True, return_idxs=return_idxs)


def load_svhn(args, root, with_extra=False, return_idxs=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    base_dataset = datasets.SVHN(root, split="train", download=True)

    # Add extra dataset to base_dataset
    if with_extra:
        extra_dataset = datasets.SVHN(root, split="extra", download=True)
        base_dataset.data = np.concatenate((base_dataset.data, extra_dataset.data), axis=0)
        base_dataset.labels = np.concatenate((base_dataset.labels, extra_dataset.labels), axis=0)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split(args, base_dataset.labels,
                                                                   use_validation=args.validation_scoring)

    # Get dataset stats
    target_tensor = torch.Tensor(np.array(base_dataset.labels)[train_labeled_idxs]).int()
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
        calib_dataset = SVHNSSL(root, calib_idxs, train=True,
                                transform=TransformFixMatch(mean=svhn_mean, std=svhn_std), with_extra=with_extra)

    train_labeled_dataset = SVHNSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, with_extra=with_extra)

    train_unlabeled_dataset = SVHNSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=svhn_mean, std=svhn_std), with_extra=with_extra, return_idxs=return_idxs)

    # Validation dataset
    if val_idxs is not None:
        test_dataset = SVHNSSL(root, val_idxs, train=True, transform=transform_val)
    else:
        test_dataset = datasets.SVHN(
            root, split="test", transform=transform_val, download=True)

    if "cp" in args and args.cp:
        return train_labeled_dataset, calib_dataset, train_unlabeled_dataset, test_dataset, ds_stats
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, ds_stats


class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False,
                 with_extra=False, return_idxs=False):
        extra_ds = None
        if train:
            split = "train"
            if with_extra:
                extra_ds = datasets.SVHN(root, split="extra", transform=transform, target_transform=target_transform,
                                         download=download)
        else:
            split = "test"

        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.return_idxs = return_idxs

        if extra_ds is not None and with_extra:
            self.data = np.concatenate((self.data, extra_ds.data), axis=0)
            self.labels = np.concatenate((self.labels, extra_ds.labels), axis=0)

        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idxs:
            return img, target, index
        return img, target
