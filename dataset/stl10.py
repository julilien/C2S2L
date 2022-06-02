import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch

from .ds_utils import TransformFixMatch, x_u_split

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2023, 0.1994, 0.2010)


def get_stl10(args, root, return_idxs=False, crop_size=96):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=crop_size,
                              padding=int(crop_size * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    base_dataset = datasets.STL10(root, split="train", download=True)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split(
        args, base_dataset.labels, use_validation=args.validation_scoring)

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
        calib_dataset = STL10SSL(root, calib_idxs, train=True,
                                 transform=TransformFixMatch(mean=stl10_mean, std=stl10_std, dimensionality=96))

    train_labeled_dataset = STL10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = STL10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=stl10_mean, std=stl10_std, dimensionality=96), return_idxs=return_idxs)

    # Validation dataset
    if val_idxs is not None:
        test_dataset = STL10SSL(root, val_idxs, train=True, transform=transform_val)
    else:
        test_dataset = datasets.STL10(
            root, split="test", transform=transform_val, download=False)

    if "cp" in args and args.cp:
        return train_labeled_dataset, calib_dataset, train_unlabeled_dataset, test_dataset, ds_stats
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, ds_stats


class STL10SSL(datasets.STL10):
    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False,
                 return_idxs=False):
        if train:
            split = "train"
        else:
            split = "test"
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idxs = return_idxs
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
