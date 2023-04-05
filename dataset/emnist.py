import logging

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch

from dataset.ds_utils import TransformFixMatch, x_u_split

logger = logging.getLogger(__name__)

mnist_mean = 0.1307
mnist_stddev = 0.3081
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

SPLIT = "byclass"
NUM_EMNIST_CLASSES = 62


def get_emnist(args, root, return_idxs=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_stddev)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_stddev)
    ])
    base_dataset = datasets.EMNIST(root, split=SPLIT, train=True, download=True)
    # base_dataset.targets -= 1

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

        unique_instances = np.unique(train_labeled_idxs)
        num_calib_instances = int(args.calibration_split * len(unique_instances))
        np.random.shuffle(unique_instances)

        tmp_calib_idxs = unique_instances[:num_calib_instances]
        tmp_train_idx = unique_instances[num_calib_instances:]

        calib_idxs = train_labeled_idxs[np.isin(train_labeled_idxs, tmp_calib_idxs)]
        train_labeled_idxs = train_labeled_idxs[np.isin(train_labeled_idxs, tmp_train_idx)]

        # Apply same transform as for the weak part
        calib_idxs = np.unique(calib_idxs)
        calib_dataset = EMNIST10SSL(root, calib_idxs, train=True,
                                    transform=TransformFixMatch(mean=mnist_mean, std=mnist_stddev))

    train_labeled_dataset = EMNIST10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = EMNIST10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=mnist_mean, std=mnist_stddev), return_idxs=return_idxs)

    # Validation dataset
    if val_idxs is not None:
        test_dataset = EMNIST10SSL(root, val_idxs, train=True, transform=transform_val)
    else:
        # test_dataset = datasets.EMNIST(root, split=SPLIT, train=False, transform=transform_val, download=True)
        test_dataset = EMNIST10SSL(root, None, train=False, transform=transform_val, download=True)

    if "cp" in args and args.cp:
        return train_labeled_dataset, calib_dataset, train_unlabeled_dataset, test_dataset, ds_stats
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, ds_stats


class EMNIST10SSL(datasets.EMNIST):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, return_idxs=False):
        super().__init__(root, train=train, split=SPLIT,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idxs = return_idxs
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        # self.targets -= 1

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # Shape before: [w/h, w/h, n_channels]
        img = Image.fromarray(img.numpy()).convert('RGB')
        # t2 = transforms.ToPILImage()
        # img = t2(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Shape afterwards: [n_channels, w/h, w/h]
        if self.return_idxs:
            return img, target, index
        return img, target
