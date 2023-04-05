import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
import torch
import os
import shutil

from .ds_utils import TransformFixMatch, x_u_split

tiny_imagenet_mean = (0.4914, 0.4822, 0.4465)
tiny_imagenet_std = (0.2023, 0.1994, 0.2010)


def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
            and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNet(ImageFolder):
    """Dataset for TinyImageNet-200"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root, split='train', download=False, **kwargs):
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url, self.data_root, filename=self.filename,
            remove_finished=True, md5=self.zip_md5)
        assert 'val' in self.splits
        normalize_tin_val_folder_structure(
            os.path.join(self.dataset_folder, 'val'))


def get_tiny_imagenet(args, root, return_idxs=False):
    transform_labeled = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=tiny_imagenet_mean, std=tiny_imagenet_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=tiny_imagenet_mean, std=tiny_imagenet_std)
    ])
    base_dataset = TinyImageNet(root, split="train", download=True)

    train_labeled_idxs, train_unlabeled_idxs, _ = x_u_split(args, base_dataset.targets, use_validation=False)

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
        calib_dataset = TinyImageNetSSL(root, calib_idxs, train=True,
                                        transform=TransformFixMatch(mean=tiny_imagenet_mean, std=tiny_imagenet_std,
                                                                    dimensionality=64))

    train_labeled_dataset = TinyImageNetSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = TinyImageNetSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=tiny_imagenet_mean, std=tiny_imagenet_std, dimensionality=64),
        return_idxs=return_idxs)

    test_dataset = TinyImageNet(
        root, split="val", transform=transform_val, download=True)

    if "cp" in args and args.cp:
        return train_labeled_dataset, calib_dataset, train_unlabeled_dataset, test_dataset, ds_stats
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, ds_stats


class TinyImageNetSSL(TinyImageNet):
    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False,
                 return_idxs=False):
        if train:
            split = "train"
        else:
            split = "val"
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idxs = return_idxs

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idxs:
            return img, target, index
        return img, target
