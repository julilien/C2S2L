import math

import numpy as np
from torchvision import transforms

from .randaugment import RandAugmentMC


def x_u_split(args, labels, use_validation=False, val_fraction=0.2):
    label_per_class = args.num_labeled // args.num_classes

    labels = np.array(labels)
    labeled_idx = []

    if use_validation:
        # Val fraction is the fraction of labeled instances to be used as validation instances
        assert 0 <= val_fraction <= 1

        val_idx = []
        val_label_per_class = int((len(labels) * val_fraction) // args.num_classes)
        print("Num validation instances:", val_label_per_class)
    else:
        val_idx = None

    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        if use_validation:
            complete_idx = np.random.choice(idx, label_per_class + val_label_per_class, False)
            tmp_val_idx = complete_idx[label_per_class:]
            val_idx.extend(tmp_val_idx)
            idx = complete_idx[:label_per_class]
        else:
            idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx, val_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, dimensionality=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=dimensionality,
                                  padding=int(dimensionality * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=dimensionality,
                                  padding=int(dimensionality * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
