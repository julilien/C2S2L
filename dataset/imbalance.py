from typing import Optional
import numpy as np

def get_img_num_per_cls(cifar_version: int, imb_factor=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = cifar_version
    img_max = 5000 if cifar_version == 10 else 500
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * ((1. / imb_factor) ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def imbalance_cifar(dataset, cifar_version: int, imb_factor: Optional[float] = None, seed: int = 42):
    imgs_per_cls = get_img_num_per_cls(cifar_version, imb_factor)

    targets = dataset.targets
    data = dataset.data

    # Shuffle data and targets equally
    np.random.seed(seed)
    p = np.random.permutation(len(data))
    data = data[p]
    targets = np.array(targets)[p]

    # dataset based on imgs per class
    new_data = []
    new_targets = []
    for cls_idx in range(cifar_version):
        cls_data = data[targets == cls_idx]
        cls_data = cls_data[:imgs_per_cls[cls_idx]]
        new_data.append(cls_data)
        new_targets.append([cls_idx] * imgs_per_cls[cls_idx])

    new_data = np.concatenate(new_data)
    new_targets = np.concatenate(new_targets)

    # Shuffle data and targets equally
    np.random.seed(seed)
    p = np.random.permutation(len(new_data))
    new_data = new_data[p]
    new_targets = new_targets[p]

    dataset.data = new_data
    dataset.targets = new_targets

    return dataset