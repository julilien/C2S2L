from .cifar import get_cifar10, get_cifar100
from .imagenet import get_tiny_imagenet
from .stl10 import get_stl10
from .svhn import get_svhn, get_svhn_with_extra

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'svhn': get_svhn,
                   'svhn_extra': get_svhn_with_extra,
                   'stl10': get_stl10,
                   'tinyimagenet': get_tiny_imagenet}
