import os
from .crop import RandomResizedCrop
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def build_image_dataset(args):
    _mean = IMAGENET_INCEPTION_MEAN
    _std = IMAGENET_INCEPTION_STD

    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])

    if args.dataset == 'cifar100_full':
        dataset_train = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), transform=transform_val, train=False, download=True)
        nb_classes = 100

    elif args.dataset == 'cifar10_full':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'), transform=transform_val, train=False, download=True)
        nb_classes = 10

    elif args.dataset == 'flowers102_full':
        from .flowers102 import Flowers102
        dataset_train = Flowers102(os.path.join(args.data_path, 'flowers102'), split='train', transform=transform_train, download=True)
        dataset_val = Flowers102(os.path.join(args.data_path, 'flowers102'), split='test', transform=transform_val, download=True)
        nb_classes = 102

    elif args.dataset == 'svhn_full':
        from torchvision.datasets import SVHN
        dataset_train = SVHN(os.path.join(args.data_path, 'svhn'), split='train', transform=transform_train, download=True)
        dataset_val = SVHN(os.path.join(args.data_path, 'svhn'), split='test', transform=transform_val, download=True)
        nb_classes = 10

    elif args.dataset == 'food101_full':
        from .food101 import Food101
        dataset_train = Food101(os.path.join(args.data_path, 'food101'), split='train', transform=transform_train, download=True)
        dataset_val = Food101(os.path.join(args.data_path, 'food101'), split='test', transform=transform_val, download=True)
        nb_classes = 101

    elif args.dataset == 'fgvc_aircraft_full':
        from .fgvc_aircraft import FGVCAircraft
        dataset_train = FGVCAircraft(os.path.join(args.data_path, 'fgvc_aircraft'), split='trainval', transform=transform_train, download=True)
        dataset_val = FGVCAircraft(os.path.join(args.data_path, 'fgvc_aircraft'), split='test', transform=transform_val, download=True)
        nb_classes = 100

    elif args.dataset == 'stanford_cars_full':
        from .stanford_cars import StanfordCars
        dataset_train = StanfordCars(os.path.join(args.data_path, 'StanfordCars'), split='train', transform=transform_train, download=True)
        dataset_val = StanfordCars(os.path.join(args.data_path, 'StanfordCars'), split='test', transform=transform_val, download=True)
        nb_classes = 196

    elif args.dataset == 'dtd_full':
        from .dtd import DTD
        dataset_train = DTD(os.path.join(args.data_path, 'dtd'), split='train', transform=transform_train, download=True)
        dataset_val = DTD(os.path.join(args.data_path, 'dtd'), split='test', transform=transform_val, download=True)
        nb_classes = 47

    elif args.dataset == 'oxford_iiit_pet_full':
        from .oxford_iiit_pet import OxfordIIITPet
        dataset_train = OxfordIIITPet(os.path.join(args.data_path, 'oxford_iiit_pet'), split='trainval', transform=transform_train, download=True)
        dataset_val = OxfordIIITPet(os.path.join(args.data_path, 'oxford_iiit_pet'), split='test', transform=transform_val, download=True)
        nb_classes = 37

    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val, nb_classes



