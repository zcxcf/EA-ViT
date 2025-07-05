import os
import PIL
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


train = torchvision.datasets.SVHN(root='SVHN',split='train',download=True)
test = torchvision.datasets.SVHN(root='SVHN',split='test',download=True)
