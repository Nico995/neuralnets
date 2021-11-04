import sys

from neuralnets.data.datasets import *
from neuralnets.util.losses import *
from neuralnets.util.metrics import *
from neuralnets.util.augmentation import *
from neuralnets.networks.unet import UNet2D, UNetRotation2D, UNetJigsaw2D
from torch.nn import ReLU, LeakyReLU, GELU, Tanh
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import *
from pytorch_lightning.callbacks import *
from torchmetrics import IoU, MeanSquaredError


def create_module(hparams):
    return getattr(sys.modules[__name__], hparams.pop(f'name'))(**hparams)
