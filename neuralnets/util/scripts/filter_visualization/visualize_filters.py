import argparse
import os
from multiprocessing import freeze_support

import PIL.Image
import pytorch_lightning as pl
import torch.nn.functional
import yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torchvision.transforms import Compose, ToTensor, Normalize
from albumentations.augmentations.geometric.rotate import Rotate

from neuralnets.data.datamodule import DataModule
from neuralnets.networks.unet import UNet2D
from neuralnets.util.augmentation import *
from neuralnets.util.common import create_module
from neuralnets.util.io import print_frm
from neuralnets.util.tools import parse_params

from skimage.transform import rotate


if __name__ == '__main__':

    # ---------- PARAMETERS PARSING ---------- #

    with open('visualize_filters.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # ---------- DATA LOADING ---------- #
    #
    # tns = ToTensor()
    #
    # data = tns(np.array(PIL.Image.open('/home/nicola/data/electron/sample.jpg'))).unsqueeze(0).unsqueeze(0)[:, :, :128, :128]
    # data = data.float()

    tns = ToTensor()
    angle = 90
    rot = Rotate(limit=[angle, angle])

    data = np.array(PIL.Image.open('/home/nicola/data/electron/sample.jpg'))[:128, :128]
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(data, cmap='gray')
    data = rot(image=data)['image']
    ax[1].imshow(data, cmap='gray')
    plt.savefig('rotations')
    data = tns(data).unsqueeze(0).unsqueeze(0)
    data = data.float()
    # ---------- MODEL LOADING ---------- #

    net = create_module(params['model'])
    state_dict = torch.load(params['pretrained_model'])['state_dict']
    net.load_state_dict(state_dict, strict=False)

    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    layer = 'encoder.features.convblock4.conv1'
    dict(net.named_modules())[layer].register_forward_hook(get_activation(layer))

    # ---------- FORWARD PASS ---------- #
    output = net(data)
    print(f'predicted rotation: {torch.argmax(torch.nn.functional.softmax(output, dim=-1), dim=-1)}degrees')

    # ---------- FEATURE MAP VIZ ---------- #

    act = activation[layer].squeeze()
    print(f'activation map shape: {act.shape}')
    row = int(np.ceil(np.sqrt(act.shape[0])))

    fig, ax = plt.subplots(row, row, figsize=(16, 16))
    ax = ax.ravel()

    for i in range(act.shape[0]):
        ax[i].imshow(act[i], cmap='gray')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_aspect('equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('activations')
