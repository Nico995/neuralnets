"""
    This is a script that illustrates training a 2D U-Net
"""

import argparse
import glob
import os
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import tifffile
import torch.nn
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from neuralnets.data.datamodule import DataModule
from neuralnets.networks.unet import UNet2D
from neuralnets.util.augmentation import *
from neuralnets.util.common import create_module
from neuralnets.util.io import print_frm
from neuralnets.util.tools import parse_params


if __name__ == '__main__':
    freeze_support()

    # ---------- PARAMETERS PARSING ---------- #
    # Parsing arguments
    print_frm('Parsing command-line arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config file", type=str, required=True)
    args = parser.parse_args()
    config_name = args.config.split('/')[-1].split('.')[0]

    # Loading parameters parameters from yaml config file
    print_frm(f"Loading parameters parameters from {args.config} config file")
    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # ---------- DATA LOADING ---------- #
    # Seed all random processes
    print_frm(f"Seeding all stochastic processes with seed {params['seed']}")
    seed_everything(params['seed'])

    # Loading data from filesystem
    print_frm("Initializing datasets")

    # Add transforms to parameters
    transforms = []
    if 'transforms' in params:
        for t in params['transforms']:
            transforms.append(create_module(t))

        params['datamodule']['dataset_hparams']['train']['transform'] = Compose(transforms)

    ds = create_module(params['dataset'])
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    # ---------- DOWNSTREAM MODEL LOADING ---------- #
    print_frm("Initializing neural network")
    net = create_module(params['model'])

    # load pretraned weights
    if 'pretrained_model' in params:
        state_dict_file = glob.glob(os.path.join(params['pretrained_model'], "*.ckpt"))[0]
        state_dict = torch.load(state_dict_file)['state_dict']
        # print(state_dict.keys())
        if 'drop_parameters' in params:
            for param_name in params['drop_parameters']:
                del state_dict[param_name]

        # load weights in network
        net.load_state_dict(state_dict, strict=False)
        # TODO: implement weights freezing
        # if params['freeze_weights']:
        #     for name, param in net.named_parameters():
        #         if not name.startswith('decoder.output'):
        #             param.requires_grad = False

    # ---------- TRAINER ---------- #

    # add root dir with the same name as the config file
    params['trainer']['default_root_dir'] = os.path.join('logs', config_name)

    trainer = pl.Trainer(**params['trainer'])
    results = trainer.predict(net, dataloaders=dl, return_predictions=True)

    slides = []
    rows = []
    images = []
    r = 1
    cc = 0
    for i, pred in enumerate(results, 1):
        pred = pred.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1).astype(np.uint8)*255
        images.append(np.moveaxis(pred, 0, -1))

        print(i)
        if i % 8 == 0:

            print(f'concat row {r}')
            rows.append(np.concatenate(images, axis=1))
            images = []

            if r % 6 == 0:
                print(f'concat slide {r % 6}')
                for i, row in enumerate(rows):
                    print(f'row {i} shape {row.shape}')
                slides.append(np.concatenate(rows, axis=0))
                rows = []

            r += 1

    print(f'we have {len(slides)} slides')
    part = np.stack(slides[:-1], axis=0)
    print(part.shape)
    print(part.dtype)

    tifffile.imwrite('pred.tif', part)

    # exit()
    # plt.imshow(part, cmap='gray')
    # plt.savefig(f'concat')


