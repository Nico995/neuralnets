"""
    This is a script that illustrates training a 2D U-Net
"""

import argparse
import os
from multiprocessing import freeze_support

import pytorch_lightning as pl
import yaml
from pytorch_lightning import seed_everything

from neuralnets.data.datamodule import DataModule
from neuralnets.util.augmentation import *
from neuralnets.util.common import create_module
from neuralnets.util.io import print_frm

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

    dm = DataModule(**params['datamodule'])

    # ---------- DOWNSTREAM MODEL LOADING ---------- #
    print_frm("Initializing neural network")
    net = create_module(params['model'])

    # load pretraned weights
    if 'pretrained_params' in params:
        net.load_from_pretext(**params['pretrained_params'])

        # TODO: implement weights freezing
        # if params['freeze_weights']:
        #     for name, param in net.named_parameters():
        #         if not name.startswith('decoder.output'):
        #             param.requires_grad = False

    # ---------- CALLBACKS ---------- #
    callbacks = []
    if 'callbacks' in params:
        for callback_params in params['callbacks']:
            callbacks.append(create_module(callback_params))
        params['trainer']['callbacks'] = callbacks

    # ---------- TRAINER ---------- #

    # add root dir with the same name as the config file
    params['trainer']['default_root_dir'] = os.path.join('logs', config_name)

    trainer = pl.Trainer(**params['trainer'])
    print_frm(f'training on GPUs {trainer.root_gpu}')

    if 'zero_shot' not in params or not params['zero_shot']:
        trainer.fit(net, datamodule=dm)
        net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    else:
        dm.setup('fit')
        trainer.validate(net, datamodule=dm)

    dm.setup(stage="test")
    trainer.test(net, datamodule=dm)

    # ---------- POST ---------- #
    print_frm("Saving running configuration")
    with open(args.config) as f_in:
        params = yaml.load(f_in, Loader=yaml.FullLoader)
        with open(os.path.join(trainer.log_dir, 'config.yaml'), 'w') as f_out:
            yaml.dump(params, f_out)
