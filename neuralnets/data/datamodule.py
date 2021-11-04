from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from neuralnets.util.common import create_module
from neuralnets.util.io import print_frm


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_hparams, workers):
        super().__init__()
        self.dataset_hparams = dataset_hparams
        self.workers = workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        print()
        # Load the dataset
        if stage == 'fit':
            self.ds_train = create_module(self.dataset_hparams['train'])
            print_frm(f'Train data size: {len(self.ds_train)}')
            self.ds_val = create_module(self.dataset_hparams['val'])
            print_frm(f'Val data size: {len(self.ds_val)}')
        elif stage == 'test' or stage is None:
            self.ds_test = create_module(self.dataset_hparams['test'])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.dataset_hparams['train']['batch_size'],
                          num_workers=self.workers['train'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.dataset_hparams['val']['batch_size'],
                          num_workers=self.workers['val'], pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.dataset_hparams['test']['batch_size'],
                          num_workers=self.workers['test'], pin_memory=True)


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, dataset_hparams, workers):
        super().__init__()
        self.dataset_hparams = dataset_hparams
        self.workers = workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        # Load the dataset
        self.ds_train = create_module(self.dataset_hparams['train'])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.dataset_hparams['train']['batch_size'],
                          num_workers=self.workers['train'], shuffle=True)

