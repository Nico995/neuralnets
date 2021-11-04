import os

import pytorch_lightning as pl
from skimage.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from neuralnets.networks.blocks import *
from neuralnets.util.augmentation import *
from neuralnets.util.losses import get_loss_function
from neuralnets.util.tools import *


class AEEncoder(nn.Module):
    """
    AutoEncoder encoder base class
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__()

        self.in_channels = in_channels
        self.features = nn.Sequential()
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation


class AEDecoder(nn.Module):
    """
    AutoEncoder decoder base class
    """

    def __init__(self, out_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__()

        self.out_channels = out_channels
        self.features = nn.Sequential()
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation


class AEEncoder2D(AEEncoder):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps
            # print(f'encoder level {i} out features {in_features} -> {out_features}')

            # convolutional block
            conv_block = Conv2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

    def forward(self, inputs):
        outputs = inputs

        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)
            # print(f'Encoder level {i} output shape: {outputs.shape}')

        return outputs


class AEDecoder2D(AEDecoder):

    def __init__(self, out_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        for i in range(levels - 1):
            in_features = 2 ** (levels - i - 1) * feature_maps
            out_features = 2 ** (levels - i - 2) * feature_maps
            deconv = Deconv2D(in_channels=in_features, out_channels=out_features)
            self.features.add_module('deconv%d' % (i + 1), deconv)

        i += 1
        in_features = out_features
        out_features = out_channels
        deconv = Deconv2D(in_channels=in_features, out_channels=out_features)
        self.features.add_module('deconv%d' % (i + 1), deconv)

        self.output = nn.Tanh()

    def forward(self, inputs):
        outputs = inputs

        for i in range(self.levels):
            outputs = getattr(self.features, 'deconv%d' % (i + 1))(outputs)

        outputs = self.output(outputs)

        return outputs


class AE(pl.LightningModule):
    def __init__(self, input_shape=(1, 256, 256), in_channels=1, feature_maps=64, levels=4,
                 norm='instance', activation='relu', dropout_enc=0.0, dropout_dec=0.0, loss_fn="mse", lr=1e-3,
                 scheduler=None, step_size=1, gamma=0.1, save_checkpoints=True):
        super().__init__()
        # parameters
        if isinstance(input_shape, str):
            self.input_shape = [int(item) for item in input_shape.split(',')]
        else:  # assuming list-like object
            self.input_shape = input_shape
        self.in_channels = int(in_channels)
        self.c = self.in_channels // 2
        self.feature_maps = int(feature_maps)
        self.levels = int(levels)
        self.norm = norm
        self.dropout_enc = float(dropout_enc)
        self.dropout_dec = float(dropout_dec)
        self.activation = activation
        self.loss_fn = get_loss_function(loss_fn)
        self.lr = float(lr)
        self.scheduler_name = scheduler
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.save_checkpoints = bool(save_checkpoints)

        if isinstance(input_shape, str):
            self.input_shape = [int(item) for item in input_shape.split(',')]
        else:  # assuming list-like object
            self.input_shape = input_shape
        self.in_channels = int(in_channels)
        self.c = self.in_channels // 2
        self.feature_maps = int(feature_maps)
        self.levels = int(levels)
        self.norm = norm
        self.dropout_enc = float(dropout_enc)
        self.dropout_dec = float(dropout_dec)
        self.activation = activation


        # print(f'tot levels AE {self.levels}')
        #
        # I did introduced the self.loss property because using MSELoss would yield error:
        # TypeError: forward() got an unexpected keyword argument 'w'
        self.loss = loss_fn
        #
        self.loss_fn = get_loss_function(loss_fn)
        self.lr = float(lr)
        self.scheduler_name = scheduler
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.save_checkpoints = bool(save_checkpoints)

        self.train_batch_id = 0
        self.val_batch_id = 0

    def forward(self, x):
        # encoder path
        encoder_outputs = self.encoder(x)
        # print(f'Encoder output shape {encoder_outputs.shape}')
        # decoder path
        decoder_outputs = self.decoder(encoder_outputs)

        return decoder_outputs

    def training_step(self, batch, batch_idx):
        # get data
        x = batch

        # forward prop
        y_pred = self(x)
        # compute loss
        loss = self.loss_fn(y_pred, x)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_result(y_pred.detach().cpu().numpy(), prefix='train/pred')
            self._log_result(x.detach().cpu().numpy(), prefix='train/truth')

        # compute mse
        mse = mean_squared_error(x.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        self.log('train/mse', mse, prog_bar=True)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # get data
        x = batch

        # forward prop
        y_pred = self(x)

        # compute loss
        loss = self.loss_fn(y_pred, x)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_result(y_pred.detach().cpu().numpy(), prefix='val/pred')
            self._log_result(x.detach().cpu().numpy(), prefix='val/truth')

        # compute mse
        mse = mean_squared_error(x.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        self.log('val/mse', mse, prog_bar=True)
        self.log('val/loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        # get data
        x = batch

        # forward prop
        y_pred = self(x)

        # compute loss
        loss = self.loss_fn(y_pred, x)

        # compute mse
        mse = mean_squared_error(x.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        self.log('test/mse', mse, prog_bar=True)
        self.log('test/loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer_dict = {"optimizer": optimizer}
        if self.scheduler_name == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, 'max', patience=self.step_size, factor=self.gamma)
            optimizer_dict.update({"lr_scheduler": scheduler, "monitor": 'val/mIoU'})
        elif self.scheduler_name == 'step_lr':
            scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            optimizer_dict.update({"lr_scheduler": scheduler})
        return optimizer_dict

    def on_epoch_start(self):
        set_seed(rnd.randint(100000))
        if self.trainer.num_training_batches > 0:
            self.train_batch_id = rnd.randint(self.trainer.num_training_batches)
        if len(self.trainer.num_val_batches) > 0:
            self.val_batch_id = rnd.randint(self.trainer.num_val_batches[0])

    def on_epoch_end(self):
        if self.save_checkpoints:
            torch.save(self.state_dict(), os.path.join(self.logger.log_dir, 'checkpoints', 'epoch=%d-step=%d.ckpt' %
                                                       (self.current_epoch, self.global_step)))

    def _log_result(self, y_pred, prefix='train'):
        # get the tensorboard summary writer
        tensorboard = self.logger.experiment

        tensorboard.add_images(prefix, y_pred, global_step=self.current_epoch, dataformats='NCHW')


class AE2D(AE):
    def __init__(self, input_shape=(1, 256, 256), in_channels=1, feature_maps=64, levels=4,
                 norm='instance', activation='relu', dropout_enc=0.0, dropout_dec=0.0, loss_fn="mse", lr=1e-3,
                 scheduler=None, step_size=1, gamma=0.1, save_checkpoints=True):
        super().__init__(input_shape=input_shape, in_channels=in_channels, feature_maps=feature_maps, levels=levels,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, scheduler=scheduler, step_size=step_size, gamma=gamma,
                         save_checkpoints=save_checkpoints)

        # print(f'tot levels ae2d {levels}')
        # encoder path
        self.encoder = AEEncoder2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                   norm=self.norm, dropout=self.dropout_enc, activation=self.activation)

        # decoder path
        self.decoder = AEDecoder2D(out_channels=in_channels, feature_maps=self.feature_maps, levels=self.levels, norm=self.norm,
                                   dropout=self.dropout_dec, activation=self.activation)
