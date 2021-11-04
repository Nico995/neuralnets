import os

import pytorch_lightning as pl
import torch
from torch.nn import Sequential

import neuralnets.util.common
from neuralnets.networks.blocks import *
# from neuralnets.util.common import create_module
from neuralnets.util.tools import *
from neuralnets.util.visualization import overlay, COLORS


class UNetEncoder(nn.Module):
    """
    U-Net encoder base class

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional dense_blocks: specify use of dense blocks
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu',
                 dense_blocks=False):
        super().__init__()

        self.features = nn.Sequential()
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation
        self.dense_blocks = dense_blocks


class UNetDecoder(nn.Module):
    """
    U-Net decoder base class

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu'):
        super().__init__()

        self.features = nn.Sequential()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.residual_connections = residual_connections
        self.norm = norm
        self.dropout = dropout
        self.activation = activation


class UNetLinearDecoder(nn.Module):
    def __init__(self, input_shape, levels, feature_maps, out_channels):
        super(UNetLinearDecoder, self).__init__()
        self.in_res = int(input_shape[1] / 2 ** levels)
        self.in_features = 2 ** levels * feature_maps

        self.output = Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.in_res ** 2 * self.in_features, out_features=out_channels),
        )

    def forward(self, x, dummy):
        return None, self.output(x)


class UNetEncoder2D(UNetEncoder):
    """
    2D U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock2D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, norm=norm,
                                           dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class UNetEncoder3D(UNetEncoder):
    """
    3D U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock3D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock3D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, norm=norm,
                                           dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class UNetDecoder2D(UNetDecoder):
    """
    2D U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = UNetConvBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = UNetConvBlock2D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            elif self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class UNetDecoder3D(UNetDecoder):
    """
    3D U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = UNetConvBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = UNetConvBlock3D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv3d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            elif self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class DenseUNetEncoder2D(UNetEncoder):
    """
    2D Dense U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu',
                 num_layers=4, k=16, bn_size=2):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # dense convolutional block
            conv_block = DenseBlock2D(in_features, out_features, num_layers, k, bn_size, norm=norm, dropout=dropout,
                                      activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = DenseBlock2D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, num_layers, k,
                                        bn_size, norm=norm, dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class DenseUNetEncoder3D(UNetEncoder):
    """
    3D Dense U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu',
                 num_layers=4, k=16, bn_size=2):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # dense convolutional block
            conv_block = DenseBlock3D(in_features, out_features, num_layers, k, bn_size, norm=norm, dropout=dropout,
                                      activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = DenseBlock3D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, num_layers, k,
                                        bn_size, norm=norm, dropout=dropout, activation=activation)

    def forward(self, inputs):
        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class DenseUNetDecoder2D(UNetDecoder):
    """
    2D Dense U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu', num_layers=4, k=16, bn_size=2):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = DenseBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = DenseBlock2D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            elif self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class DenseUNetDecoder3D(UNetDecoder):
    """
    3D Dense U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional residual_connections: use residual connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional num_layers: number of dense layers
    :param optional k: how many filters to add each layer
    :param optional bn_size: multiplicative factor for number of bottle neck layers
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', dropout=0.0, activation='relu', num_layers=4, k=16, bn_size=2):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels,
                         skip_connections=skip_connections, residual_connections=residual_connections, norm=norm,
                         dropout=dropout, activation=activation)

        self.num_layers = num_layers
        self.k = k
        self.bn_size = bn_size

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections and not residual_connections:
                conv_block = DenseBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = DenseBlock3D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                          num_layers, k, bn_size, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv3d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            if self.residual_connections:
                outputs = encoder_outputs[self.levels - i - 1] + \
                          getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # residual connection
            elif self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[self.levels - i - 1],
                                                                       outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class UNet(pl.LightningModule):

    def __init__(self, loss_hparams, optimizer_hparams, metric_hparams, scheduler_hparams, input_shape,
                 with_labels=True,
                 log_images_type=None, save_checkpoints=False, return_features=False):
        super(UNet, self).__init__()

        self.example_input_array = torch.zeros((1,) + tuple(input_shape), dtype=torch.float32)
        self.with_labels = with_labels
        self.log_images_type = log_images_type
        self.save_checkpoints = save_checkpoints
        self.return_features = return_features

        # ---------- optimizer, loss & metric ---------- #
        # Save optimizer params
        self.optimizer_hparams = optimizer_hparams
        self.optimizer_hparams['params'] = self.parameters()

        # Save scheduler params
        self.scheduler_hparams = scheduler_hparams

        # Create loss function
        self.loss_module = neuralnets.util.common.create_module(loss_hparams)

        # Create metric
        self.metric_name = metric_hparams['name']
        self.metric = neuralnets.util.common.create_module(metric_hparams)

        # ---------- everything else ---------- #
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        # self.save_hyperparameters()

    def forward(self, x):
        # contractive path
        aux, out = self.encoder(x)

        # expansive path
        aux, out = self.decoder(out, aux)

        if self.return_features:
            return out, aux[-1]
        else:
            return out

    def configure_optimizers(self):
        optimizer_dict = {}

        # optimizer
        optimizer = neuralnets.util.common.create_module(self.optimizer_hparams)
        self.scheduler_hparams['optimizer'] = optimizer
        optimizer_dict['optimizer'] = optimizer

        if 'Plateau' in self.scheduler_hparams['name']:
            optimizer_dict['monitor'] = self.scheduler_hparams.pop('monitor')

        # scheduler
        scheduler = neuralnets.util.common.create_module(self.scheduler_hparams)
        optimizer_dict['lr_scheduler'] = scheduler

        return optimizer_dict

    def training_step(self, batch, batch_idx):
        imgs, labels, w = self._unpack(batch)

        # print(torch.unique(labels))

        # print(labels)
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log(f"train/{self.metric_name}", metric, prog_bar=True)
        self.log(f"train/loss", loss)

        if batch_idx == 0:
            if self.log_images_type == 'plain':
                self._log_result(preds.detach().cpu().numpy(), prefix='train/preds')
                self._log_result(labels.detach().cpu().numpy(), prefix='train/labels')
            elif self.log_images_type == 'masked':
                self._log_predictions(imgs, labels, preds, prefix='train')

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, w = self._unpack(batch)

        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log(f"val/{self.metric_name}", metric, prog_bar=True)
        self.log(f"val/loss", loss)

        if batch_idx == 0:
            if self.log_images_type == 'plain':
                self._log_result(preds.detach().cpu().numpy(), prefix='val/preds')
                self._log_result(labels.detach().cpu().numpy(), prefix='val/labels')
            elif self.log_images_type == 'masked':
                self._log_predictions(imgs, labels, preds, prefix='val')

        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels, w = self._unpack(batch)

        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log(f"test/{self.metric_name}", metric, prog_bar=True)
        self.log(f"test/loss", loss)

        if batch_idx == 0:
            if self.log_images_type == 'plain':
                self._log_result(preds.detach().cpu().numpy(), prefix='val/preds')
                self._log_result(labels.detach().cpu().numpy(), prefix='val/labels')
            elif self.log_images_type == 'masked':
                self._log_predictions(imgs, labels, preds)

    def on_epoch_end(self):
        if self.save_checkpoints:
            torch.save(self.state_dict(), os.path.join(self.logger.log_dir, 'checkpoints', 'epoch=%d-step=%d.ckpt' %
                                                       (self.current_epoch, self.global_step)))

    def _log_result(self, preds, prefix='train'):
        # get the tensorboard summary writer
        tensorboard = self.logger.experiment
        tensorboard.add_images(prefix, preds, global_step=self.current_epoch, dataformats='NCHW')

    def _log_predictions(self, x, y, y_pred, prefix='train'):

        # get the tensorboard summary writer
        tensorboard = self.logger.experiment

        # select the center slice if 3D
        if x.ndim == 4:  # 2D data
            x = x[:, x.size(1) // 2, :, :]
            if y is not None:
                y = y[:, 0, :, :]
            y_ = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        else:  # 3D data
            s = x.size(2) // 2
            x = x[:, 0, s, :, :]
            if y is not None:
                y = y[:, 0, s, :, :]
            y_ = np.argmax(y_pred[:, :, s].detach().cpu().numpy(), axis=1)

        # overlay the image with the labels (boundary) and predictions (pixel-wise)
        x_ = np.zeros((x.size(0), x.size(1), x.size(2), 3))
        for b in range(x.size(0)):
            tmp = x[b].cpu().numpy()
            if y is not None:
                tmp = overlay(tmp, y[b].cpu().numpy(), colors=COLORS, boundaries=True)
            x_[b] = overlay(tmp, y_[b], colors=COLORS)

        tensorboard.add_images(prefix + '/prediction', x_, global_step=self.current_epoch, dataformats='NHWC')

    def _unpack(self, batch):
        # TODO: Implement back the weights
        if self.with_labels:
            x, y = batch
            return x, y, None

        return batch, batch, None

    def load_from_pretext(self, pretrained_model, pretext_used_skip=True, drop_parameters=[]):
        # Laod state dict from filesystem
        state_dict = torch.load(pretrained_model)['state_dict']

        # Drop unwanted layers
        for layer in drop_parameters:
            del state_dict[layer]

        if pretext_used_skip:
            # If pretext used skip connection, no fancy operation is needed
            self.load_state_dict(state_dict)
        else:
            # Else, we need to manually load those weights that now doubled in 'in_channel' size
            for i in range(self.decoder.levels):
                # Remove parameters from state dict and load them into the model
                weights = state_dict.pop(f'decoder.features.convblock{i + 1}.conv1.unit.0.weight')
                # print(self.decoder.features)
                with torch.no_grad():
                    self.decoder.features[i * 2 + 1].conv1.unit[0].weight[:, weights.shape[1]:, :, :] = weights

            # Now load the remaining parameters
            self.load_state_dict(state_dict, strict=False)


class UNet2D(UNet):

    def __init__(self, encoder_hparams, decoder_hparams, loss_hparams, optimizer_hparams, metric_hparams,
                 scheduler_hparams, input_shape, with_labels, log_images_type, save_checkpoints, return_features):
        super(UNet2D, self).__init__(loss_hparams, optimizer_hparams, metric_hparams, scheduler_hparams, input_shape,
                                     with_labels, log_images_type, save_checkpoints, return_features)

        # self.save_hyperparameters()
        # contractive path
        self.encoder = UNetEncoder2D(**encoder_hparams)
        # expansive path
        self.decoder = UNetDecoder2D(**decoder_hparams)


class UNetLinearDecoder2D(UNet):
    def __init__(self, encoder_hparams, decoder_hparams, loss_hparams, optimizer_hparams, metric_hparams,
                 scheduler_hparams, input_shape, with_labels, log_images_type, save_checkpoints, return_features):
        super(UNetLinearDecoder2D, self).__init__(loss_hparams, optimizer_hparams, metric_hparams, scheduler_hparams,
                                                  input_shape, with_labels, log_images_type, save_checkpoints,
                                                  return_features)

        # self.save_hyperparameters()
        # contractive path
        self.encoder = UNetEncoder2D(**encoder_hparams)
        # expansive path
        self.decoder = UNetLinearDecoder(**decoder_hparams)


class UNetRotation2D(UNetLinearDecoder2D):
    def __init__(self, encoder_hparams, decoder_hparams, loss_hparams, optimizer_hparams, metric_hparams,
                 scheduler_hparams, input_shape, with_labels, log_images_type, save_checkpoints, return_features):
        super(UNetRotation2D, self).__init__(encoder_hparams, decoder_hparams, loss_hparams, optimizer_hparams,
                                             metric_hparams, scheduler_hparams, input_shape, with_labels,
                                             log_images_type, save_checkpoints, return_features)


class UNetJigsaw2D(UNetLinearDecoder2D):
    def __init__(self, encoder_hparams, decoder_hparams, loss_hparams, optimizer_hparams, metric_hparams,
                 scheduler_hparams, input_shape, with_labels, log_images_type, save_checkpoints, return_features):
        super(UNetJigsaw2D, self).__init__(encoder_hparams, decoder_hparams, loss_hparams, optimizer_hparams,
                                           metric_hparams, scheduler_hparams, input_shape, with_labels, log_images_type,
                                           save_checkpoints, return_features)


class UNet3D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, scheduler=None, step_size=1, gamma=0.1, return_features=False,
                 save_checkpoints=True):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, scheduler=scheduler, step_size=step_size, gamma=gamma,
                         return_features=return_features, save_checkpoints=save_checkpoints)

        # contractive path
        self.encoder = UNetEncoder3D(self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                     norm=self.norm, dropout=self.dropout_enc, activation=self.activation)
        # expansive path
        self.decoder = UNetDecoder3D(self.out_channels, feature_maps=self.feature_maps, levels=self.levels,
                                     skip_connections=self.skip_connections,
                                     residual_connections=self.residual_connections, norm=self.norm,
                                     dropout=self.dropout_dec, activation=self.activation)


class DenseUNet2D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, num_layers=4, k=16, bn_size=2, loss_fn='ce', lr=1e-3, scheduler=None, step_size=1,
                 gamma=0.1, return_features=False, save_checkpoints=True):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, scheduler=scheduler, step_size=step_size, gamma=gamma,
                         return_features=return_features, save_checkpoints=save_checkpoints)

        # parameters
        self.num_layers = int(num_layers)
        self.k = int(k)
        self.bn_size = float(bn_size)

        # contractive path
        self.encoder = DenseUNetEncoder2D(self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                          norm=self.norm, dropout=self.dropout_enc, activation=self.activation,
                                          num_layers=self.num_layers, k=self.k, bn_size=self.bn_size)
        # expansive path
        self.decoder = DenseUNetDecoder2D(self.out_channels, feature_maps=self.feature_maps, levels=self.levels,
                                          skip_connections=self.skip_connections,
                                          residual_connections=self.residual_connections, norm=self.norm,
                                          dropout=self.dropout_dec, activation=self.activation,
                                          num_layers=self.num_layers, k=self.k, bn_size=self.bn_size)


class DenseUNet3D(UNet):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, num_layers=4, k=16, bn_size=2, loss_fn='ce', lr=1e-3, scheduler=None, step_size=1,
                 gamma=0.1, return_features=False, save_checkpoints=True):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, scheduler=scheduler, step_size=step_size, gamma=gamma,
                         return_features=return_features, save_checkpoints=save_checkpoints)

        # parameters
        self.num_layers = int(num_layers)
        self.k = int(k)
        self.bn_size = float(bn_size)

        # contractive path
        self.encoder = DenseUNetEncoder3D(self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                          norm=self.norm, dropout=self.dropout_enc, activation=self.activation,
                                          num_layers=self.num_layers, k=self.k, bn_size=self.bn_size)
        # expansive path
        self.decoder = DenseUNetDecoder3D(self.out_channels, feature_maps=self.feature_maps, levels=self.levels,
                                          skip_connections=self.skip_connections,
                                          residual_connections=self.residual_connections, norm=self.norm,
                                          dropout=self.dropout_dec, activation=self.activation,
                                          num_layers=self.num_layers, k=self.k, bn_size=self.bn_size)
