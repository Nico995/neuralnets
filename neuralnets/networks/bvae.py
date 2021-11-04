import pytorch_lightning as pl
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from neuralnets.networks.blocks import UNetConvBlock2D, UNetUpSamplingBlock2D
from neuralnets.util.losses import get_loss_function
from neuralnets.util.tools import *


def _reparametrise(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())

    return mu + std * eps


class Encoder(nn.Module):
    """
    2D convolutional encoder

    :param optional input_size: size of the inputs that propagate through the encoder
    :param optional bottleneck_dim: dimensionality of the bottleneck
    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, input_size, bottleneck_dim=2, in_channels=1, feature_maps=64, levels=5, norm='instance',
                 dropout=0.0, activation='relu'):
        super(Encoder, self).__init__()

        self.features = nn.Sequential()
        self.input_size = input_size
        self.bottleneck_dim = bottleneck_dim
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        in_features = in_channels
        for i in range(levels):
            out_features = feature_maps // (2 ** i)

            # convolutional block
            conv_block = UNetConvBlock2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(
            in_features=feature_maps // (2 ** (levels - 1)) * (input_size[0] // 2 ** levels) * (
                    input_size[1] // 2 ** levels), out_features=bottleneck_dim * 2))

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.bottleneck(outputs.view(outputs.size(0), outputs.size(1) * outputs.size(2) * outputs.size(3)))

        return encoder_outputs, outputs


class Decoder(nn.Module):
    """
    2D convolutional decoder

    :param optional input_size: size of the inputs that propagate through the encoder
    :param optional bottleneck_dim: dimensionality of the bottleneck
    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, input_size=512, bottleneck_dim=2, out_channels=2, feature_maps=64, levels=5,
                 norm='instance', dropout=0.0, activation='relu'):
        super(Decoder, self).__init__()

        self.features = nn.Sequential()
        self.input_size = input_size
        self.bottleneck_dim = bottleneck_dim
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        # bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(in_features=bottleneck_dim,
                                                  out_features=feature_maps // (2 ** (levels - 1)) * (
                                                          input_size[0] // 2 ** levels) * (
                                                                       input_size[1] // 2 ** levels)))

        for i in range(levels - 1):
            # upsampling block
            upconv = UNetUpSamplingBlock2D(feature_maps // (2 ** (levels - i - 1)),
                                           feature_maps // (2 ** (levels - i - 2)),
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            conv_block = UNetConvBlock2D(feature_maps // (2 ** (levels - i - 2)), feature_maps // 2 ** (levels - i - 2),
                                         norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # upsampling block
        upconv = UNetUpSamplingBlock2D(feature_maps, feature_maps, deconv=True)
        self.features.add_module('upconv%d' % (levels), upconv)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        fm = self.feature_maps // (2 ** (self.levels - 1))
        inputs = self.bottleneck(inputs).view(inputs.size(0), fm, self.input_size[0] // (2 ** self.levels),
                                              self.input_size[1] // (2 ** self.levels))

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            if i < self.levels - 1:
                outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class BVAE(pl.LightningModule):
    """
    2D beta variational autoencoder (VAE)

    :param optional beta: beta value of the autoencoder (beta=1 results in the classical VAE)
    :param optional input_size: size of the inputs that propagate through the encoder
    :param optional bottleneck_dim: dimensionality of the bottleneck
    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional dropout_enc: encoder dropout factor
    :param optional dropout_dec: decoder dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, beta=0.5, input_size=512, bottleneck_dim=2, in_channels=1, out_channels=1, feature_maps=64,
                 levels=5, norm='instance', activation='relu', dropout_enc=0.0, dropout_dec=0.0, step_size=2, gamma=0.1,
                 lr=0.001):
        super(BVAE, self).__init__()

        self.beta = beta
        self.input_size = input_size
        self.bottleneck_dim = bottleneck_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.encoder_outputs = None
        self.decoder_outputs = None
        self.mu = None
        self.logvar = None
        self.z = None

        self.step_size = step_size
        self.gamma = gamma
        self.lr = lr

        self.loss_rec_fn = get_loss_function('mse')
        self.loss_kl_fn = get_loss_function('kld')

        self.train_batch_id = 0
        self.val_batch_id = 0

        # contractive path
        self.encoder = Encoder(input_size=input_size, bottleneck_dim=bottleneck_dim, in_channels=in_channels,
                               feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout_enc,
                               activation=activation)
        # expansive path
        self.decoder = Decoder(input_size=input_size, bottleneck_dim=bottleneck_dim, out_channels=out_channels,
                               feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout_dec,
                               activation=activation)

    def forward(self, inputs):

        # contractive path
        self.encoder_outputs, bottleneck = self.encoder(inputs)

        self.mu = bottleneck[:, :self.bottleneck_dim]
        self.logvar = bottleneck[:, self.bottleneck_dim:]

        # reparameterization
        self.z = _reparametrise(self.mu, self.logvar)

        # expansive path
        self.decoder_outputs, outputs = self.decoder(self.z, self.encoder_outputs)

        return outputs

    def training_step(self, batch, batch_idx):

        x = batch

        # forward prop
        y_pred = torch.sigmoid(self(x))

        # compute loss
        loss_rec = self.loss_rec_fn(y_pred, x)
        loss_kl = self.loss_kl_fn(self.mu, self.logvar)
        loss = loss_rec + self.beta * loss_kl

        # log images
        if batch_idx == self.train_batch_id:
            self._log_result(y_pred.detach().cpu().numpy() * 255, prefix='train/pred')
            self._log_result(x.detach().cpu().numpy() * 255, prefix='train/truth')

        # compute mse
        self.log('train/mse', loss_rec, prog_bar=True)
        self.log('train/kld', loss_kl, prog_bar=True)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        x = batch

        # forward prop
        y_pred = torch.sigmoid(self(x))

        # compute loss
        loss_rec = self.loss_rec_fn(y_pred, x)
        loss_kl = self.loss_kl_fn(self.mu, self.logvar)
        loss = loss_rec + self.beta * loss_kl

        # log images
        if batch_idx == self.train_batch_id:
            self._log_result(y_pred.detach().cpu().numpy() * 255, prefix='train/pred')
            self._log_result(x.detach().cpu().numpy() * 255, prefix='train/truth')

        # compute mse
        self.log('val/mse', loss_rec, prog_bar=True)
        self.log('val/kld', loss_kl, prog_bar=True)
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer_dict = {"optimizer": optimizer}
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=self.step_size, factor=self.gamma)
        optimizer_dict.update({"lr_scheduler": scheduler, "monitor": 'val/loss'})
        return optimizer_dict

    def _log_result(self, y_pred, prefix='train'):
        # get the tensorboard summary writer
        tensorboard = self.logger.experiment

        tensorboard.add_images(prefix, y_pred, global_step=self.current_epoch, dataformats='NCHW')
