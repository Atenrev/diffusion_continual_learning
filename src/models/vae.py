################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""

File to place any kind of generative models 
and their respective helper functions.

"""

import torch
import torch.nn as nn
from abc import abstractmethod
from torchvision import transforms
from avalanche.models.utils import MLP, Flatten
from avalanche.models.base_model import BaseModel


class Generator(BaseModel):
    """
    A base abstract class for generators
    """

    @abstractmethod
    def generate(self, batch_size=None, condition=None):
        """
        Lets the generator sample random samples.
        Output is either a single sample or, if provided,
        a batch of samples of size "batch_size"

        :param batch_size: Number of samples to generate
        :param condition: Possible condition for a condotional generator
                          (e.g. a class label)
        """


###########################
# VARIATIONAL AUTOENCODER #
###########################


class VAEMLPEncoder(nn.Module):
    """
    Encoder part of the VAE, computer the latent represenations of the input.

    :param shape: Shape of the input to the network: (channels, height, width)
    :param latent_dim: Dimension of last hidden layer
    """

    def __init__(self, shape, units_dim: tuple = (400, 400), latent_dim: int = 100, use_bn: bool = False):
        super(VAEMLPEncoder, self).__init__()

        flatten_size = torch.Size(shape).numel()
        prev_size = flatten_size
        self.encode = [Flatten(), ]

        for i in range(len(units_dim)):
            self.encode.append(nn.Linear(prev_size, units_dim[i]))
            if use_bn:
                self.encode.append(nn.BatchNorm1d(units_dim[i]))
            self.encode.append(nn.ReLU())
            prev_size = units_dim[i]

        self.encode = nn.Sequential(*self.encode)
        self.z_mean = nn.Linear(prev_size, latent_dim)
        self.z_log_var = nn.Linear(prev_size, latent_dim)

    def forward(self, x, y=None):
        x = self.encode(x)
        if torch.isnan(x).any():
            print("NAN in VAE")
        mean = self.z_mean(x)
        logvar = self.z_log_var(x)
        return x, mean, logvar


class VAEMLPDecoder(nn.Module):
    """
    Decoder part of the VAE. Reverses Encoder.

    :param shape: Shape of output: (channels, height, width).
    :param nhid: Dimension of input.
    """

    def __init__(self, shape, units_dim: tuple = (400, 400), latent_dim: int = 100, use_bn: bool = False):
        super(VAEMLPDecoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        prev_size = latent_dim
        self.shape = shape
        self.decode = []

        for i in range(len(units_dim)):
            self.decode.append(nn.Linear(prev_size, units_dim[i]))
            if use_bn:
                self.decode.append(nn.BatchNorm1d(units_dim[i]))
            self.decode.append(nn.ReLU())
            prev_size = units_dim[i]

        self.decode.append(nn.Linear(prev_size, flattened_size))
        self.decode.append(nn.Sigmoid())
        self.decode = nn.Sequential(*self.decode)
        # self.inv_trans = transforms.Compose(
        #     [transforms.Normalize((0.1307,), (0.3081,))]
        # )

    def forward(self, z, y=None):
        if y is None:
            # return self.inv_trans(self.decode(z).view(-1, *self.shape))
            return self.decode(z).view(-1, *self.shape)
        else:
            # return self.inv_trans(self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape))
            return self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape)


class MlpVAE(Generator, nn.Module):
    """
    Variational autoencoder module:
    fully-connected and suited for any input shape and type.

    The encoder only computes the latent represenations
    and we have then two possible output heads:
    One for the usual output distribution and one for classification.
    The latter is an extension the conventional VAE and incorporates
    a classifier into the network.
    More details can be found in: https://arxiv.org/abs/1809.10635
    """

    def __init__(self, shape, encoder_dims, decoder_dims, latent_dim, n_classes=10, device="cpu"):
        """
        :param shape: Shape of each input sample
        :param nhid: Dimension of latent space of Encoder.
        :param n_classes: Number of classes -
                        defines classification head's dimension
        """
        super(MlpVAE, self).__init__()
        assert latent_dim % 2 == 0, "Latent dimension must be even"

        self.dim = latent_dim
        if device is None:
            device = 'cpu'

        self.device = torch.device(device)
        self.encoder = VAEMLPEncoder(shape, encoder_dims, latent_dim)
        self.decoder = VAEMLPDecoder(shape, decoder_dims, latent_dim)
        self.classification = nn.Linear(encoder_dims[-1], n_classes)

    def get_features(self, x):
        """
        Get features for encoder part given input x
        """
        return self.encoder(x)

    def generate(self, batch_size=None):
        """
        Generate random samples.
        Output is either a single sample if batch_size=None,
        else it is a batch of samples of size "batch_size".
        """
        z = (
            torch.randn((batch_size, self.dim)).to(self.device)
            if batch_size
            else torch.randn((1, self.dim)).to(self.device)
        )

        with torch.no_grad():
            res = self.decoder(z)

        if not batch_size:
            res = res.squeeze(0)
        return res

    def sampling(self, mean, logvar):
        """
        VAE 'reparametrization trick'
        """
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x_o):
        """
        Forward.
        """
        x, mean, logvar = self.encoder(x_o)
        z = self.sampling(mean, logvar)
        x_hat = self.decoder(z)

        if torch.isnan(x_hat).any():
            print("NAN in VAE")

        return x_hat, mean, logvar


BCE_loss = nn.BCELoss(reduction="mean")
MSE_loss = nn.MSELoss(reduction="mean")


def VAE_loss(X, forward_output):
    """
    Loss function of a VAE using mean squared error for reconstruction loss.
    This is the criterion for VAE training loop.

    :param X: Original input batch.
    :param forward_output: Return value of a VAE.forward() call.
                Triplet consisting of (X_hat, mean. logvar), ie.
                (Reconstructed input after subsequent Encoder and Decoder,
                mean of the VAE output distribution,
                logvar of the VAE output distribution)
    """
    from torch.nn import functional as F
    X_hat, mean, logvar = forward_output
    batch_size = X.shape[0]

    if batch_size == 0:
        return torch.tensor(0.0)
    
    # reconstruction_loss = MSE_loss(X_hat, X)
    # reconstruction_loss /= X.shape[1] * X.shape[2] * X.shape[3]
    reconstruction_loss = F.binary_cross_entropy(input=X_hat.view(batch_size, -1), target=X.view(batch_size, -1),
                                            reduction='none')
    reconstruction_loss = torch.mean(reconstruction_loss, dim=1)
    reconstruction_loss = torch.mean(reconstruction_loss)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2, dim=1)
    KL_divergence = torch.mean(KL_divergence)
    KL_divergence /= X.shape[1] * X.shape[2] * X.shape[3]
    return reconstruction_loss + KL_divergence


__all__ = ["MlpVAE"]
