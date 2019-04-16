import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .representation import TowerRepresentation
from .generator import GeneratorNetwork

class GenerativeQueryNetwork(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, dgf_dim, L=12, pool=False):
        super(GenerativeQueryNetwork, self).__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=pool)

        self.bn = nn.BatchNorm1d(r_dim)
        self.input_layer = nn.Linear(r_dim, dgf_dim)
        self.output_layer = nn.Linear(dgf_dim, r_dim)
        for param in self.input_layer.parameters():
            param.requires_grad = False
        for param in self.output_layer.parameters():
            param.requires_grad = False



    def forward(self, context_x, context_v, query_x, query_v):
        """
        Forward through the GQN.

        :param x: batch of context images [b, m, c, h, w]
        :param v: batch of context viewpoints for image [b, m, k]
        :param x_q: batch of query images [b, c, h, w]
        :param v_q: batch of query viewpoints [b, k]
        """
        # Merge batch and view dimensions.
        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.shape
        phi = phi.view((b, m, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        r_dgf = r.view((b, -1))
        dgf = F.relu(self.input_layer(self.bn(r_dgf)))

        r = self.bn(self.output_layer(dgf))
        r = r.view((b, *phi_dims))

        # Use random (image, viewpoint) pair in batch as query
        x_mu, kl = self.generator(query_x, query_v, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return (x_mu, r, kl)

    def sample(self, context_x, context_v, query_v):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        batch_size, n_views, _, h, w = context_x.shape
        
        _, _, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)
        
        r_dgf = r.view((batch_size, -1))
        mu1 = r_dgf.transpose(0,1).contiguous().mean(1)
        var1 = r_dgf.transpose(0,1).contiguous().var(1)
        dgf = F.relu(self.input_layer(self.bn(r_dgf)))

        r = self.output_layer(dgf)
        mu2 = r.transpose(0,1).contiguous().mean(1)
        var2 = r.transpose(0,1).contiguous().var(1)
        r = self.bn(r)
        r = r.view((batch_size, *phi_dims))

        x_mu = self.generator.sample((h, w), query_v, r)
        return x_mu, (mu1, var1), (mu2, var2)

    def get_dgf(self, context_x, context_v, bn_stats):
        """
        Get DGF from context and context_v

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        """
        batch_size, n_views, _, h, w = context_x.shape
        
        _, _, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        self.phi_dims = phi_dims
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)
        
        r_dgf = r.view((batch_size, -1))
        
        mu, var = bn_stats
        r_dgf = (r_dgf - mu) / torch.sqrt(var+0.00001)
        r_dgf = self.bn.weight*r_dgf + self.bn.bias
        
        dgf = self.input_layer(r_dgf)
        return dgf
    
    def sample_from_dgf(self, x_shape, r_dgf, query_v, bn_stats):
        """
        Get DGF from context and context_v

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        """
        batch_size, h, w = x_shape
        dgf = F.relu(r_dgf)
        
        r = self.output_layer(dgf)
        mu, var = bn_stats
        r = (r - mu) / torch.sqrt(var+0.00001)
        r = self.bn.weight*r + self.bn.bias

        r = r.view((batch_size, *self.phi_dims))


        x_mu = self.generator.sample((h, w), query_v, r)
        return x_mu
