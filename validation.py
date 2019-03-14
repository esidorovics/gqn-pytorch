from gqn import GenerativeQueryNetwork, partition, Annealer
from dataset import GQN_Dataset
from torch.distributions import Normal
import shutil
import os
import numpy as np
import cv2
import torch

shutil.rmtree("data/tests")
os.makedirs("data/tests")
valid_dataset = GQN_Dataset(root_dir="/media/data/gqn-dataset/rooms_ring_camera/", train=False)
x, v = valid_dataset[1]
x = x.view((1, *x.shape))
v = v.view((1, *v.shape))

max_m=5
x, v, x_q, v_q = partition(x, v, max_m)
batch, *_ = x.shape
print(x.shape)
print(v.shape)
print(x_q.shape)
device = torch.device("cpu")
model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=12).to(device)
# checkpoint = torch.load("./checkpoints/checkpoint_model_28000.pth")
checkpoint = torch.load("./chkpnts/checkpoint_model_3000.pth")
model.load_state_dict(checkpoint)

x_mu, r, _ = model(x, v, x_q, v_q)
ll = Normal(x_mu, 1).log_prob(x_q)
ll3 = Normal(x_mu, 1).log_prob(x_q)
ll2 = Normal(x_q , 1).log_prob(x_q)
ll = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
ll2 = torch.mean(torch.sum(ll2, dim=[1, 2, 3]))
ll3 = torch.mean(torch.sum(ll3, dim=[1, 2, 3]))
print(ll)
print(ll2)
print(ll3)
x_mu = x_mu.detach().cpu().numpy()
print(x_mu)


x = x.numpy()
x_q = x_q.numpy()
x = np.moveaxis(x, [2, 3, 4], [-1, 2, 3])
x_q = np.moveaxis(x_q, [1, 2, 3], [-1, -3, -2])
x_mu = np.moveaxis(x_mu, [1, 2, 3], [-1, -3, -2])

n = 5
for i in range(batch//n):
    start = i*n
    end = start+n
    x_qs = x_q[start:end]
    xs = x[start:end]
    x_mus = x_mu[start:end]

    full = np.concatenate(xs, axis=1)
    full = np.concatenate(full, axis=1)
    x_qs = np.concatenate(x_qs, axis=0)
    x_mus = np.concatenate(x_mus, axis=0)
    space = np.ones(shape=(full.shape[0], 10, 3))*255
    full = np.concatenate([full, space, x_mus, space, x_qs], axis=1)
    cv2.imwrite("data/tests/test_{}.png".format(i), full)