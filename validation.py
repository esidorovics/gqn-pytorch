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
x, v, x_q, v_q = partition(x, v, max_m, 5)
batch, *_ = x.shape
print(x.shape)
print(v.shape)
print(x_q.shape)
device = torch.device("cpu")
model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=6).to(device)
checkpoint = torch.load("./checkpoints/checkpoint_model_50000.pth") #
# checkpoint = torch.load("./chkpnts/current/checkpoint_model_40000.pth") #520
# checkpoint = torch.load("./chkpnts/current/checkpoint_model_100000.pth") #485
model.load_state_dict(checkpoint)

# x_mu = model.sample(x, v, v_q)*255
x_mu, r, kl = model(x, v, x_q, v_q)
# print(kl.shape)
ll = Normal(x_mu, 0.1).log_prob(x_q*255)
l = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
print(l.detach().cpu().numpy()/1000)
x_mu = x_mu.detach().cpu().numpy()
print(np.std(x_mu))


x = x.numpy()
x*=255
x_q = x_q.numpy()*255
print(np.std(x_q))
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
