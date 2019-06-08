"""
Creates predictions from specified number of context images
Context, Prediction and Ground truth images are saved in data/tests
"""
from gqn import GenerativeQueryNetwork, partition
from dataset import GQN_Dataset
import shutil
import os
import numpy as np
import cv2
import torch
import glob

shutil.rmtree("data/tests", ignore_errors=True)
os.makedirs("data/tests")
valid_dataset = GQN_Dataset(root_dir="data/rooms_ring_camera/", train=False)
x, v = valid_dataset[1]
x = x.view((1, *x.shape))
v = v.view((1, *v.shape))

max_m=5
x, v, x_q, v_q = partition(x, v, max_m, 4)
batch, *_ = x.shape
device = torch.device("cpu")
model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=10).to(device)
models = glob.glob("checkpoints/checkpoint_model_*.pth")
models.sort(key=lambda x: os.path.getmtime(x))
last_checkpoint = models[-1]
checkpoint = torch.load(last_checkpoint, map_location="cpu")
model.load_state_dict(checkpoint)

torch.manual_seed(7)
x_mu = model.sample(x, v, v_q)
x_mu = x_mu.detach().cpu().numpy()*255

x = x.numpy()
x*=255
x_q = x_q.numpy()*255
x = np.moveaxis(x, [2, 3, 4], [-1, 2, 3])
x_q = np.moveaxis(x_q, [1, 2, 3], [-1, -3, -2])
x_mu = np.moveaxis(x_mu, [1, 2, 3], [-1, -3, -2])

n = 1

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
