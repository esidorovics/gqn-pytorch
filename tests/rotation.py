from gqn import GenerativeQueryNetwork, partition, Annealer
from dataset import GQN_Dataset
import glob
import os
import numpy as np
import cv2
import torch

valid_dataset = GQN_Dataset(root_dir="../data/rooms_ring_camera/", train=False)
x, v = valid_dataset[4]
x = x.view((1, *x.shape))
v = v.view((1, *v.shape))

max_m=5
x, v, x_q, v_q = partition(x, v, max_m, 4)
batch, *_ = x.shape
device = torch.device("cpu")
model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=10).to(device)
models = glob.glob("../checkpoints/checkpoint_model_*.pth")
models.sort(key=lambda x: os.path.getmtime(x))
last_checkpoint = models[-1]

checkpoint = torch.load(last_checkpoint, map_location="cpu")
model.load_state_dict(checkpoint)

scenes = [1, 2]
for scene in scenes:
    x_sc = x[scene]
    x_sc = x_sc.repeat(36, 1, 1, 1, 1)
    v_sc = v[scene]
    coord1, coord2, coord3, _, _, p1, p2 = v_q[scene]
    v_sc = v_sc.repeat(36, 1, 1)
    for j in range(36):
        v_q[j] = torch.FloatTensor([coord1, coord2, coord3, np.cos(0.2 * j), np.sin(0.2 * j), p1, p2])

    for i in range(1):
        x_mu = model.sample(x_sc, v_sc, v_q)
        x_mu = x_mu.detach().cpu().numpy() * 255
        x_mu = np.moveaxis(x_mu, [1, 2, 3], [-1, -3, -2])
        x_mu = x_mu.astype(np.uint8)

        for j in range(batch):
            savepath = "../data/rotation/{}".format(scene)
            img = cv2.cvtColor(x_mu[j], cv2.COLOR_BGR2RGB)

            if not os.path.exists(savepath):
                os.makedirs(savepath)
            cv2.imwrite(savepath+"/test_{}.png".format(j), img)
