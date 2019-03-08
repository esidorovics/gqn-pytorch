import numpy as np
import torch
from torch import nn
SCALE=4
x = torch.zeros([36, 3, 64, 64])
downsample = nn.Conv2d(3, 3, kernel_size=SCALE, stride=SCALE, padding=0, bias=False)
print(x.shape)
x = downsample(x)
print(x.shape)