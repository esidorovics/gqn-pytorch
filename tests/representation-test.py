from gqn import representation
import torch
from torch.autograd import Variable

n_channels = 3
v_dim = 7
r_dim = 256
m = 5
h_dim = 64
b = 36
pool = True
net = representation.TowerRepresentation(n_channels, v_dim, r_dim, pool)


image = Variable(torch.zeros([m*b, n_channels, h_dim, h_dim]))
view = Variable(torch.zeros([m*b, v_dim]))
y = net.forward(image, view)
print("Image shape:", image.shape)
print("View shape:", view.shape)
print("Output shape:", y.shape)

_, *phi_dims = y.shape
phi = y.view((b, m, *phi_dims))

# sum over view representations
r = torch.sum(phi, dim=1)

print(r.shape)

