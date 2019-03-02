from gqn import representation
import torch
from torch.autograd import Variable

n_channels = 3
v_dim = 7
r_dim = 256
batch = 5
h_dim = 64
pool = False
net = representation.TowerRepresentation(n_channels, v_dim, r_dim, pool)


image = Variable(torch.zeros([5, n_channels, h_dim, h_dim]))
view = Variable(torch.zeros([5, 7]))
y = net.forward(image, view)
print("Image shape:", image.shape)
print("View shape:", view.shape)
print("Output shape:", y.shape)

_, *phi_dims = y.shape
phi = y.view((batch, m, *phi_dims))

# sum over view representations
r = torch.sum(phi, dim=1)

print(y.shape)

