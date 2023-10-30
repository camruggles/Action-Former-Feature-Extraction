import torch

from torch import nn

net = torch.nn.Sequential(
    nn.Linear(1024,1024), nn.ReLU(), 
    nn.Linear(1024,1024), nn.ReLU(),
    nn.Linear(1024,1), nn.ReLU())


x = torch.rand(1024)
x.requires_grad=True


y = net(x)
y2 = 256
loss = y-y2
loss.backward()

print(x.grad)
