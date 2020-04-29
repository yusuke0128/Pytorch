import torch
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from torch.autograd import Variable
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image
from model import Net

net = Net()

net.train()
optimizer = optim.Adam()
criterion = nn.MSELoss()


for i in range(100):

    out = net(input)
    loss = c




