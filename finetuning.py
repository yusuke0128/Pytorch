import torch
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from torch.autograd import Variable
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        net = models.vgg16(pretrained=True)
        self.fc2 = nn.Linear(4096, 8)

    def forward(self, x):
        x = self.net(x)
        x = self.fc2(x)
        return x


