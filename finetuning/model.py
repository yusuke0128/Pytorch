import torch
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from PIL import Image


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096, 51)

    def forward(self, x):
        x = F.log_softmax(self.net(x),dim=1)
        return x
