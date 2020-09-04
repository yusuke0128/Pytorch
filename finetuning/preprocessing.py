import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image


class Preprocessing:

    _path=""
    _transform=""
    _dataset=""

    def set_path(self,arg):
        self._path = arg

    def get_path(self):
        return self._path

    def set_transform(self):
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(244, 244),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def get_transform(self):
        return self._transform

    def set_dataset(self):
        self._dataset = datasets.ImageFolder(root=self._path, transform=self._transform)

    #def get_dataset(self):
    #    return self._dataset

    #def set_dataloder(self):
    #    train_loader = data.DataLoader(dataset=self._dataset, batch_size=100, shuffle=True,num_workers=3)

