import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.tensor as tensor
import torch.utils.data as utilsdata
from torch.autograd import Variable
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from model import Net
import preprocessing

from finetuning import preprocessing


class Train:
    _device = None
    _optimizer = None
    _criterion = None
    _net = None
    _train_dataset = None
    _val_dataset = None
    _train_loader = None
    _val_loader = None
    _path = None
    _transform = None
    _train_loss = 0
    _train_acc = 0
    _val_loss = 0
    _val_acc = 0
    _epoch = 100

    # d.set_path(_path)

    def set_train_config(self):

        d = preprocessing.Preprocessing()
        d.set_transform()


        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._net = Net().to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(), lr=0.02)
        self._criterion = nn.MSELoss()
        self._path = "/Users/yusuke/dataset/best-artworks-of-all-time/images/images"
        dataset = datasets.ImageFolder(root=self._path, transform=d.get_transform())
        train_id, val_id = train_test_split(list(range(len(dataset))), test_size=0.2)
        self._train_dataset = utilsdata.Subset(dataset, train_id)
        self._val_dataset = utilsdata.Subset(dataset, val_id)
        self._train_loder = utilsdata.DataLoader(dataset=self._train_dataset, batch_size=100, shuffle=True, num_workers=3)
        self._val_loader = utilsdata.DataLoader(dataset=self._val_dataset, batch_size=100, shuffle=True, num_workers=3)

    # def set_configfile(self, arg):

    # def show_config(self):

    def train_start(self):

        self._net.train()

        for i, (inputs, labels) in enumerate(self._train_loder):
            outputs = self.net(inputs)
            loss = self._criterion(outputs, labels)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._train_loss += loss.item()

        self._train_loss = self._train_loss / len(self._train_loder)

        return self._train_loss

    def val_start(self):

        self._net.eval()

        with torch.nn.no_grad:

            for i, (inputs, labels) in enumerate(self._val_loader):

                outputs = self.net(inputs)
                loss = self._criterion(outputs, labels)
                if outputs == labels:
                    self._val_acc += 1

                self._val_loss += loss.item()

            self._val_acc = self._val_acc / len(self._val_loder)
            self._val_loss = self._val_loss / len(self._val_loder)

        return self._val_loss, self._val_acc
