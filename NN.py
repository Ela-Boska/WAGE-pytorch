import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import layer
from option import use_cuda
import quantize



def relu(x):
    return F.relu(x,True)

class AlanNet(Module):

    def __init__(self,input_size=(3,32,32), num_classes=10):
        super(AlanNet, self).__init__()
        self.seen = 0
        self.features = nn.Sequential(
            layer.conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1, lr=None, activation=relu),      #input
            nn.MaxPool2d(kernel_size=2, stride=2,padding = 0),          #input/2
            layer.conv2d(64, 128, kernel_size=3, stride = 1, padding=1, lr=None, activation=relu),
            nn.MaxPool2d(kernel_size=2, stride=2,padding = 0),          #input/4
            layer.conv2d(128, 256, kernel_size=3, stride = 1, padding=1, lr=None, activation=relu),
            layer.conv2d(256, 256, kernel_size=3, stride = 1, padding=1, lr=None, activation=relu),
            layer.conv2d(256, 256, kernel_size=3, stride = 1, padding=1, lr=None, activation=relu),
            layer.conv2d(256, 512 ,kernel_size=2, stride=2, padding =0, lr=None, activation=None),               #input/8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            layer.linear(512*input_size[1]*input_size[2]//64 , 512, None, relu),
            nn.Dropout(),
            layer.linear(512, 256, None, activation=relu),
            nn.Dropout(),
            layer.linear(256, num_classes, None)
        )
        if use_cuda:
            self.cuda()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


