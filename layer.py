import torch
from torch import nn
from torch.nn import Module
import option
import quantize
import torch.nn.functional as F
import pdb

LR    = option.lr
bitsW = option.bitsW
bitsA = option.bitsA
bitsG = option.bitsG
bitsE = option.bitsE
bitsR = option.bitsR
beta = option.beta
L2 = option.L2

sigma_W = 1 / quantize.S(bitsW)
sigma_G = 1 / quantize.S(bitsG)

def clamp_weights(model):
    if type(model) == linear or type(model) == conv2d:
        model.weight.data = model.weight.data.clamp(-1+sigma_G, 1-sigma_G)
        model.bias.data = model.bias.data.clamp(-1+sigma_G, 1-sigma_G)

def change_LR(lr):
    def change(model):
        if type(model) == linear or type(model) == conv2d:
            model.GQ = quantize.GQ(lr).apply
    return change


class conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, activation=None):
        super(conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.L1 = (6/in_channels*stride**2)**0.5
        self.L2 = beta / quantize.S(bitsW)
        self.L = max(self.L1,self.L2)
        self.alpha = max(quantize.Shift(self.L / self.L1), 1)
        self.weight = nn.Parameter( 
            quantize.Q(self.L*(2*torch.rand([out_channels, in_channels, kernel_size, kernel_size])-1),bitsG)
        )
        self.bias = nn.Parameter(
            quantize.Q(self.L*(2*torch.rand([out_channels])-1),bitsG)
        )
        self.GQ = quantize.GQ().apply
        self.EQ = quantize.EQ(in_channels*stride**2).apply
        self.activation = activation

    def forward(self, input):
        weight_tmp = self.GQ(self.weight)
        bias_tmp = self.GQ(self.bias)
        input = F.conv2d(input, weight_tmp, bias_tmp, self.stride, self.padding, self.dilation)
        if self.activation:
            input = self.activation(input)
        input = self.EQ(input)  # constant scaling is included in EQ operation
        return input

class linear(Module):
    def __init__(self, n_in, n_out, activation = None):
        super(linear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.L1 = (6/n_in)**0.5
        self.L2 = beta / quantize.S(bitsW)
        self.L = max(self.L1,self.L2)
        self.alpha = max(quantize.Shift(self.L / self.L1), 1)
        self.weight = nn.Parameter(
            quantize.Q(self.L*(2*torch.rand([n_out, n_in])-1),bitsG)
        )
        self.bias = nn.Parameter(
            quantize.Q(self.L*(2*torch.rand([n_out])-1),bitsG)
        )
        self.GQ = quantize.GQ().apply
        self.EQ = quantize.EQ(n_in).apply
        self.activation = activation
        
    def forward(self, input):
        weight_tmp = self.GQ(self.weight.data)
        bias_tmp = self.GQ(self.bias.data)
        input = F.linear(input, weight_tmp, bias_tmp)
        if self.activation:
            input = self.activation(input)
        input = self.EQ(input)
        return input