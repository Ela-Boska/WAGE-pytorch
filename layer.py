import torch
from torch import nn
from torch.nn import Module
import option
import quantize
import torch.nn.functional as F

LR    = option.lr
bitsW = option.bitsW
bitsA = option.bitsA
bitsG = option.bitsG
bitsE = option.bitsE
bitsR = option.bitsR
beta = option.beta
L2 = option.L2


# the initialization still needs modification
class conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation = 1):
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
        self.weight = nn.Parameter(torch.randint(1-2**(bitsW-1),2**(bitsW-1),self.conv.weight.data.shape) / quantize.S(bitsW), True)
        self.bias = nn.Parameter(torch.randint(1-2**(bitsW-1),2**(bitsW-1),self.conv.bias.data.shape) / quantize.S(bitsW), True)
        self.GQ = quantize.GQ.apply
        self.EQ = quantize.EQ(in_channels*stride**2).apply

    def forward(self, input):
        weight_tmp = self.GQ(self.weight.data)
        bias_tmp = self.GQ(self.bias.data)
        input = F.conv2d(input, weight_tmp, bias_tmp, self.stride, self.padding, self.dilation)
        input = input / self.alpha
        input = self.EQ(input)
        return input

class linear(Module):
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.L1 = (6/n_in)**0.5
        self.L2 = beta / quantize.S(bitsW)
        self.L = max(self.L1,self.L2)
        self.alpha = max(quantize.Shift(self.L / self.L1), 1)
        self.weight = nn.Parameter(torch.randint(1-2**(bitsW-1),2**(bitsW-1),[n_in, n_out]) / quantize.S(bitsW), True)
        self.bias = nn.Parameter(torch.randint(1-2**(bitsW-1),2**(bitsW-1), [n_out]) / quantize.S(bitsW), True)
        self.GQ = quantize.GQ.apply
        self.EQ = quantize.EQ(in_channels*stride**2).apply
        
    def forward(self, input):
        weight_tmp = self.GQ(self.weight.data)
        bias_tmp = self.GQ(self.bias.data)
        input = F.linear(input, weight_tmp, bias_tmp)
        return input