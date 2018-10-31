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

sigma_W = 1 / quantize.S(bitsW)

def clamp_weights(model):
    if type(model) == linear or type(model) == conv2d:
        model.weight.data = model.weight.data.clamp(-1+sigma_W, 1-sigma_W)
        model.bias.data = model.bias.data.clamp(-1+sigma_W, 1-sigma_W)


class conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation = 1):
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
            self.L*(2*torch.rand([out_channels, in_channels, kernel_size, kernel_size])-1)
        )
        self.bias = nn.Parameter(
            self.L*(2*torch.rand([out_channels])-1)
        )
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
        super(linear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.L1 = (6/n_in)**0.5
        self.L2 = beta / quantize.S(bitsW)
        self.L = max(self.L1,self.L2)
        self.alpha = max(quantize.Shift(self.L / self.L1), 1)
        self.weight = nn.Parameter( 
            self.L*(2*torch.rand([n_out, n_in])-1)
        )
        self.bias = nn.Parameter(
            self.L*(2*torch.rand([n_out])-1)
        )
        self.GQ = quantize.GQ.apply
        self.EQ = quantize.EQ(n_in).apply
        
    def forward(self, input):
        weight_tmp = self.GQ(self.weight.data)
        bias_tmp = self.GQ(self.bias.data)
        input = F.linear(input, weight_tmp, bias_tmp)
        input /= self.alpha
        input = self.EQ(input)
        return input