import option
import torch
import time
from torch.autograd import Function
import pdb

LR    = option.lr
bitsW = option.bitsW
bitsA = option.bitsA
bitsG = option.bitsG
bitsE = option.bitsE
bitsR = option.bitsR
beta = option.beta
L2 = option.L2
use_cuda = option.use_cuda

def S(bits):
    return 2.0 ** (bits - 1)

def Shift(x):
    if use_cuda:
        ans =  2 ** torch.round(torch.log(torch.tensor(x)).cuda() / torch.log(torch.tensor(2.0).cuda()))
    else:
        ans =  2 ** torch.round(torch.log(torch.tensor(x)) / torch.log(torch.tensor(2.0)))
    return ans
    

def C(x, bits=32):
    if bits > 15 or bits == 1:
        delta = 0.
    else:
        delta = 1 / S(bits)
    MAX = +1 - delta
    MIN = -1 + delta
    x = torch.clamp(x, MIN, MAX)
    return x

def Q(x, bits):
    if bits > 15:
        return x
    elif bits == 1:  # BNN
        return torch.sign(x)
    else:
        SCALE = S(bits)
    return C(torch.round(x * SCALE) / SCALE, bits)

def G(x,use_bn=False):
    if bitsG > 15:
        return x
    else:
        if use_bn:
            return x  # batch norm parameters, not quantize now

        xmax = torch.max(torch.abs(x))
        x = x / Shift(xmax)

        norm = Q(LR * x, bitsR)

        rand_float = torch.rand(x.shape)
        if use_cuda:
            rand_float = rand_float.cuda()
        floor = torch.floor(norm)
        fraction = norm-floor
        norm = floor + 0.5 * (torch.sign(fraction - rand_float) + 1)

        return norm / S(bitsG)


def error(op, x):
    if bitsE > 15:
        return x
    else:
        xmax = torch.max(torch.abs(x))
        xmax_shift = Shift(xmax)
        return Q(C( x /xmax_shift, bitsE), bitsE)

class GQ(Function):
    

    @staticmethod
    def forward(ctx , i):
        result = Q(i,bitsW)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        gs = LR * grad_output / Shift(torch.max(grad_output))
        return G(gs)

def EQ(n_in):

    class E_Q(Function):
    
        L1 = (6/n_in)**0.5
        L2 = beta / S(bitsW)
        L = max(L1,L2)
        alpha = Shift(L/L1)

        @staticmethod
        def forward(ctx , i):
            result = Q(i/E_Q.alpha, bitsA)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            return Q(grad_output / Shift(torch.max(grad_output.abs())), bitsE)
        
    return E_Q