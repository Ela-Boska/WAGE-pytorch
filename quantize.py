import option
import torch
import time
from torch.autograd import Function

LR    = option.lr
bitsW = option.bitsW
bitsA = option.bitsA
bitsG = option.bitsG
bitsE = option.bitsE
bitsR = option.bitsR
beta = option.beta
L2 = option.L2

def S(bits):
    return 2.0 ** (bits - 1)

def Shift(x):
    return 2 ** torch.round(torch.log(torch.tensor(x)) / torch.log(torch.tensor(2.0)))

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

def G_(x,use_bn=False):
    if bitsG > 15:
        return x
    else:
        if use_bn:
            return x  # batch norm parameters, not quantize now

        xmax = torch.max(torch.abs(x))
        x = x / Shift(xmax)

        norm = Q(LR * x, bitsR)

        norm_sign = torch.sign(norm)
        norm_abs = torch.abs(norm)
        norm_int = torch.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = torch.rand(x.shape)
        norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )

        return norm / S(bitsG)

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
        floor = torch.floor(norm)
        fraction = norm-floor
        norm = floor + 0.5 * (torch.sign(fraction - rand_float) + 1)

        return norm / S(bitsG)

def test(n):
    x = torch.rand(n,n)
    t1 = time.time()
    a = G_(x)
    t2 = time.time()
    b = G(x)
    t3 = time.time()
    error = (a-b).abs().sum().item()
    print('G_ takes {}s while G takes {}s, error = {}'.format(t2-t1,t3-t2,error))


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