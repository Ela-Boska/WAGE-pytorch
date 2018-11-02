import quantize as q
import torch

EQ = q.EQ.apply
a = torch.tensor([0.125,0.25],requires_grad=True)
b = EQ(a.cuda(), 8)
c = (b*b).sum()
c.backward()