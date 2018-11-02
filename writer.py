import itertools
import torch
from torchvision import models


def log_weights(writer,model,step):
    for name, param in model.named_parameters():
        writer.add_histogram(name,param,step)

def log_grads(writer,model,step):
    for name, param in model.named_parameters():
        if type(param.grad) != type(None):
            writer.add_histogram(name,param.grad,step)

def log_graph(writer,model):
    inputs = torch.randn(1,3,32,32)
    writer.add_graph(model,inputs)