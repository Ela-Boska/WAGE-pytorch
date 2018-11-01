import time
import torch
import NN
from torchvision import transforms


Time        =   time.strftime('%Y-%m-%d', time.localtime())
Notes       =   'temp'

log_file    =   '../WAGE-runs/AlanNet-2888/'
use_cuda    =   True
GPU         =   [0]
batchSize   =   128

train_data  =   ['../cifar10/data_batch_1','../cifar10/data_batch_2','../cifar10/data_batch_3','../cifar10/data_batch_4','../cifar10/data_batch_5']
val_data    =   ['../cifar10/test_batch']

model       =   NN.AlanNet

loadModel   =   None
saveModel   =   '../WAGE-models/AlanNet-2888.pth'


bitsW       =   2  # bit width of we ights
bitsA       =   8  # bit width of activations
bitsG       =   8  # bit width of gradients
bitsE       =   8  # bit width of errors

bitsR       =   16  # bit width of randomizer

beta        =   1.5

lr          =   8
lr_modify   =   {
                0:8,
                200:1,
                250:1/8
                }
max_epoches =   300
L2          =   0

lossFunc    =   torch.nn.MSELoss()
optimizer   =   torch.optim.SGD
seed        =   None
W_scale     =   []

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])