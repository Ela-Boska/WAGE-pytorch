import time
import torch
from torchvision import transforms
import torch.nn.functional as F

debug       =   True

Time        =   time.strftime('%Y-%m-%d', time.localtime())
Notes       =   'temp'

log_file    =   '../WAGE-runs/AlanNet-2888/'
use_cuda    =   True
GPU         =   [0]
batchSize   =   128

train_data  =   ['../cifar10/data_batch_1','../cifar10/data_batch_2','../cifar10/data_batch_3','../cifar10/data_batch_4','../cifar10/data_batch_5']
val_data    =   ['../cifar10/test_batch']

model       =   'NN.AlanNet()'

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
                0:torch.tensor(8).float(),
                200:torch.tensor(1).float(),
                250:torch.tensor(1/8).float()
                }
if use_cuda:
    for key,item in lr_modify.items():
        lr_modify[key] = item.cuda()
max_epoches =   300
L2          =   0

def loss_fc(input,label):
    mask = torch.zeros_like(input)
    mask[torch.arange(len(input)),label] = 1
    temp = F.relu(1-input[mask==1],inplace=True)
    loss = torch.sum(temp**2)
    temp = F.relu(input[mask!=1]+1,inplace=True)
    loss += torch.sum(temp**2)
    return loss


lossFunc    =   loss_fc
optimizer   =   torch.optim.SGD
seed        =   None
W_scale     =   []

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
])