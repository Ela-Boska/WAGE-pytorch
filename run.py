import wrapper
import preprocess
import option
import torch
from torchvision import transforms
import os

if not option.loadModel:
    model   =   option.model()
else:
    model   =   torch.load(option.loadModel)

train_dataloader    =   torch.utils.data.DataLoader(
                    preprocess.dataset(option.train_data, transform=option.transform_train),
                    shuffle =True, batch_size=option.batchSize)

val_dataloader      =   torch.utils.data.DataLoader(
                    preprocess.dataset(option.val_data, transform=option.transform_test),
                    shuffle =True, batch_size=option.batchSize)

ShengZhiyao =   wrapper.wraper(model, train_dataloader, val_dataloader, option.optimizer)

max_epoches = option.max_epoches
while ShengZhiyao.epoch < max_epoches:
    ShengZhiyao.Train()
    ShengZhiyao.Val()