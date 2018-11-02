import wrapper
import preprocess
import option
import torch
from torchvision import transforms
import os
import NN

if not option.loadModel:
    model   =   eval(option.model)
else:
    model   =   wrapper.load(option.loadModel)

train_dataset   =   preprocess.dataset(option.train_data, transform=option.transform_train)

val_dataset     =   preprocess.dataset(option.val_data, transform=option.transform_test)

ShengZhiyao =   wrapper.wraper(model, train_dataset, val_dataset, option.optimizer)

max_epoches = option.max_epoches
while ShengZhiyao.epoch < max_epoches:
    ShengZhiyao.Train()
    ShengZhiyao.Val()