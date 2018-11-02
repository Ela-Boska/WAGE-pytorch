import torch
import option
from tensorboardX import SummaryWriter
from torch.nn import Module
from layer import clamp_weights, change_LR
import time
import NN
import quantize
import layer

use_cuda    =   option.use_cuda
lr_modify   =   option.lr_modify
max_epoches =   option.max_epoches
saveModel   =   option.saveModel
log_file    =   option.log_file
batch_size  =   option.batchSize
loss_fc     =   option.lossFunc
optimizer   =   option.optimizer

writer = SummaryWriter(log_file)
class wraper(Module):

    def __init__(self, model, train_dataset, val_dataset, optimizer, trainning=True):
        super(wraper,self).__init__()
        self.model = model
        self.loss_fc = loss_fc
        self.train_dataloader = torch.utils.data.DataLoader(
                                train_dataset,
                                shuffle =True, batch_size=option.batchSize)
        self.val_dataloader = torch.utils.data.DataLoader(
                                val_dataset,
                                shuffle =True, batch_size=option.batchSize)
        self.data_size = len(train_dataset)
        self.saveModel = saveModel
        self.log_file = log_file
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters(),1)
        self.trainning = trainning
        self.precision = []
        self.eval_loss = []
        self.seen = 0
        self.epoch = 0
        self.loss_total = 0
        self.total = 0
        self.correct = 0
        self.lr = None
        self.best_precision = 0

    def train(self,t=True):
        super(wraper, self).train(t)
        self.trainning = t

    def forward(self,input,label):
        if self.trainning:
            if use_cuda:
                input = input.cuda()
                label = label.cuda()
            output = self.model(input)
            loss = self.loss_fc(output, label)
            step = self.seen // self.batch_size
            self.seen += len(label)
            writer.add_scalar('training loss',loss.item(),step)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.apply(clamp_weights)

        else:
            if use_cuda:
                input = input.cuda()
                label = label.cuda()
            output = self.model(input)
            loss = self.loss_fc(output, label).item()
            length = len(label)
            self.loss_total += loss*length
            self.total += length

            predict = output.argmax(-1)
            self.correct += sum( predict==label ).item()

    def Train(self):
        self.train()
        if self.epoch in lr_modify:
            self.lr = lr_modify[self.epoch]
            self.apply(change_LR(self.lr))
        now = time.time()
        now_date = time.asctime( time.localtime(now) )
        print(now_date,'lr =',self.lr.item())
        for input,label in self.train_dataloader:
            self.forward(input,label)
        self.epoch += 1
        


    def Val(self):
        self.eval()
        with torch.no_grad():
            for input,label in self.val_dataloader:
                self.forward(input,label)
        loss = self.loss_total / self.total
        precision = self.correct / self.total
        print('epoch {0:>3}, precision = {1:>5.4}%, loss = {1:>5.4}'.format(self.epoch, 100*precision, loss))
        writer.add_scalar('precision',precision,self.epoch)
        writer.add_scalar('val_loss',loss,self.epoch)
        self.precision.append(precision)
        self.eval_loss.append(loss)
        if precision>self.best_precision:
            self.best_precision = precision
            torch.save(self,self.saveModel)
        self.loss_total = self.total = self.correct = 0