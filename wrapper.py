import torch
from option import use_cuda
from tensorboardX import SummaryWriter
from torch.nn import Module
from layer import clamp_weights

class wraper(Module):

    def __init__(self, model, loss_fc, train_dataset, val_dataset, log_file, batch_size, optimizer, trainning=True):
        super(wraper,self).__init__()
        self.model = model
        self.loss_fc = loss_fc
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_file = log_file
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.trainning = trainning
        self.precision = []
        self.eval_loss = []
        self.seen = 0
        self.epoch = 0
        self.writer = SummaryWriter(log_file)

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
            step = self.seen / len(self.train_dataset)
            self.seen += len(label)
            self.writer.add_scalar('training loss',loss.item(),step)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.apply(clamp_weights)

        else:
            if use_cuda:
                input = input.cuda()
                label = label.cuda()
            output = self.model(input)
            loss = self.loss_fc(output, label)

            