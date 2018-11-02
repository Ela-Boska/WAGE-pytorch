import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image
import random
import matplotlib.pyplot as plt
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class dataset(Dataset):
    def __init__(self,file_list, transform=None, target_transform=None,num_classes=10):
        super(Dataset,self).__init__()
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        data = [unpickle(file) for file in file_list]
        self.names = unpickle('../cifar10/batches.meta')[b'label_names']
        self.label = []
        arrays = []
        for x in data:
            self.label = self.label+x[b'labels']
            arrays = arrays + [x[b'data']]
        data = torch.tensor(np.stack(arrays,0)).view(-1,3,32*32)
        data = data.transpose(-1,-2).view(-1,32,32,3).numpy()
        self.data =[]
        for img in data:
            self.data.append(Image.fromarray(img))
        self.label = torch.LongTensor(self.label)
        

    def __getitem__(self, index):
        assert index< len(self.label),'index out of range!'
        if self.transform:
            data = self.transform(self.data[index])
        else:
            data = self.data[index]
        if self.target_transform:
            label = self.target_transform(self.label[index])
        else:
            label = self.label[index]
        return data,label

    def __len__(self):
        return len(self.label)

    def show_sample(self):
        id =random.randint(0,len(self)-1)
        img = self.data[id]
        label = self.label[id].item()
        label = self.names[label]
        print(label)
        plt.imshow(img)
        plt.show()