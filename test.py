from AlanNet import *
from itertools import chain

input = torch.randn(128,3,32,32)
x = input.clone()
model = AlanNet()
models1 = list(model.features.modules())[1:]
ouputs = [input]
for i in range(0,len(models1)):
    x = models1[i](x.clone())
    ouputs.append(x)
models2 = list(model.classifier.modules())[1:]
x = x.view(128,-1)
for i in range(0,len(models2)):
    x = models2[i](x.clone())
    ouputs.append(x)
for i in ouputs:
    print(i.std(), i.mean())
print()
for i in chain(models1,models2):
    if type(i) == layer.linear or type(i) == layer.conv2d:
        print(i.GQ(i.weight.data).std())
