import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from torch.utils.data import DataLoader,random_split,TensorDataset

from datamaestro import prepare_dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import mnist
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var
import tensorflow as tf
BATCH_SIZE = 311
train_ratio = 0.8
(X_train, y_train), (poulet) = mnist.load_data()
X_train = X_train[:50000]
y_train = y_train[:50000] 
X_test = X_train[50000:]
y_test = y_train[50000:]
print(X_train.shape,y_train.shape)

dim_in = X_train.shape[1]*X_train.shape[2]
dim_out = np.unique(y_train).shape[0]
train_length = int(X_train.shape[0])


ds_train = TensorDataset(torch.tensor(X_train).view(-1,dim_in).float()/255., torch.tensor(y_train).long())
ds_test = TensorDataset(torch.tensor(X_test).view(-1,dim_in).float()/255., torch.tensor(y_test).long())
train_loader = DataLoader(ds_train,batch_size=BATCH_SIZE)
test_loader = DataLoader(ds_test,batch_size=BATCH_SIZE)


class MLP3(nn.Module):
    def __init__(self,dim_in,l,dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in,l),nn.ReLU(),nn.Linear(l,l),nn.ReLU(),nn.Linear(l,dim_out))

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
epoch = 100
dim_latent = 100
model = MLP3(dim_in,dim_latent,dim_out).to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
soft = nn.Softmax(dim=1)

def train(train_loader, model, optimizer, epoch, device):
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    for e in range(epoch):
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {e} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                loss_train.append(loss.item())
                acc_train.append(((soft(output).argmax(1).detach().to('cpu')==target.detach().to('cpu')).sum()/len(target)))
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(test_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output,target)
                if batch_idx % 10 == 0:
                    print(f'Test Epoch: {e} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                    loss_test.append(loss.item())
                    acc_test.append(((soft(output).argmax(1).detach().to('cpu')==target.detach().to('cpu')).sum()/len(target)))
    return loss_train, loss_test, acc_train, acc_test

loss_train, loss_test, acc_train, acc_test = train(train_loader, model, optimizer, epoch, device)

print(type(loss_train))
plt.plot(loss_train,label="Loss_train")
plt.plot(loss_test,label="Loss_test")
plt.plot(acc_train,label="Acc_train")
plt.plot(acc_test,label="Acc_test")
plt.legend()
plt.show()

class MLP3_dropout(nn.Module):
    def __init__(self,dim_in,l,dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in,l),nn.ReLU(),nn.Dropout(0.1),nn.Linear(l,l),nn.ReLU(),nn.Dropout(0.1),nn.Linear(l,dim_out))

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x