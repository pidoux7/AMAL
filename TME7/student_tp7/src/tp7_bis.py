import logging
logging.basicConfig(level=logging.INFO)
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy

import click
from torch.utils.data import DataLoader,random_split,TensorDataset
from datamaestro import prepare_dataset
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Ratio du jeu de train à utiliser

nom = "tp7_bis.py"


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

#####################################   Load data   #########################################
BATCH_SIZE = 311
TRAIN_RATIO = 0.05
(X, y), (poulet) = mnist.load_data()
length = int(X.shape[0]*TRAIN_RATIO)
X_train = X[:length]
y_train = y[:length] 
X_val = X[length:20000]
y_val = y[length:20000] 
X_test = X[50000:]
y_test = y[50000:]
print(X_train.shape,y_train.shape, X_val.shape,y_val.shape, X_test.shape,y_test.shape)

dim_in = X_train.shape[1]*X_train.shape[2]
dim_out = np.unique(y_train).shape[0]
train_length = int(X_train.shape[0])

ds_train = TensorDataset(torch.tensor(X_train).view(-1,dim_in).float()/255., torch.tensor(y_train).long())
ds_val = TensorDataset(torch.tensor(X_val).view(-1,dim_in).float()/255., torch.tensor(y_val).long())
ds_test = TensorDataset(torch.tensor(X_test).view(-1,dim_in).float()/255., torch.tensor(y_test).long())
train_loader = DataLoader(ds_train,batch_size=BATCH_SIZE)
val_loader = DataLoader(ds_val,batch_size=BATCH_SIZE)
test_loader = DataLoader(ds_test,batch_size=BATCH_SIZE)


#####################################   MLP 3 couches   #########################################
class MLP3(nn.Module):
    def __init__(self,dim_in,l,dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in,l)
        self.linear2 = nn.Linear(l,l)
        self.linear3 = nn.Linear(l,dim_out)
        self.relu = nn.ReLU()
        self.model = nn.Sequential(self.linear1,self.relu,self.linear2,self.relu,self.linear3)

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

class MLP3_dropout(nn.Module):
    def __init__(self,dim_in,l,dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in,l),nn.ReLU(),nn.Dropout(0.1),nn.Linear(l,l),nn.ReLU(),nn.Dropout(0.1),nn.Linear(l,dim_out))

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x
#####################################   paramètres   #########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 1000
dim_latent = 100
lr = 0.01
soft = nn.Softmax(dim=-1)
Loss = nn.CrossEntropyLoss()

#####################################   PCH  #########################################
class State :
    def __init__(self, model, optim) :
        self.model = model
        self.optim = optim
        self.epoch = 0
        #self.iteration = 0

savepath = Path(f"tp7/{nom}.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)
        model = state.model.to(device)
        optimizer = state.optim
        ep = state.epoch

#creer un nouvel état à partir d'un modèle et d'un optimiseur
else:
    model = MLP3(dim_in,dim_latent,dim_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state = State(model, optimizer)
    
#####################################   writer   #########################################

writer = SummaryWriter("tp7/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
accuracy_train = Accuracy(task="multiclass", num_classes=dim_out).to(device)
accuracy_val = Accuracy(task="multiclass", num_classes=dim_out).to(device)
accuracy_test = Accuracy(task="multiclass", num_classes=dim_out).to(device)



#####################################   Train   #########################################
def entropy(output,soft):
    return torch.special.entr(soft(output)).mean(dim=0)

def train_val(train_loader,val_loader, model, optimizer,Loss, epoch, device):

    for epoch in tqdm(range(epochs)):
        ########### TRAIN ############
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = Loss(output,target)
            if epoch % 50 == 0:
                for param in model.parameters():
                    store_grad(param)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:  # Enregistrer toutes les 50 itérations
            writer.add_scalar('Loss train', loss.item(), epoch)
        
            # Enregistrer les poids des couches
            for name, weight in model.named_parameters():
                writer.add_histogram(name, weight, epoch)
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            # Enregistrer l'entropie (ajuster selon votre méthode de calcul de l'entropie)
            entr= entropy(output,soft)
            writer.add_histogram('Entropy train', entr, epoch)

            # Enregistrer l'accuracy
            pred = soft(output).argmax(1).detach().to('cpu')
            target = target.detach().to('cpu')
            accuracy = accuracy_train(pred,target)
            writer.add_scalar('Accuracy train', accuracy.item(), epoch)

        '''
        if epoch % 10 == 0:  # Enregistrer toutes les 50 itérations
            with savepath.open("wb") as fp:
            ep += 1
            torch.save(state, fp)
        '''
        ########### EVAL ############

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = Loss(output,target)

            if epoch % 50 == 0:  # Enregistrer toutes les 50 itérations
                writer.add_scalar('Val Loss', loss.item(), epoch)
                pred = soft(output).argmax(1).detach().to('cpu')
                target = target.detach().to('cpu')
                accuracy = accuracy_val(pred,target)
                writer.add_scalar('Val Accuracy', accuracy.item(), epoch)
                
    writer.close()

def test(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        loss = 0
        accuracy = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += Loss(output,target).item()
            pred = soft(output).argmax(1).detach().to('cpu')
            target = target.detach().to('cpu')
            accuracy += accuracy_test(pred,target)
        loss /= len(test_loader.dataset)
        accuracy /= len(test_loader.dataset)
    return loss, accuracy

def main ():
    
    train_val(train_loader,val_loader, model, optimizer,Loss, epochs, device)
    loss_test, acc_test = test(test_loader, model, device)
    print(f"Loss test : {loss_test}, Accuracy test : {acc_test}")


if __name__ == "__main__":
    main()