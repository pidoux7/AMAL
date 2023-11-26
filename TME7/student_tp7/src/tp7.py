import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader,TensorDataset, Dataset, random_split
from datamaestro import prepare_dataset
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, GaussianBlur


#####################################   utils   #########################################
logging.basicConfig(level=logging.INFO)

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

class State :
    def __init__(self, model, optim) :
        self.model = model
        self.optim = optim
        self.epoch = 0
        #self.iteration = 0

def entropy(output,soft):
    return torch.special.entr(soft(output)).mean(dim=0)

def l1_regularization(model, lambda_l1):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.abs(param).sum()
    return lambda_l1 * l1_loss


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
        self.linear1 = nn.Linear(dim_in,l)
        self.linear2 = nn.Linear(l,l)
        self.linear3 = nn.Linear(l,dim_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.model = nn.Sequential(self.linear1,self.relu,self.dropout1,self.linear2,self.relu,self.dropout2,self.linear3)

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

class MLP3_batch_norm(nn.Module):
    def __init__(self,dim_in,l,dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in,l)
        self.linear2 = nn.Linear(l,l)
        self.linear3 = nn.Linear(l,dim_out)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(l)
        self.bn2 = nn.BatchNorm1d(l)
        self.model = nn.Sequential(self.linear1,self.bn1,self.relu,self.linear2,self.bn2,self.relu,self.linear3)

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

class MLP3_dropout_batch_norm(nn.Module):
    def __init__(self, dim_in, l, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, l)
        self.bn1 = nn.BatchNorm1d(l)  # BatchNorm après la première couche linéaire
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.linear2 = nn.Linear(l, l)
        self.bn2 = nn.BatchNorm1d(l)  # BatchNorm après la deuxième couche linéaire
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.linear3 = nn.Linear(l, dim_out)
        # Pas de BatchNorm après la dernière couche linéaire

        # Construction du modèle séquentiel
        self.model = nn.Sequential(
            self.linear1, self.bn1, self.relu1, self.dropout1,
            self.linear2, self.bn2, self.relu2, self.dropout2,
            self.linear3
        )

    def forward(self, x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x
    
class MLP3_dropout_layer_norm(nn.Module):
    def __init__(self, dim_in, l, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, l)
        self.ln1 = nn.LayerNorm(l)  # BatchNorm après la première couche linéaire
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.linear2 = nn.Linear(l, l)
        self.ln2 = nn.LayerNorm(l)  # BatchNorm après la deuxième couche linéaire
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.linear3 = nn.Linear(l, dim_out)
        # Pas de BatchNorm après la dernière couche linéaire

        # Construction du modèle séquentiel
        self.model = nn.Sequential(
            self.linear1, self.ln1, self.relu1, self.dropout1,
            self.linear2, self.ln2, self.relu2, self.dropout2,
            self.linear3
        )

    def forward(self, x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x
    
#####################################   data augmentation   #########################################
def apply_transform(data, transform):
    transformed_data = []
    for x in data:
        # Convertir en PIL Image si nécessaire, puis appliquer la transformation
        x_transformed = transform(x)
        transformed_data.append(x_transformed)
    return torch.stack(transformed_data)

#####################################   Train   #########################################


def train_val(train_loader,val_loader, model, optimizer,Loss, epochs, device, writer, soft, accuracy_train, accuracy_val, reg=None):
    lambda_l1 = 0.0001
    for epoch in tqdm(range(epochs)):
        ########### TRAIN ############
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = Loss(output,target)
            if reg == 'L1' or reg == 'L1L2' :
                loss += l1_regularization(model, lambda_l1)

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
        if epoch % 10 == 0:  # Enregistrer toutes les 10 itérations
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

#####################################   Test   #########################################

def test(test_loader, model, device, Loss, soft, accuracy_test):
    model.eval()
    with torch.no_grad():
        loss = 0
        accuracy = 0
        length = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += Loss(output,target).item()
            pred = soft(output).argmax(1).detach().to('cpu')
            target = target.detach().to('cpu')
            accuracy += accuracy_test(pred,target)
            length += 1
        loss /= length
        accuracy /= length
    return loss, accuracy
    



def main ():
    #####################################   Load data   #########################################
    parser = argparse.ArgumentParser(description="Entraînement de modèle MLP pour MNIST")
    parser.add_argument("--model", choices=['MLP3', 'MLP3_dropout','MLP3_batch_norm','MLP3_dropout_batch_norm', 'MLP3_dropout_layer_norm'], required=True, help="Choisir le modèle à utiliser: 'MLP3' ou 'MLP3_dropout'")
    parser.add_argument("--reg", choices=['L1', 'L2','L1L2'], required=False, help="Choisir la regularisation à utiliser: 'L1' ou 'L2'")
    parser.add_argument("--data_aug", choices=["blur"], required = False, help="Utiliser la data augmentation")
    # Analyse des arguments
    args = parser.parse_args()
    nom = "tp7_ter"
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
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_val shape:', X_val.shape)
    print('y_val shape:', y_val.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    dim_in = X_train.shape[1]*X_train.shape[2]
    dim_out = np.unique(y_train).shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    epochs = 1000
    dim_latent = 100
    lr = 0.01
    soft = nn.Softmax(dim=-1)
    Loss = nn.CrossEntropyLoss()

    if args.data_aug == "blur":
        print("blur")
        transform = transforms.Compose([
            ToTensor(),
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))])  
        X_train = apply_transform(X_train, transform)

    ds_train = TensorDataset(torch.tensor(X_train).view(-1,dim_in).float()/255., torch.tensor(y_train).long())
    ds_val = TensorDataset(torch.tensor(X_val).view(-1,dim_in).float()/255., torch.tensor(y_val).long())
    ds_test = TensorDataset(torch.tensor(X_test).view(-1,dim_in).float()/255., torch.tensor(y_test).long())
    train_loader = DataLoader(ds_train,batch_size=BATCH_SIZE)
    val_loader = DataLoader(ds_val,batch_size=BATCH_SIZE)
    test_loader = DataLoader(ds_test,batch_size=BATCH_SIZE)


    #####################################   PCH  #########################################

    
    savepath = Path(f"tp7/{nom}.pch")
    if savepath.is_file():
        with savepath.open("rb") as fp:
            state = torch.load(fp)
            model = state.model.to(device)
            optimizer = state.optim
            ep = state.epoch
    #creer un nouvel état à partir d'un modèle et d'un optimiseur
    else:
        # Configuration des models
        if args.model == 'MLP3':
            print("MLP3")
            model = MLP3(dim_in, dim_latent, dim_out).to(device)
        elif args.model == 'MLP3_dropout':
            print("MLP3_dropout")
            model = MLP3_dropout(dim_in, dim_latent, dim_out).to(device)
        elif args.model == 'MLP3_batch_norm':
            print("MLP3_batch_norm")
            model = MLP3_batch_norm(dim_in, dim_latent, dim_out).to(device)
        elif args.model == 'MLP3_dropout_batch_norm':
            print("MLP3_dropout_batch_norm")
            model = MLP3_dropout_batch_norm(dim_in, dim_latent, dim_out).to(device)
        elif args.model == 'MLP3_dropout_layer_norm':
            print("MLP3_dropout_layer_norm")
            model = MLP3_dropout_layer_norm(dim_in, dim_latent, dim_out).to(device)
        # Configuration de l'optimiseur
        if args.reg == 'L1':
            print("L1")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            reg = 'L1'
        elif args.reg == 'L2':
            print("L2")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
            reg = 'L2'
        elif args.reg == "L1L2":
            print("L1L2")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
            reg = 'L1L2'
        elif args.reg == None:
            print("Pas de regularisation")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            reg = None
        state = State(model, optimizer)
        
    #####################################   writer   #########################################

    writer = SummaryWriter("tp7/"+f'{args.model}_{args.reg}_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    accuracy_train = Accuracy(task="multiclass", num_classes=dim_out).to(device)
    accuracy_val = Accuracy(task="multiclass", num_classes=dim_out).to(device)
    accuracy_test = Accuracy(task="multiclass", num_classes=dim_out).to(device)


    #####################################   Train_test   #########################################
    train_val(train_loader,val_loader, model, optimizer,Loss, epochs, device,  writer, soft, accuracy_train, accuracy_val,reg=reg)
    loss_test, acc_test = test(test_loader, model, device,Loss, soft, accuracy_test)
    print(f"Loss test : {loss_test}, Accuracy test : {acc_test}")


if __name__ == "__main__":
    main()