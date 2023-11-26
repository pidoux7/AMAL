#####################################################################################################
from utils2 import RNN, ForecastMetroDataset, State
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


#####################################################################################################
# Nombre de stations utilisé
CLASSES = 10
#vLongueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 32
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Chargement des données
PATH = "/home/pidoux/master/deepdac/AMAL/TME4/data/"

#####################################################################################################
'''CHARGEMENT DES DONNEES'''
matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

writer = SummaryWriter("runs/runs1"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#####################################################################################################
'''INITIALISATION DU RNN'''
DIM_LATENT = 5

nb_epochs = 100
lr = 0.01
f_cout = nn.MSELoss()
pas_temps = 1
model = RNN(input_dim=DIM_INPUT, latent_dim=DIM_LATENT, 
            output_dim=DIM_INPUT, activation = nn.Tanh(), 
            decode_activation = nn.Sigmoid())
model = model.to(device)
optim = torch.optim.SGD(model.parameters(), lr=lr)
state = State(model, optim)
#####################################################################################################
# APPRENTISSAGE
for epoch in tqdm(range(nb_epochs)):
    for X, y in data_train:
        batch_len,seq_len,classes,dim = X.size()
        X = torch.transpose(X, 0, 1).reshape(seq_len, -1, dim)
        y = torch.transpose(y, 0, 1).reshape(seq_len, -1, dim)
        X = X.to(device)
        y = y.to(device)
        #print('X',X.size())
        #print('y',y.size())
        state.optim.zero_grad()
        h = torch.zeros((X.size(1), DIM_LATENT)).to(device)
        #print('h',h.size())
        loss = 0
        for t in range(X.size(0) - pas_temps):
            ht = h
            h = state.model.one_step(X[t], h)
            for p in range(0,pas_temps+1):
                ht = state.model.one_step(X[t+p], ht)
            y_pred = state.model.decode(ht)
            y_true = y[t+pas_temps]
            loss += f_cout(y_pred, y_true)
        loss.backward()
        state.optim.step()
        writer.add_scalar("Loss/train", loss, epoch)
    print("epoch : ", epoch, "loss_train: ", loss)

    with torch.no_grad():
        for X, y in data_train:
            batch_len,seq_len,classes,dim = X.size()
            X = torch.transpose(X, 0, 1).reshape(seq_len, -1, dim)
            y = torch.transpose(y, 0, 1).reshape(seq_len, -1, dim)
            X = X.to(device)
            y = y.to(device)
            h = torch.zeros((X.size(1), DIM_LATENT)).to(device)
            loss = 0
            for t in range(X.size(0) - pas_temps):
                ht = h
                h = state.model.one_step(X[t], h)
                for p in range(0,pas_temps+1):
                    ht = state.model.one_step(X[t+p], ht)
                y_pred = state.model.decode(ht)
                y_true = y[t+pas_temps]
                loss += f_cout(y_pred, y_true)
            writer.add_scalar("Loss/test", loss, epoch)
        print("epoch : ", epoch, "loss_test: ", loss)
writer.flush()

print("fin")
