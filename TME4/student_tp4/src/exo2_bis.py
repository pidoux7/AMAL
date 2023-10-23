from utils import RNN, SampleMetroDataset
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class State :
    def __init__(self, model, optim) :
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0,0

# Nombre de stations utilisé
CLASSES = 3
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Chargement des données
PATH = "/home/pidoux/master/deepdac/AMAL/TME4/data/"

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
accuracy_train = torchmetrics.classification.Accuracy(task="multiclass", num_classes=CLASSES).to(device)
accuracy_test = torchmetrics.classification.Accuracy(task="multiclass", num_classes=CLASSES).to(device)

#INITIALISATION DU RNN
latent_dim = 15
nb_epochs = 20
lr = 0.001
f_cout = nn.CrossEntropyLoss()


print(f"running on {device}")
savepath = Path("model_classes_3.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)
#creer un nouvel état à partir d'un modèle et d'un optimiseur
else:
    model = RNN(input_dim=DIM_INPUT, latent_dim=latent_dim, output_dim=CLASSES, activation = nn.Tanh(), decode_activation = nn.Softmax())
    model = model.to(device)
    optim= torch.optim.Adam(model.parameters(), lr=lr)
    state = State(model, optim)

# APPRENTISSAGE
for epoch in tqdm(range(nb_epochs)):
    list_loss = []
    for X, y in data_train:
        X = X.to(device)
        y = y.to(device)
        state.optim.zero_grad()
        h = torch.zeros((X.size(0),latent_dim)).to(device)
        h = state.model.forward(X, h).to(device)
        y_pred = state.model.decode(h[:, -1]).to(device)
        loss = f_cout(y_pred, y)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy_train(y_pred.argmax(1), y), epoch)
        accuracy_train.reset()
        loss.backward()
        state.optim.step()
    with savepath.open("wb") as fp:
        state.epoch += 1
        torch.save(state, fp)

#TEST
    with torch.no_grad():
        list_loss = []
        for X, y in data_test:
            X = X.to(device)
            y = y.to(device)
            h = torch.zeros(X.size(0),latent_dim).to(device)
            h = state.model.forward(X, h).to(device)
            y_pred = state.model.decode(h[:,-1]).to(device)
            loss = f_cout(y_pred, y)
            writer.add_scalar("Loss/test", loss, epoch)
            writer.add_scalar("Accuracy/test", accuracy_train(y_pred.argmax(1), y), epoch)


print("fin")
