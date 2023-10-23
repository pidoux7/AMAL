from utils2 import RNN, SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics


# Nombre de stations utilisé
CLASSES = 2
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


#INITIALISATION DU RNN
latent_dim = 10
rnn = RNN(input_dim=DIM_INPUT, latent_dim=latent_dim, output_dim=CLASSES, activation = nn.Tanh(), decode_activation = nn.Softmax())

# init APPRENTISSAGE
rnn.to(device)
nb_epochs = 10
lr = 0.001
f_cout = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
list_loss_train = []
list_loss_test = []
list_accuracy_train = []
list_accuracy_test = []
accuracy_train = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5).to(device)
accuracy_test = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5).to(device)

# APPRENTISSAGE
for epoch in tqdm(range(nb_epochs)):
    list_loss = []
    for X, y in data_train:
        X = torch.transpose(X ,0,1)
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        h = torch.zeros((X.size(1),latent_dim)).to(device)
        h = rnn.forward(X, h).to(device)
        y_pred = rnn.decode(h[-1, :])
        loss = f_cout(y_pred, y)
        list_loss.append(loss.item())
        accuracy_train(y_pred.argmax(1), y)
        loss.backward()
        optimizer.step()
    list_loss_train.append(np.mean(list_loss))
    list_accuracy_train.append(float(accuracy_train.compute()))

#TEST
    with torch.no_grad():
        list_loss = []
        for X, y in data_test:
            X = torch.transpose(X ,0,1)
            X = X.to(device)
            y = y.to(device)
            h = torch.zeros(X.size(1),latent_dim).to(device)
            h = rnn.forward(X, h).to(device)
            y_pred = rnn.decode(h[-1,:]).to(device)
            loss = f_cout(y_pred, y)
            list_loss.append(loss.item())
            accuracy_test(y_pred.argmax(1), y)
        list_loss_test.append(np.mean(list_loss))
        list_accuracy_test.append(float(accuracy_test.compute()))

#COUT EN APPRENTISSAGE ET EN TEST
plt.figure(figsize=(7,5))
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("Loss (CrossEntropy) et Accuracy")
plt.plot(list_loss_train, label = "L_train")
plt.plot(list_loss_test, label = "L_test")
plt.plot(list_accuracy_train, label = "A_train")
plt.plot(list_accuracy_test, label = "A_test")
plt.legend()
plt.show()

print("fin")
accuracy_train.reset()
accuracy_test.reset()


