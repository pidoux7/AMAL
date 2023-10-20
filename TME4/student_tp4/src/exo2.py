from utils import RNN, SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

latent_dim = 15
rnn = RNN(input_dim=DIM_INPUT, latent_dim=latent_dim, output_dim=CLASSES, activation = nn.Tanh(), decode_activation = nn.Softmax())
#rnn.to(device)

#TODO writer 

#APPRENTISSAGE
nb_epochs = 10
lr = 0.001
f_cout = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
list_loss_train = []
for epoch in tqdm(range(nb_epochs)):
    list_loss = []
    h = torch.zeros(BATCH_SIZE,latent_dim) #TODO avant epoch ?
    for X, y in data_train:
        print(X.shape)
        print(h.shape)
        optimizer.zero_grad()
        #X = X.to(device)
        last_h = rnn.forward(X, h)
        y_pred = rnn.decode(last_h)
        loss = f_cout(y_pred, y)
        list_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    list_loss_train.append(np.mean(list_loss))

#TEST
list_loss_test = []
for epoch in tqdm(range(nb_epochs)):
    list_loss = []
    h = torch.zeros(BATCH_SIZE,latent_dim) #TODO avant epoch ?
    for X, y in data_test:
        #X = X.to(device)
        optimizer.zero_grad()
        last_h = rnn.forward(X, h)
        y_pred = rnn.decode(last_h)
        loss = f_cout(y_pred, y)
        list_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    list_loss_test.append(np.mean(list_loss))

#COUT EN APPRENTISSAGE ET EN TEST
plt.figure(figsize=(7,5))
plt.title("Loss")
plt.xlabel("episode")
plt.ylabel("Loss (CrossEntropy)")
plt.plot(list_loss_train, label = "train")
plt.plot(list_loss_test, label = "test")
plt.legend()
plt.show()

print("fin")





