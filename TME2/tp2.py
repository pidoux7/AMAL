# import
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import datetime

(x_train, y_train), (x_test, y_test) =tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=1234
)

# Dataset
class Dat(Dataset):
	def __init__(self, x, y):
		super(Dat, self).__init__()
		self.labels = torch.from_numpy(y).double()
		self.data = torch.from_numpy(x).double()
	def __getitem__(self, index):
		return self.data[index], self.labels[index]
	def __len__(self):
		return len(self.labels)

# Normalisation
y_train /= y_train.max()
x_train = normalize(x_train, axis=0)
x_test = normalize(x_test, axis=0)
y_test /= y_test.max()
data_test = Dat(x_test, y_test)
x_test2 = data_test[0][0]
y_test2 = data_test[1][0]
train_dataset = Dat(x_train, y_train)



# Mini-batch
writer = SummaryWriter(comment='mini-batch-size-40')
model = torch.nn.Linear(13, 1).double()
lossfn = torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_dataset, batch_size=40, shuffle=True)
learningRate = 0.0005
epochs = 1000

for e in tqdm(range(epochs)):
    loss_mean = [] # pour calculer ensuite la moyenne des loss sur tous les batches
    loss_mean_test = [] # pour calculer ensuite la moyenne des loss sur le test

    for x_batch, y_batch in trainloader: # batch
        # forward
        y_pred = model(x_batch)
        loss = lossfn(y_pred, y_batch)
        loss_mean.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)
        #backward
        loss.backward()
        #descente de gradient
        with torch.no_grad():
            model.weight -= learningRate * model.weight.grad
            model.bias -= learningRate * model.bias.grad
        model.weight.grad.zero_() #remise a zero des gradients
        model.bias.grad.zero_()
    with torch.no_grad():
        y_pred_test = model(x_test2)
        loss_test = lossfn(y_pred_test, y_test2)
        loss_mean_test.append(loss_test.item())
        writer.add_scalar("Loss/test", loss_test, e)

    if (e % 10) == 0:
        print("epoch %d " % e , "train MSE: ", np.array(loss_mean).mean(), "val MSE: ", np.array(loss_mean_test).mean())
writer.close()



# Batch
writer = SummaryWriter(comment='batch')
model = torch.nn.Linear(13, 1).double()
lossfn = torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_dataset, batch_size=len(x_train), shuffle=True)
learningRate = 0.001
epochs = 1000

for e in tqdm(range(epochs)):
    loss_mean = [] # pour calculer ensuite la moyenne des loss sur tous les batches
    loss_mean_test = [] # pour calculer ensuite la moyenne des loss sur le test

    for x_batch, y_batch in trainloader: # batch
        # forward
        y_pred = model(x_batch)
        loss = lossfn(y_pred, y_batch)
        loss_mean.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)
        #backward
        loss.backward()
        #descente de gradient
        with torch.no_grad():
            model.weight -= learningRate * model.weight.grad
            model.bias -= learningRate * model.bias.grad
        model.weight.grad.zero_() #remise a zero des gradients
        model.bias.grad.zero_()
    with torch.no_grad():
        y_pred_test = model(x_test2)
        loss_test = lossfn(y_pred_test, y_test2)
        loss_mean_test.append(loss_test.item())
        writer.add_scalar("Loss/test", loss_test, e)

    if (e % 10) == 0:
        print("epoch %d " % e , "train MSE: ", np.array(loss_mean).mean(), "val MSE: ", np.array(loss_mean_test).mean())
writer.close()



# Stochastic
writer = SummaryWriter(comment='stochastique')
model = torch.nn.Linear(13, 1).double()
lossfn = torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
learningRate = 0.001
epochs = 1000

for e in tqdm(range(epochs)):
    loss_mean = [] # pour calculer ensuite la moyenne des loss sur tous les batches
    loss_mean_test = [] # pour calculer ensuite la moyenne des loss sur le test

    for x_batch, y_batch in trainloader: # batch
        # forward
        y_pred = model(x_batch)
        loss = lossfn(y_pred, y_batch)
        loss_mean.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)
        #backward
        loss.backward()
        #descente de gradient
        with torch.no_grad():
            model.weight -= learningRate * model.weight.grad
            model.bias -= learningRate * model.bias.grad
        model.weight.grad.zero_() #remise a zero des gradients
        model.bias.grad.zero_()
    with torch.no_grad():
        y_pred_test = model(x_test2)
        loss_test = lossfn(y_pred_test, y_test2)
        loss_mean_test.append(loss_test.item())
        writer.add_scalar("Loss/test", loss_test, e)

    if (e % 10) == 0:
        print("epoch %d " % e , "train MSE: ", np.array(loss_mean).mean(), "val MSE: ", np.array(loss_mean_test).mean())
writer.close()


# On peut voir que la methode batch est la plus rapide suivi de minibatch puis stochastic. 
# En effet la methode batch est la plus rapide car elle ne fait qu'une seule iteration sur l'ensemble des donnees.
# La methode stochastic est la plus lente car elle fait une iteration par donnee.
# De plus avec la metode stochastic on peut voir que la loss est plus instable que les autres methodes.
# l'interet d'utiliser la methode batch est qu'elle est plus rapide et plus stable que stochastic
# mais elle permet de diminuer la possibilité de tomber dans un minimum local contrairement à la méthode batch.
# ici avec une couche lineaire et une MSE, on ne peut pas tomber dans un minimum local car la fonction est convexe.

#optimiseur SGD
writer = SummaryWriter(comment='optim')
model = torch.nn.Linear(13, 1).double()
lossfn = torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True)
learningRate = 0.001
epochs = 1000

optim=torch.optim.SGD(params=model.parameters(), lr=learningRate)
optim.zero_grad()
for e in tqdm(range(epochs)): 
    for x_batch, y_batch in trainloader:
        loss = lossfn(model(x_batch),y_batch)
        loss.backward()
        optim.step()
        optim.zero_grad()
        loss_mean.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)

    with torch.no_grad():
        mult_test = model(x_test2)
        loss_test = lossfn(mult_test, y_test2)
        loss_mean_test.append(loss_test.item())
        writer.add_scalar("Loss/test", loss_test, e)

    if (e % 10) == 0:
        print("epoch %d " % e , "train MSE: ", np.array(loss_mean).mean(), "val MSE: ", np.array(loss_mean).mean())
writer.close()



#optimiseur sequential 
writer = SummaryWriter(comment='sequential')
lin = torch.nn.Linear(13, 5).double()
tanh = torch.nn.Tanh().double()
lin2 = torch.nn.Linear(5, 1).double()
lossfn = torch.nn.MSELoss()
model = torch.nn.Sequential(lin, tanh, lin2)
trainloader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True)
learningRate = 0.01
epochs = 1000

optim=torch.optim.SGD(params=model.parameters(), lr=learningRate)
optim.zero_grad()
for e in tqdm(range(epochs)): 
    for x_batch, y_batch in trainloader:
        loss = lossfn(model(x_batch),y_batch)
        loss.backward()
        optim.step()
        optim.zero_grad()
        loss_mean.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)

    with torch.no_grad():
        mult_test = model(x_test2)
        loss_test = lossfn(mult_test, y_test2)
        loss_mean_test.append(loss_test.item())
        writer.add_scalar("Loss/test", loss_test, e)

    if (e % 10) == 0:
        print("epoch %d " % e , "train MSE: ", np.array(loss_mean).mean(), "val MSE: ", np.array(loss_mean).mean())
writer.close()

# paremetre sont tous mis à jour grace à l'autograd à partir de la loss sur le modele avec sequential