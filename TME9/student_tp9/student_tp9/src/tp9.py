#########################################################################################################
############################################# imports ###################################################
#########################################################################################################


import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy as cp
from typing import List, Tuple, Dict, Set, Union, Iterable, Callable, TypeVar, Optional
from torchmetrics.classification import BinaryAccuracy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorboard


#########################################################################################################
############################################# preprocessing #############################################
#########################################################################################################

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]
    
    @staticmethod
    def collate(batch):
        data = [torch.LongTensor(item[0]) for item in batch]
        labels = [item[1] for item in batch]
        return pad_sequence(data,padding_value=400001 ,batch_first=True), torch.LongTensor(labels)

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

#########################################################################################################
############################################# models ####################################################
#########################################################################################################

class model_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, output_size), # Hidden layer to output layer
            nn.Sigmoid() # Hidden layer to output layer
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.model(x)
        return x



#########################################################################################################
    
class model_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, output_size), # Hidden layer to output layer
            nn.Sigmoid() # Hidden layer to output layer
        )
        self.q = torch.nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        att = nn.functional.softmax(torch.matmul(x,self.q),dim=1) #[batch_size, seq_len, 1]
        x = att.view(batch_size,seq_len,1)*x #[batch_size, seq_len, input_size]
        x = x.sum(dim=1)
        x = self.model(x)
        return x
    

#########################################################################################################
    
class model_3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, output_size), # Hidden layer to output layer
            nn.Sigmoid() # Hidden layer to output layer
        )

        self.q = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        tm = x.mean(dim=1)  # shape: [batch_size, input_size]
        qtm = self.q(tm)    # shape: [batch_size, input_size]
        qtm = qtm.unsqueeze(1)  # shape: [batch_size, 1, input_size]
        
        # Compute attention weights
        att = torch.matmul(x, qtm.transpose(1, 2))  # shape: [batch_size, seq_len, 1]
        att = nn.functional.softmax(att, dim=1)   # shape: [batch_size, seq_len, 1]
        
        # Apply attention weights
        x = self.v(x)  # shape: [batch_size, seq_len, input_size]
        x = x * att  # shape: [batch_size, seq_len, input_size]
        x = x.sum(dim=1)  # shape: [batch_size, input_size]
        
        #MLP
        x = self.model(x) # shape: [batch_size, output_size]
        return x
    
#########################################################################################################

class model_4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, output_size), # Hidden layer to output layer
            nn.Sigmoid() # Hidden layer to output layer
        )

        self.q = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)
        self.gru = nn.GRU(input_size, input_size, batch_first=True)


    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        x, h = self.gru(x)

        tm = x.mean(dim=1)  # shape: [batch_size, input_size]
        qtm = self.q(tm)    # shape: [batch_size, input_size]
        qtm = qtm.unsqueeze(1)  # shape: [batch_size, 1, input_size]
        
        # Compute attention weights
        att = torch.matmul(x, qtm.transpose(1, 2))  # shape: [batch_size, seq_len, 1]
        att = nn.functional.softmax(att, dim=1)   # shape: [batch_size, seq_len, 1]
        
        # Apply attention weights
        x = self.v(x)  # shape: [batch_size, seq_len, input_size]
        x = x * att  # shape: [batch_size, seq_len, input_size]
        x = x.sum(dim=1)  # shape: [batch_size, input_size]
        
        #MLP
        x = self.model(x) # shape: [batch_size, output_size]
        return x

#########################################################################################################

class model_5(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, output_size), # Hidden layer to output layer
            nn.Sigmoid() # Hidden layer to output layer
        )

        self.q = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)
        self.gru = nn.GRU(input_size, input_size, batch_first=True)


    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        x, h = self.gru(x)

        tm = x.mean(dim=1)  # shape: [batch_size, input_size]
        qtm = self.q(tm)    # shape: [batch_size, input_size]
        qtm = qtm.unsqueeze(1)  # shape: [batch_size, 1, input_size]
        
        # Compute attention weights
        att = torch.matmul(x, qtm.transpose(1, 2))  # shape: [batch_size, seq_len, 1]
        att = nn.functional.softmax(att, dim=1)   # shape: [batch_size, seq_len, 1]
        att_entropy = -torch.sum(att * torch.log(att + 1e-9), dim=1)  # Add a small value to avoid log(0)

        
        # Apply attention weights
        x = self.v(x)  # shape: [batch_size, seq_len, input_size]
        x = x * att  # shape: [batch_size, seq_len, input_size]
        x = x.sum(dim=1)  # shape: [batch_size, input_size]
        
        #MLP
        x = self.model(x) # shape: [batch_size, output_size]
        return x,att_entropy

#########################################################################################################
############################################# training #################################################
#########################################################################################################

def training(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        cpt = 0
        for X,y in train_loader:
            X = X.to(device)
            y = y.float().to(device)
            X_emb = emb(X)
            y_pred = model(X_emb)
            loss = criterion(y_pred.view(-1), y)
            total_loss += loss.item()
            acc_train(y_pred.view(-1),y)
            total_acc += acc_val(y_pred.view(-1),y)
            cpt+=1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/train", total_acc/cpt, epoch)
        
        print(f"\n Epoch {epoch + 1}/{epochs} :")
        print(f'loss_train = {total_loss:.4f}')
        print(f'acc_train = {total_acc/cpt:.4f}')

        model.eval()
        total_loss = 0
        total_acc = 0
        cpt =0
        with torch.no_grad():
            for X,y in val_loader:
                X = X.to(device)
                y = y.float().to(device)
                X_emb = emb(X)
                y_pred = model(X_emb)
                loss = criterion(y_pred.view(-1), y)
                total_loss += loss.item()
                total_acc += acc_val(y_pred.view(-1),y)
                cpt+=1

        writer.add_scalar("Loss/val", total_loss, epoch)
        writer.add_scalar("Accuracy/val", total_acc/cpt, epoch)
        print(f"Epoch {epoch + 1}/{epochs} :")
        print(f'loss_test = {total_loss:.4f}')
        print(f'acc_test = {total_acc/cpt:.4f}')


#########################################################################################################
############################################# data ######################################################
#########################################################################################################


dic, embeddings,train,test = get_imdb_data()
dic['pad'] = 400001
embeddings = np.vstack((np.array(embeddings),np.zeros((1,50))))
train_dataset = DataLoader(train, collate_fn= FolderText.collate, batch_size=32, shuffle=True)
test_dataset = DataLoader(test,collate_fn=FolderText.collate, batch_size=32, shuffle=True)


#########################################################################################################
######################################## main exercice 1 #####################################################
#########################################################################################################

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_size = 50
hidden_size = 25
output_size = 1
lr = 0.001

emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings)).to(device).requires_grad_(False)
mlp = model_1(input_size, hidden_size, output_size).to(device)
critere = nn.BCELoss()
optim = torch.optim.Adam(mlp.parameters(), lr=lr)

writer = SummaryWriter()
acc_train = BinaryAccuracy().to(device)
acc_val = BinaryAccuracy().to(device)

#training(mlp,train_dataset,test_dataset,critere,optim,10)
'''
#########################################################################################################
######################################## main exercice 2 #####################################################
#########################################################################################################
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_size = 50
hidden_size = 25
output_size = 1
lr = 0.001

emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings)).to(device).requires_grad_(False)
mlp2 = model_2(input_size, hidden_size, output_size).to(device)
critere = nn.BCELoss()
optim = torch.optim.Adam(mlp2.parameters(), lr=lr)

writer = SummaryWriter()
acc_train = BinaryAccuracy().to(device)
acc_val = BinaryAccuracy().to(device)

training(mlp2,train_dataset,test_dataset,critere,optim,10)
'''

#########################################################################################################
######################################## main exercice 3 #####################################################
#########################################################################################################
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_size = 50
hidden_size = 50
output_size = 1
lr = 0.001

emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings)).to(device).requires_grad_(False)
mlp3 = model_3(input_size, hidden_size, output_size).to(device)
critere = nn.BCELoss()
optim = torch.optim.Adam(mlp3.parameters(), lr=lr)

writer = SummaryWriter()
acc_train = BinaryAccuracy().to(device)
acc_val = BinaryAccuracy().to(device)

training(mlp3,train_dataset,test_dataset,critere,optim,10)
'''

#########################################################################################################
######################################## main exercice 4 #####################################################
#########################################################################################################
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_size = 50
hidden_size = 50
output_size = 1
lr = 0.001

emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings)).to(device).requires_grad_(False)
mlp4 = model_4(input_size, hidden_size, output_size).to(device)
critere = nn.BCELoss()
optim = torch.optim.Adam(mlp4.parameters(), lr=lr)

writer = SummaryWriter()
acc_train = BinaryAccuracy().to(device)
acc_val = BinaryAccuracy().to(device)

training(mlp4,train_dataset,test_dataset,critere,optim,10)
'''

#########################################################################################################
######################################## main exercice 5 #####################################################
#########################################################################################################
def training_entropy(model, train_loader, val_loader, criterion, optimizer, epochs,lambda_entropy=0.01):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        entropy_losses = 0
        total_losses = 0
        cpt = 0
        for X,y in train_loader:
            X = X.to(device)
            y = y.float().to(device)
            X_emb = emb(X)
            y_pred, att_entropy  = model(X_emb)
            loss = criterion(y_pred.view(-1), y)
            entropy_loss = lambda_entropy * att_entropy.mean()
            losses = loss / entropy_loss
            total_loss += loss.item()
            entropy_losses += entropy_loss.item()
            total_losses += losses.item()
            acc_train(y_pred.view(-1),y)
            total_acc += acc_val(y_pred.view(-1),y)
            cpt+=1


            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Loss/entropy_train", entropy_losses, epoch)
        writer.add_scalar("Loss/total_train", total_losses, epoch)
        writer.add_scalar("Accuracy/train", total_acc/cpt, epoch)
        
        print(f"\n Epoch {epoch + 1}/{epochs} :")
        print(f'loss_train = {total_loss:.4f}')
        print(f'entropy_loss = {entropy_losses:.4f}')
        print(f'total_loss = {total_losses:.4f}')
        print(f'acc_train = {total_acc/cpt:.4f}')

        model.eval()
        total_loss = 0
        total_acc = 0
        cpt =0
        with torch.no_grad():
            for X,y in val_loader:
                X = X.to(device)
                y = y.float().to(device)
                X_emb = emb(X)
                y_pred,att_entropy = model(X_emb)
                loss = criterion(y_pred.view(-1), y)
                entropy_loss = lambda_entropy * att_entropy.mean()
                losses = loss / entropy_loss
                total_loss += loss.item()
                entropy_losses += entropy_loss.item()
                total_losses += losses.item()
                total_acc += acc_val(y_pred.view(-1),y)
                cpt+=1

        writer.add_scalar("Loss/val", total_loss, epoch)
        writer.add_scalar("Accuracy/val", total_acc/cpt, epoch)
        writer.add_scalar("Loss/entropy_val", entropy_losses, epoch)
        writer.add_scalar("Loss/total_val", total_losses, epoch)
        print(f"Epoch {epoch + 1}/{epochs} :")
        print(f'loss_test = {total_loss:.4f}')
        print(f'acc_test = {total_acc/cpt:.4f}')






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_size = 50
hidden_size = 50
output_size = 1
lr = 0.001
lambda_entropy = 0.01

emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings)).to(device).requires_grad_(False)
mlp5 = model_5(input_size, hidden_size, output_size).to(device)
critere = nn.BCELoss()
optim = torch.optim.Adam(mlp5.parameters(), lr=lr)

writer = SummaryWriter()
acc_train = BinaryAccuracy().to(device)
acc_val = BinaryAccuracy().to(device)

training_entropy(mlp5,train_dataset,test_dataset,critere,optim,10)