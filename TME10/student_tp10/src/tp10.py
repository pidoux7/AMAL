#########################################################################################################
############################################# imports ###################################################
#########################################################################################################

import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy
from utils import *

#########################################################################################################
############################################# preprocessing #############################################
#########################################################################################################

MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)

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
    def get_txt(self,ix):
        s = self.files[ix]
        return s if isinstance(s,str) else s.read_text(), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        'edu.stanford.glove.6b.%d' % embedding_size).load()
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

############################################# model 1 ####################################################

class Self_attention(nn.Module):
    def __init__(self, embed_dim, c=0):
        super().__init__()
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.soft = nn.Softmax(dim=1)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.c = torch.tensor(c).requires_grad_(False)
        self.relu = nn.ReLU()
        self._embed_dim = torch.tensor(embed_dim).requires_grad_(False)
    
    def softmax_masque(self, x, l):
        # faire un masque qui ne prenne pas en compte les tokens de padding
        # x: (batch_size, seq_len, emb_size)
        # l: (batch_size)
        #print('x.shape',x.shape)
        #print('l.shape',l.shape)
        #print(l.device)
        #print(x.device)
        mask = torch.arange(x.size(1))[None, :].to("cuda") >= l[:, None].to("cuda") #> ou >= ???????????
        #print('mask.shape',mask.shape)
        #print(mask.device)
        mask = mask.unsqueeze(2).expand(x.size())
        x[mask] = float('-inf')

        return self.soft(x)

    def forward(self, x, l):
        # x: (batch_size, seq_len_false, emb_size)
        # l: (seq_len_true)

        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)
        #print('x_lin.shape',x.shape)
        #print('q.shape',q.shape)
        #print('k.shape',k.shape)
        #print('v.shape',v.shape)

        # quand on appelle le soft-mmax il faut utiliser un mask afin d'appliquer
        # le softmax jusqu'a la fin de la phrase mais pas au dela (on ne prend pas
        # en compte les tokens de padding) 
        attention = self.softmax_masque(self.c + torch.div(q @ torch.transpose(k, dim0=1, dim1=2), torch.sqrt(self._embed_dim)),l)
        #print('attention.shape',attention.shape)
        
        x = self.lin( attention @ v )
        #print('att.shape',x.shape)
        x = self.relu(x)

        return x


class Model_attention(nn.Module):
    def __init__(self, embed_dim, c=0):
        super().__init__()
        self.Att1 = Self_attention(embed_dim, c)
        self.Att2 = Self_attention(embed_dim, c)

        self.Att3 = Self_attention(embed_dim, c)
        self.lin = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._embed_dim = embed_dim
        self.c = c        

    def forward(self, x, l):
        #print('x_entrée.shape',x.shape)
        x = self.Att1(x, l)
        x = self.Att2(x, l)
        x = self.Att3(x, l)
        #print('x_sortie.shape',x.shape)
        x = torch.mean(x, dim=1)
        x = self.lin(x)
        #print('x_lin_sortie.shape',x.shape)
        x = self.sigmoid(x)
        #print('x_sigmoid_sortie.shape',x.shape)

        return x

############################################# model 2 ####################################################
class Self_attention2(nn.Module):
    def __init__(self, embed_dim, c=0):
        super().__init__()
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.soft = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)
        self.c = torch.tensor(c).requires_grad_(False)
        self.relu = nn.ReLU()
        self._embed_dim = torch.tensor(embed_dim).requires_grad_(False)
    
    def softmax_masque(self, x, l):
        mask = torch.arange(x.size(1))[None, :].to("cuda") >= l[:, None].to("cuda")
        mask = mask.unsqueeze(2).expand(x.size())
        x[mask] = float('-inf')
        return self.soft(x)
    
    def forward(self, x, l):
        x = self.norm(x)
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)
        attention = self.softmax_masque(self.c + torch.div(q @ torch.transpose(k, dim0=1, dim1=2), torch.sqrt(self._embed_dim)),l)
        attention = self.lin( attention @ v )
        attention = self.relu(x)
        x = self.lin2(attention + x)
        return x


class Model_attention2(nn.Module):
    def __init__(self, embed_dim, c=0):
        super().__init__()
        self.Att1 = Self_attention2(embed_dim, c)
        self.Att2 = Self_attention2(embed_dim, c)
        self.Att3 = Self_attention2(embed_dim, c)
        self.lin = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._embed_dim = embed_dim
        self.c = c        

    def forward(self, x, l):
        x = self.Att1(x, l)
        x = self.Att2(x, l)
        x = self.Att3(x, l)
        x = torch.mean(x, dim=1)
        x = self.lin(x)
        x = self.sigmoid(x)

        return x



#########################################################################################################
############################################# training #################################################
#########################################################################################################

def train(model, train_loader, test_loader, optimizer, criterion, epochs, device, writer, acc_train, acc_test):
    for epoch in tqdm(range(epochs)):
        print(f"\n\n\n Epoch {epoch + 1}/{epochs} :")
        model.train()
        total_loss = 0
        total_acc = 0
        cpt = 0
        for x,y,l in tqdm(train_loader):
            x,y,l = x.to(device), y.float().to(device), l.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(x,l).to(device)
            loss = criterion(y_pred.squeeze(), y)
            total_loss += loss.item()
            total_acc += acc_train(y_pred.squeeze(),y)
            cpt += 1
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/train", total_acc/cpt, epoch)
        print(f'loss_train = {total_loss:.4f}')
        print(f'acc_train = {total_acc/cpt:.4f}')

        model.eval()
        total_loss = 0
        total_acc = 0
        cpt = 0
        with torch.no_grad():
            for x,y,l in tqdm(test_loader):
                x,y,l = x.to(device), y.float().to(device), l.to(device)
                y_pred = model(x,l)
                loss = criterion(y_pred.squeeze(), y)
                total_loss += loss.item()
                total_acc += acc_test(y_pred.squeeze(),y)
                cpt += 1
            writer.add_scalar("Loss/test", total_loss, epoch)
            writer.add_scalar("Accuracy/test", total_acc/cpt, epoch)
            print(f'loss_test = {total_loss:.4f}')
            print(f'acc_test = {total_acc/cpt:.4f}')



#########################################################################################################
######################################## main exercice #####################################################
#########################################################################################################

@click.command()
@click.option('--test-iterations', default=1000, type=int, help='Number of training iterations (batches) before testing')
@click.option('--epochs', default=50, help='Number of epochs.')
@click.option('--modeltype', required=True, type=int, help="0: base, 1 : Attention1, 2: Attention2")
@click.option('--emb_size', default=100, help='embeddings size')
@click.option('--batch_size', default=20, help='batch size')


def main(epochs, test_iterations, modeltype, emb_size, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))

    def collate(batch):
        """ Collate function for DataLoader """
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return emb_layer(torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD)).to(device), torch.LongTensor(labels).to(device), torch.Tensor(lens).to(device)


    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=collate,shuffle=False)
    
    # hyperparametres
    c = 0
    if modeltype == 1:
        model = Model_attention(emb_size, c).to(device)
    elif modeltype == 2:
        model = Model_attention2(emb_size, c).to(device)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    acc_train = BinaryAccuracy().to(device)
    acc_test = BinaryAccuracy().to(device)

    train(model, train_loader, test_loader, optimizer, criterion, epochs, device, writer, acc_train, acc_test)


if __name__ == "__main__":
    main()
