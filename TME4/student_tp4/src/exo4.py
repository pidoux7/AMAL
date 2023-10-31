import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from utils2 import RNN, State
import numpy as np
import datetime
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import torchmetrics

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]

writer = SummaryWriter("trump"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

PATH = '/home/pidoux/master/deepdac/AMAL/TME4/data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Taille du batch
BATCH_SIZE = 256

# Chargement des données
data_trump = DataLoader(
    TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),
                    maxlen=-30), 
                    batch_size= BATCH_SIZE, shuffle=True)

# On crée le réseau
DIM_INPUT_FIRST = len(id2lettre)
print(DIM_INPUT_FIRST)
DIM_INPUT = 96
DIM_LATENT = 50
DIM_OUTPUT = DIM_INPUT_FIRST
nb_epochs = 5
lr = 0.01
f_cout = nn.CrossEntropyLoss()
accuracy_train = torchmetrics.classification.Accuracy(task="multiclass", num_classes=DIM_INPUT_FIRST).to(device)

print(f"running on {device}")
savepath = Path("trump.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)
#creer un nouvel état à partir d'un modèle et d'un optimiseur
else:
    Lin1 = nn.Linear(DIM_INPUT_FIRST, DIM_INPUT)
    Tanh = nn.Tanh()
    rnn = RNN(input_dim=DIM_INPUT, 
                latent_dim=DIM_LATENT, 
                output_dim=DIM_OUTPUT, 
                activation = nn.Tanh(), 
                decode_activation = nn.Softmax(dim=-1))
    model = nn.Sequential(Lin1, Tanh, rnn)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    state = State(model, optim)

print(f"running on {device}")

for epoch in tqdm(range(nb_epochs)):
    for X, y in data_trump:
        X = F.one_hot(X, num_classes=DIM_INPUT_FIRST).to(device).float().transpose(0,1)
        y = F.one_hot(y, num_classes=DIM_INPUT_FIRST).to(device).float().transpose(0,1)
        #print(X.size(), y.size())
        state.optim.zero_grad()
        embedding = state.model[1](state.model[0](X))
        #print(embedding.size())
        h = torch.zeros((embedding.size(1), DIM_LATENT)).to(device)
        loss = 0
        for t in tqdm(range(embedding.size(0))):
            h = state.model[2].one_step(embedding[t], h)
            y_pred = state.model[2].decode(h)
            y_true = y[t]
            loss += f_cout(y_pred, y_true)
            #print(y_pred.size(), y_true.size())
            #print(y_pred.argmax(-1).size(), y_true.argmax(-1).size())
            #rint(y_pred.argmax(-1), y_true.argmax(-1))
            writer.add_scalar("Accuracy/train", accuracy_train(y_pred.argmax(-1), y_true.argmax(-1)), epoch)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        state.optim.step()
        writer.add_scalar("Loss/train", loss, epoch)
        print("epoch : ", epoch, "loss_train: ", loss)
        
        with savepath.open("wb") as fp:
            state.epoch += 1
            torch.save(state, fp)