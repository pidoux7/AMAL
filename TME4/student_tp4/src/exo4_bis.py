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

def tirage(Y_pred):
    """ Tirage d'un caractère parmis les plus probables. """
    return torch.multinomial(Y_pred,1).item()

savepath = Path("trump.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)

nb_characters = 100
DIM_INPUT_FIRST = len(id2lettre)
DIM_INPUT = 96
DIM_LATENT = 50
DIM_OUTPUT = DIM_INPUT_FIRST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f_cout = nn.CrossEntropyLoss()
with torch.no_grad():
    X = torch.tensor(F.one_hot(string2code("c"),num_classes = DIM_INPUT_FIRST)).reshape(1,1,-1).to(device).float().transpose(0,1)
    h = torch.zeros((1, DIM_LATENT)).to(device)

    embedding = state.model[1](state.model[0](X)).to(device)
    #print('embed',embedding.shape)
    phrase_argmax = []
    phrase_bin = []
    for i in range(nb_characters):
        h = state.model[2].one_step(embedding,h).to(device)
        y_pred = state.model[2].decode(h).to(device)
        embedding = state.model[1](state.model[0](y_pred)).to(device)
        l = int(tirage(y_pred[0,0,:]))
        phrase_bin.append(l)
        lettre = int(y_pred.argmax(dim=2)[0,0])
        phrase_argmax.append(lettre)
    print(phrase_bin)
    p_binomial = code2string(phrase_bin)
    p_argmax= code2string(phrase_argmax)
    print(f'Binomial : Token_début {p_binomial} Token_fin')
    print(f'Argmax   : Token_début {p_argmax} Token_fin')
