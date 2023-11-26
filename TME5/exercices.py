import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
#from generate import *
import datetime
from pathlib import Path
import sys
import unicodedata
import string
from typing import List
#from torch.utils.data import Dataset, DataLoader
import torch
import re
from textloader import  string2code, id2lettre
import math
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from utils import *

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

test = "C'est. Un. Test."
pred = "C'ess. Un. Tast."
ds = TextDataset(test)
ds_pred = TextDataset(pred)
loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
loader_pred = DataLoader(ds_pred, collate_fn=pad_collate_fn, batch_size=3)
y = None
for x in loader : 
    y = x
y = y.long()
pred = None
for x in loader_pred : 
    pred = x
pred = torch.nn.functional.one_hot(pred.long(), num_classes=len(lettre2id))
pred = torch.Tensor.float(pred)

maskedCrossEntropy(pred, y, PAD_IX)



################################################################################################
########################################    RNN     ############################################
################################################################################################
BATCH_SIZE = 32
PATH = "./data/"

data_trump = DataLoader(
    TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100),
        collate_fn=pad_collate_fn, 
        batch_size= BATCH_SIZE,
        shuffle=True)

dim_input = len(id2lettre)
dim_latent = 50
dim_output = dim_input
dim_emb = 80
epoch = 10

rnn = RNN(dim_input, dim_latent, dim_input, dim_emb)
learn(rnn, data_trump)

generate(rnn, rnn.embedding, rnn.decode, EOS_IX, start="Hello", maxlen=200)


################################################################################################
########################################    LSTM     ###########################################
################################################################################################

BATCH_SIZE = 32
PATH = "/home/ubuntu/Documents/Sorbonne/M2/M2-AMAL/TME4/data/"

data_trump = DataLoader(
    TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100),
    collate_fn=pad_collate_fn, 
    batch_size= BATCH_SIZE,
    shuffle=True)

dim_input = len(id2lettre)
dim_latent = 50
dim_output = dim_input
dim_emb = 80
epoch = 10

lstm = LSTM(dim_input, dim_latent, dim_input, dim_emb)
learn(lstm, data_trump, is_lstm=True)

s = generate(lstm, lstm.embedding, lstm.decode, EOS_IX, start="Hello", maxlen=200,  is_lstm=True)
print(s)

################################################################################################
########################################    GRU      ###########################################
################################################################################################


BATCH_SIZE = 32
PATH = "/home/ubuntu/Documents/Sorbonne/M2/M2-AMAL/TME4/data/"

data_trump = DataLoader(
    TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100),
    collate_fn=pad_collate_fn,
    batch_size= BATCH_SIZE,
    shuffle=True)

dim_input = len(id2lettre)
dim_latent = 50
dim_output = dim_input
dim_emb = 80
epoch = 10

gru = GRU(dim_input, dim_latent, dim_input, dim_emb)
learn(gru, data_trump)

s = generate(gru, gru.embedding, gru.decode, EOS_IX, start="Hello ", maxlen=200)
print(s)


###############################################################################################
########################################    BEAM    ###########################################
###############################################################################################

generate_beam(gru, gru.embedding, gru.decode, id2lettre[EOS_IX], 2, start="Hello", maxlen=10)