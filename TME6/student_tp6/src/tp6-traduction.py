import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import datetime
import numpy as np
import time
import re
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)

FILE = "data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
class Encoder(nn.Module):
    def __init__(self, vocab_enc, dim_latent_enc, dim_hidden_enc, pad_index):
        super(Encoder, self).__init__()
        self.enc_emb =  nn.Embedding(vocab_enc, dim_latent_enc, padding_idx=pad_index) 
        self.enc_gru = nn.GRU(dim_latent_enc, dim_hidden_enc)
    
    def forward(self, x):
        x_emb = self.enc_emb(x)
        _, h_n = self.enc_gru(x_emb)
        return h_n

class Decoder(nn.Module):
    def __init__(self, vocab_dec, dim_latent_dec, dim_hidden_dec, pad_index) :
        super(Decoder, self).__init__()
        self.vocab_dec = vocab_dec
        self.dec_emb =  nn.Embedding(vocab_dec, dim_latent_dec, padding_idx = pad_index) 
        self.dec_gru = nn.GRU(dim_latent_dec, dim_hidden_dec)
        self.decode = nn.Linear(dim_hidden_dec, vocab_dec) 
    
    def forward(self, x, hidden):
        emb = self.dec_emb(x)
        _, h_n = self.dec_gru(emb, hidden)
        dec = self.decode(h_n)   
        return h_n, dec 
    
    def generate(self, hidden, lenseq=None, use_teacher_forcing=False, target=None):
        sos = Vocabulary.SOS
        eos = Vocabulary.EOS

        batch_size = hidden.shape[1] 
                
        #trad = torch.full((1, batch_size), sos, dtype=torch.long, device=hidden.device)
        #trad = torch.nn.functional.one_hot(trad, num_classes=self.vocab_dec)
        trad = []
        x = torch.full((1, batch_size), sos, dtype=torch.long, device=hidden.device)

        ht = hidden 
        i = 0
        cpt_eos = 0
        
        while lenseq==None or i<lenseq :
            ht, dec = self.forward(x, ht)
            output = nn.functional.softmax(dec, dim=-1)

            x = torch.argmax(output, axis = 2).reshape(1,-1)
            
            if use_teacher_forcing : 
                #trad = torch.cat((trad, dec), dim = 0)
                trad.append(dec)
                x = target[i,:].reshape(1,-1)
            else : 
                #trad = torch.cat((trad, dec), dim = 0)
                trad.append(dec)

            cpt_eos += torch.sum(x==eos).item()
            if cpt_eos ==  batch_size: 
                break
            i+=1

        #return trad[1:]
        trad = torch.cat(trad, dim=0)
        return trad
    

def train(encoder, decoder, criterion, train_loader, test_loader, teacher_forcing_prob = 0.5 , lr=0.3, nb_epoch = 50 ):

    writer = SummaryWriter("traduction/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params = parameters, lr = lr)


    liste_loss_train = []
    liste_loss_val = []
    for epoch in tqdm(range(nb_epoch)):
        
        liste_loss_batch = []

        for input_seq, idx_pad_input, target_seq, idx_pad_target in train_loader:
            input_seq, target_seq, idx_pad_target = input_seq.to(device), target_seq.to(device), idx_pad_target.to(device)
            optimizer.zero_grad()
            
            use_teacher_forcing_proba = (nb_epoch-epoch-1) / nb_epoch
            use_teacher_forcing = (torch.rand(1).item() <= use_teacher_forcing_proba)

            hidden = encoder(input_seq)
            yhat = decoder.generate(hidden, lenseq=torch.max(idx_pad_target), use_teacher_forcing=use_teacher_forcing, target=target_seq)
            
            pad_rows = target_seq.size(0) - yhat.size(0)
            if pad_rows > 0:
                padding = torch.full((pad_rows, yhat.size(1), yhat.size(2)), Vocabulary.PAD, dtype=yhat.dtype).to(device)
                yhat = torch.cat((yhat, padding), dim=0)
            
            yhat = torch.transpose(yhat,1,2)

            loss = criterion(yhat, target_seq)

            writer.add_scalar("Loss/train", loss, epoch)
            
            loss.backward()
            
            optimizer.step()
            
            with torch.no_grad():
                liste_loss_batch.append(loss.item())
            
        liste_loss_train.append(np.mean(liste_loss_batch))



        with torch.no_grad():
            
            liste_loss_batch = []

            for input_seq, idx_pad_input, target_seq, idx_pad_target in test_loader:
                input_seq, target_seq, idx_pad_target = input_seq.to(device), target_seq.to(device), idx_pad_target.to(device)

                use_teacher_forcing_proba = (nb_epoch-epoch-1) / nb_epoch
                use_teacher_forcing = (torch.rand(1).item() <= use_teacher_forcing_proba)

                hidden = encoder(input_seq).to(device)
                yhat = decoder.generate(hidden, lenseq=torch.max(idx_pad_target), use_teacher_forcing=use_teacher_forcing, target=target_seq)

                pad_rows = target_seq.size(0) - yhat.size(0)
                if pad_rows > 0:
                    padding = torch.full((pad_rows, yhat.size(1), yhat.size(2)), Vocabulary.PAD, dtype=yhat.dtype).to(device)
                    yhat = torch.cat((yhat, padding), dim=0)
                
                yhat = torch.transpose(yhat,1,2)

                loss = criterion(yhat, target_seq)

                writer.add_scalar("Loss/test", loss, epoch)
                            
                liste_loss_batch.append(loss.item())
                    
            liste_loss_val.append(np.mean(liste_loss_batch))



        plt.figure()
        plt.plot(np.arange(len(liste_loss_train)), liste_loss_train, label='Loss train', color='tab:orange')
        plt.plot(np.arange(len(liste_loss_val)), liste_loss_val, label='Loss val', color='tab:blue')
        plt.xlabel("Epochs")
        plt.title("Loss en train et en validation")
        plt.legend(loc='upper left')
        plt.show()


vocab_enc = vocEng.__len__()  
dim_latent_enc = 60
dim_hidden = 40

vocab_dec = vocFra.__len__() 
dim_latent_dec =  60

pad_index = Vocabulary.PAD
lr = 1e-2


encoder = Encoder(vocab_enc, dim_latent_enc, dim_hidden, pad_index).to(device)
decoder = Decoder(vocab_dec, dim_latent_dec, dim_hidden, pad_index).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
train(encoder, decoder, criterion, train_loader, test_loader)

def traduction(sentence, encoder, decoder):
    x = torch.tensor([vocEng.__getitem__(w) for w in sentence.split()]).reshape(-1,1).to(device)
    print(x)
    hidden = encoder(x)
    print(hidden)
    trad = decoder.generate(hidden, lenseq=20)
    print(trad)
    trad = torch.softmax(trad, dim=-1)
    trad = torch.argmax(trad, axis = 2).reshape(-1)
    return " ".join(vocFra.getwords(trad))

sentence = "hello i love cats and also dogs"
trad = traduction(sentence, encoder, decoder)
print(trad)