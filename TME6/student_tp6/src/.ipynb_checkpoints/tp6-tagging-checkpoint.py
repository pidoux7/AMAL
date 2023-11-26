import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)
from torch.optim.lr_scheduler import StepLR
import datetime
from pathlib import Path
ds = prepare_dataset('org.universaldependencies.french.gsd')


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]


class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx =0)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output


class State :
    def __init__(self, model, optim,scheduler) :
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0,0
        self.scheduler = scheduler


def deactivate_words(X, oov_rate, pad_token_id):
    # Generate a random tensor with the same shape as X
    random_values = torch.rand(X.shape).to(device)
    
    # Create a mask for the random values less than oov_rate
    # and where X is not equal to pad_token_id (assuming PAD tokens should not be masked)
    mask = (random_values < oov_rate).to(device) * (X != pad_token_id).to(device)
    
    # Replace masked values with the ID for OOV (which is 1 here)
    modified_X = X.masked_fill(mask, 1)
    
    return modified_X


def train_model(model, train_loader, criterion, optimizer, epoch,writer,oov_rate, pad_tokenID ):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs,targets = inputs.to(device), targets.to(device)
        inputs = deactivate_words(inputs,oov_rate, pad_tokenID)
        outputs = model(inputs)
        out = torch.transpose(outputs,1,2)
        #print('inputs', inputs.shape)
        #print('outputs', outputs.shape)
        #print('out', out.shape)
        #print('targets',targets.shape)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item() ,  epoch*len(train_loader) +i )
        writer.add_scalar("Accuracy/train", accuracy(soft(outputs.to('cpu')).argmax(2),targets.to('cpu')) ,  epoch * len(train_loader)+i )
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
def evaluate_model(model, train_loader, criterion, epoch,writer):
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            inputs,targets = inputs.to(device), targets.to(device)
            inputs = deactivate_words(inputs,oov_rate, pad_tokenID)
            outputs = model(inputs)
            out = torch.transpose(outputs,1,2)
            loss = criterion(out, targets)
            writer.add_scalar("Loss/test", loss.item() ,  epoch*len(train_loader) +i )
            writer.add_scalar("Accuracy/test", accuracy(soft(outputs.to('cpu')).argmax(2),targets.to('cpu')) ,  epoch * len(train_loader)+i )
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')






logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)
logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100
train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


#  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)
# paramètre
writer = SummaryWriter("seq/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(tags))
PATH = '/home/pidoux/master/deepdac/AMAL/TME4/data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 80
hidden_size = 50
output_size = len(tags)
vocab_size = len(words)
nb_epochs = 5
oov_rate = 0.1
lr = 0.01
pad_tokenID = 0
criterion = nn.CrossEntropyLoss()
soft = nn.Softmax(dim=-1) ###### peut etre à changer
print(f"running on {device}")

savepath = Path("seq2seq.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)
else:
    model = Seq2SeqModel(input_size, hidden_size, output_size, vocab_size)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optim, step_size=1, gamma=0.9)
    state = State(model, optim, scheduler)


for epoch in tqdm(range(nb_epochs)):
    train_model(state.model, train_loader, criterion, state.optim, epoch, writer, oov_rate,pad_tokenID)
    evaluate_model(state.model, train_loader, criterion, epoch, writer)
    state.scheduler.step()
    
print('fin')               