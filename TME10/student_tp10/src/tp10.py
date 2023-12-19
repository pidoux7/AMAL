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

#  TODO: 

class Self_attention(nn.Module):
    def __init__(self, embed_dim, c=0):
        super(Self_attention, self).__init__()
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.soft = nn.Softmax(dim=1)
        self.lin_final = nn.Linear(embed_dim, embed_dim)
        self.c = c


    def softmax_masque(self, x, l):
        # faire un masque qui ne prenne pas en compte les tokens de padding
        # x: (batch_size, seq_len, emb_size)
        # l: (batch_size)
        mask = torch.arange(x.size(1))[None, :] >= l[:, None] #> ou >= ???????????
        mask = mask.unsqueeze(2).expand(x.size())
        x[mask] = float('-inf')

        return self.softmax(x)

    def forward(self, x, l):
        # x: (batch_size, seq_len_false, emb_size)
        # l: (seq_len_true)

        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)

        # quand on appelle le soft-mmax il faut utiliser un mask afin d'appliquer
        # le softmax jusqu'a la fin de la phrase mais pas au dela (on ne prend pas
        # en compte les tokens de padding) 
        attention = self.softmax_masque(self.c + torch.div(q @ torch.transpose(k, dim0=1, dim1=2), torch.sqrt(self._embed_dim)))
        x = self.lin_final( attention @ v )

        return x


class Model_attention(nn.Module):
    def __init__(self, embed_dim, c=0):
        super(Model_attention, self).__init__()
        self.Att1 = Self_attention(embed_dim, c)
        self.Att2 = Self_attention(embed_dim, c)
        self.Att3 = Self_attention(embed_dim, c)
        self.lin = nn.Linear(embed_dim, 2)
        self.sigmoid = nn.Sigmoid()
        self.TransformerEncoder = nn.Sequential(self.Att1, self.Att2, self.Att3, self.lin, self.sigmoid)
        

    def forward(self, x, l):
        x = self.TransformerEncoder(x, l)
        return x






@click.command()
@click.option('--test-iterations', default=1000, type=int, help='Number of training iterations (batches) before testing')
@click.option('--epochs', default=50, help='Number of epochs.')
@click.option('--modeltype', required=True, type=int, help="0: base, 1 : Attention1, 2: Attention2")
@click.option('--emb-size', default=100, help='embeddings size')
@click.option('--batch-size', default=20, help='batch size')
def main(epochs,test_iterations,modeltype,emb_size,batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ##  TODO: 
    # hyperparametres
    embed_dim  = 100
    c = 0
    model = Model_attention(embed_dim, c).to(device)
    criterion = nn.BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        for x,y,l in tqdm(train_loader):
            optimizer.zero_grad()
            y_pred = model(x,l)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            writer.add_scalar("Loss/train", loss_epoch, epoch)
            writer.add_scalar("Accuracy/train", torch.sum(torch.argmax(y_pred, dim=1) == y).item() / len(y), epoch)


        model.eval()
        with torch.no_grad():
            for x,y,l in tqdm(test_loader):
                y_pred = model(x,l)
                loss = criterion(y_pred, y)
                loss_epoch += loss.item()
                writer.add_scalar("Loss/train", loss_epoch, epoch)
                writer.add_scalar("Accuracy/test", torch.sum(torch.argmax(y_pred, dim=1) == y).item() / len(y), epoch)
        
    



if __name__ == "__main__":
    main()
