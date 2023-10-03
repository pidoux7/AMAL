import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import torch.nn as nn
from tqdm import tqdm

# taille 
taille_echanti = 50
dimension_entree = 13
taille_sortie = 1

# Génération de données
x = torch.randn(taille_echanti, dimension_entree)
y = torch.randn(taille_echanti,taille_sortie)

# reseau de neurone
writer = SummaryWriter()
epsilon = 0.01
lin = nn.Linear(13, 1)
mse = nn.MSELoss()
seq = nn.Sequential(lin)
optim = torch.optim.SGD(seq.parameters(), lr=epsilon)
epochs = 1000


for epoch in range(epochs):
    optim.zero_grad()
    outputs = seq(x)
    loss = mse(outputs, y)
    loss.backward()
    optim.step()

    writer.add_scalar("Loss/train", loss.mean(), epoch)

    # Sortie directe
    print(f"Itérations {epoch}: loss {loss.mean()}")
