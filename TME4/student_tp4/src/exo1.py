from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import torch.optim as optim
from tqdm import tqdm
from typing import Callable

class RNN_many_to_one(nn.Module):
    """
    RNN many to one
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh()

    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.activation = activation
        self.decode_activation = nn.Softmax()
        self.h0 = nn.Parameter(torch.zeros(1, latent_dim))


        f_x = nn.linear(input_dim, latent_dim)
        f_h = nn.linear(latent_dim, latent_dim)

        def forward(self, input):
            h = self.h0
            for x in input:
                h = self.activation(self.f_x(x) + self.f_h(h))
            return h

        def decode(self, h):
            return self.decode_activation(nn.linear(h, self.output_dim))
        bloc = 

        for layer in range(input_dim):
            f_x = nn.sequential(
        self.nonlinear = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.gate = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])






    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in range(self.num_layers):
            linear = current_input
            nonlinear = self.nonlinear[layer](current_input)
            nonlinear = self.activation(nonlinear)
            gate = self.gate[layer](current_input)
            gate = torch.sigmoid(gate)
            current_input = gate * nonlinear + (1 - gate) * linear
        return current_input


class HighwayLayer(nn.Module):
    def __init__(self, input_size):
        super(HighwayLayer, self).__init__()
        self.lin1 = nn.Linear(input_size, input_size)
        self.lin2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        #non-linear transform H
        H = F.relu(self.lin1(x))

        # non-linear transforms T
        T = torch.sigmoid(self.lin2(x))

        # non-linear transforms C
        C= torch.mul(x,(1 - T))

        # Calcul de l'output
        output = torch.add(torch.mul(H, T), C)
        return output                    


class HighwayAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(HighwayAutoencoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encodeur
        for _ in range(num_layers ):
            self.encoder.append(HighwayLayer(input_size))
        self.encoder.append(nn.Linear(input_size, hidden_size))

        self.decoder.append(nn.Linear(hidden_size, input_size))
        # Decodeur
        for _ in range(num_layers ):
            self.decoder.append(HighwayLayer(input_size))


    def forward(self, x):
        # Encodeur
        for layer in self.encoder:
            x = layer(x)
        
        # Decodeur
        for layer in self.decoder:
            x = layer(x)
        
        return x
