import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Callable
from torch.utils.data import Dataset, DataLoader



class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        output_dim,
        activation = nn.Tanh(),
        decode_activation = nn.Softmax(),
        first_step = False,
        *args,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.activation = activation
        self.decode_activation = decode_activation
        self.first_step = first_step
        self.f_x = nn.Linear(input_dim, latent_dim)
        self.f_h = nn.Linear(latent_dim, latent_dim)
        self.f_d = nn.Linear(latent_dim, output_dim)

    def forward(self, x, h):
        '''
        x : batch x seq_len x dim
        h : batch x latent
        return :  seq_len x batch x latent        
        '''
        
        H = torch.zeros((x.size(0), x.size(1), h.size(1)))
        # cas du premier pas de temps
        for i in range(x.size(0)):
            h = self.one_step(x[i,:,:], h)
            H[i,:,:] = h
        return H
    
    def one_step(self, x, h):
        '''
        x :  batch x dim
        h :  batch x latent
        return : batch x latent
        '''
        return self.activation(self.f_x(x) + self.f_h(h))
    

    def decode(self, h):
        """
        h : batch x latent
        return : batch x output_dim
        """
        return self.decode_activation(self.f_d(h))




class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

