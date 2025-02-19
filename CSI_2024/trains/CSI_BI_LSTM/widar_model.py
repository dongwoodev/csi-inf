import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from arcface_loss import ArcFaceLoss

class Widar_BiLSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(169,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,64)
    def forward(self,x):
        x = x.view(-1,120,169)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs
        
class Widar_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LSTM, self).__init__()
        self.lstm = nn.LSTM(169, 64, num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,120,169)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

class Widar_GRU(nn.Module):
    def __init__(self,num_classes):
        super(Widar_GRU,self).__init__()
        self.gru = nn.GRU(169,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,120,169)
        x = x.permute(1,0,2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs
    
class Widar_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Widar_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(120*169,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,120*169)
        x = self.fc(x)
        return x
