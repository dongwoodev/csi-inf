import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Widar_BiLSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(169,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,120,169)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

