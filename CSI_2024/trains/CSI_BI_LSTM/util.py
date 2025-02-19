from dataset import *
from widar_model import *
import torch
from torch.utils.data import DataLoader


def load_data_n_model(model_name, root):
    num_classes = 3
    train_loader = DataLoader(dataset=Csi_Dataset(root + 'R/train'), batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=Csi_Dataset(root + 'R/test'),  batch_size=128, shuffle=False)
    if model_name == 'BiLSTM':
        print("사용하는 모델: BiLSTM")
        model = Widar_BiLSTM(num_classes)
        train_epoch = 200

    elif model_name == 'LSTM':
        print("사용하는 모델 : LSTM")
        model = Widar_LSTM(num_classes)
        train_epoch = 100 #20
    
    elif model_name == 'GRU':
        print("사용하는 모델: GRU")
        model = Widar_GRU(num_classes)
        train_epoch = 100 
    elif model_name == 'MLP':
        print("using model: MLP")
        model = Widar_MLP(num_classes)
        train_epoch = 30 #20
    return train_loader, test_loader, model, train_epoch
