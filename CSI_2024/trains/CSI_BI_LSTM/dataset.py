import pandas as pd
import numpy as np
import glob
import torch
from torch.utils.data import Dataset

class Csi_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = sorted(glob.glob(root_dir+'/*/'))
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        print(self.category)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_dir = self.data_list[idx]

        cols = [i for i in range(1, 193)] # 192
        y = self.category[sample_dir.split('/')[-2]] # ex) 3-stand
        x = np.genfromtxt(sample_dir, delimiter=',', usecols=cols)
        x = pd.DataFrame(x)
        a = x.iloc[:,6:32]
        b = x.iloc[:,33:59]
        c = x.iloc[:,66:125]    
        d = x.iloc[:,134:192]          
        x = pd.concat([a,b,c,d], axis=1)

        x = x.iloc[:120, :]
        x = x.to_numpy()
        #print(x, len(x))


        # x=np.pad(x, (0,4), 'constant', constant_values=0)
        # x = x[:-4,:]

        # print("학습할 데이터의 shape 입니다", x.shape) # 사이즈
        # 정규화(normalize)
        # x = (x - 0.0025)/0.0119

        

        x = x.reshape(120, 13, 13)
        # reshape: 121, 192 -> x = x.reshape(121, )
        
        # reshape: 141,192 -> 
        # x = x.reshape(22,16,12)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y
