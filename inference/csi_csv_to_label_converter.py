## YOU NEED MODEL_R.pt and widar_model.py
# Standard
import os
import glob
import argparse
import csv
import json

# third-party
import pandas as pd
import numpy as np

# torch
import torch
from widar_model import *

model = torch.load("model_R.pt")
model = model.eval()
padd256 = [0 for _ in range(128)] # pilot  Subcarrier
padd128 = [0 for _ in range(256)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labelFolderPath = "./label"
if not os.path.exists(labelFolderPath):
    os.makedirs(labelFolderPath)

class CSVLabelConverter:
    def __init__(self, path):
        self.datas = glob.glob(f"{path}" + "/*.csv")

    @staticmethod
    def preprocess(csi_cols: str):
        """
        ## preprocess
        - args : csi_cols(csi data 1 row) string type
        - return csi_data applicated amplitude (169)
        """
        csi_cols = json.loads(csi_cols)
        csi_amp = []

        if len(csi_cols) < 384:
            """
            해당 부분은 traits이 384개보다 작은 경우입니다.
            - 새 ESP라면 수정해야하는 부분
            """ 
            if len(csi_cols) == 256:
                csi_cols = csi_cols + padd256 # 256 + 129 = 384
            elif len(csi_cols) == 128:
                csi_cols = csi_cols + padd128
        # Ampulitude
        for i in range(0, len(csi_cols), 2):
            csi_amp.append(np.sqrt(np.square(csi_cols[i]) + np.square(csi_cols[i+1]))) # Amplitude^2 = a^2 + b^2
        
        # 192 ->169(13x13)
        csi_amp = csi_amp[6:32] + csi_amp[33:59] + csi_amp[66:125] + csi_amp[134:192]
        return csi_amp

    @staticmethod
    def infer(csi_seq):
        """
        ## Inference method
        - args : csi_seq(ndarray)
        - return : class softmax result (list)
        """
        csi_seq = torch.Tensor(csi_seq)
        csi_seq = torch.as_tensor(csi_seq, device=device)
        output = model(csi_seq)
        result = torch.softmax(output,dim=1)
        result = result.tolist()
        return result
        
        
    def load_data(self, nums):
        """
        ### load_data
        - args
          - nums : num of csi_sequence (int)
        """
        for data in self.datas: # O(num of CSVs)
            df = pd.read_csv(data) # load csv file.

            # for save csv (labeling) #
            csvFileName = data[-24:-5] # file name
            csvFile = open('./label/' + csvFileName + '_label.csv', 'w', newline='', encoding='utf-8')
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(['start_time','end_time', 'none', 'sit', 'stand', 'label']) # set columns

            limit = 0
            for csi_data, timestamp in zip(df['data'], df['Timestamp']):
                if limit == nums:
                    csi_seq = csi_seq.reshape(nums, 13, 13)
                    result = self.infer(csi_seq) # inference
                    temp_dict = {result[0][0] : 'none', result[0][1] : 'sit', result[0][2] : 'stand'}
                    end_time = timestamp
                    csvWriter.writerow([start_time, end_time, result[0][0], result[0][1], result[0][2], temp_dict[max(result[0])]])
                    limit = 0 # reset data from 120 to 0
                elif limit == 0:
                    csi_data = self.preprocess(csi_data) # Preprocessing
                    csi_seq = np.array([csi_data])
                    start_time = timestamp
                    limit += 1
                else:
                    csi_data = self.preprocess(csi_data) # Preprocessing
                    csi_data = np.array([csi_data])
                    csi_seq = np.append(csi_seq, csi_data, axis=0)
                    limit += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='path', action='store', default='./data')
    parser.add_argument('-s', '--size', dest='size', action='store', default=120)
    args = parser.parse_args()
    path = args.path # path of csv data files

    data = CSVLabelConverter(path) # init

    data.load_data(args.size) # forward, 