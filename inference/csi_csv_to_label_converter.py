## YOU NEED MODEL_R.pt and widar_model.py + locationfile.csv
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
        self.checked = glob.glob(f"{path}" + "/*_checked.csv")

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
        
    @staticmethod
    def n_of_human(data):
        """
        ### n of human
        """
        data = data.reset_index()
        #data.columns = ['timestamp', 'nums', 'N1', 'N2', 'location', 'skeleton']
        dataR = data[data['timestamp'].str.contains('_R')]
        # dataL = data[data['timestamp'].str.contains('_L')]

        dataR.loc[:, 'timestamp'] = dataR.loc[:, 'timestamp'].str[-29:-7]
        dataR.loc[:, 'timestamp'] = dataR.loc[:, 'timestamp'].str.replace('_',' ')
        # dataL.loc[:, 'timestamp'] = dataL.loc[:, 'timestamp'].str[-29:-7]

        # dataR.loc[:, 'timestamp'] = pd.to_datetime(dataR.loc[:, 'timestamp'],  format='%Y-%m-%d_%H:%M:%S.%f')
        # dataL.loc[:, 'timestamp'] = pd.to_datetime(dataL.loc[:, 'timestamp'], format='%Y-%m-%d_%H:%M:%S.%f')

        # loc_data = pd.merge(dataL, dataR, on='timestamp', how='inner')
        loc_data = dataR

        # loc_data= loc_data[['timestamp', 'nums_y', 'location_y', 'skeleton_y']]
        loc_data= loc_data[['timestamp', 'nums', 'location', 'skeleton']] 

        # for timestamp, nums, locs, skels in zip(loc_data[0], loc_data['1_y'], loc_data['4_y'], loc_data['5_y']):
        loc_data= loc_data.sort_values(by=['timestamp'])

        loc_data.to_csv('./label/location.csv', index=False, encoding='utf-8')
        
        return loc_data
        
    def load_data(self, nums):
        """
        ### load_data
        - args
          - nums : num of csi_sequence (int)
        """
        for data, checked in zip(self.datas, self.checked): # O(num of CSVs)
            df = pd.read_csv(data) # load csv file.


            ck_column_names = ['timestamp', 'nums', 'N1', 'N2', 'location', 'skeleton']
            checked = pd.read_csv(checked, header=None, names=ck_column_names) # load checked file (N of humans, Location)
            checked = self.n_of_human(data=checked) # N of human, location preprocessing

            # for save csv (labeling) #
            csvFileName = data[-24:-5] # file name
            csvFile = open('./label/' + csvFileName + '_label.csv', 'w', newline='', encoding='utf-8')
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(['start_time','end_time', 'none', 'sit', 'stand', 'label']) # set columns

            limit = 0
            for csi_data, timestamp in zip(df['data'], df['Timestamp']):
                # Ensure CSI Timestamp and Human Location Timestamp are coverted to datetime.
                timestamp = pd.to_datetime(timestamp)


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


def merge_data():
    """
    ### merge_data
    - csi_data + location data (N of Human, location)
    - return : csv data including the Number of Human data
    """
    csi_data = glob.glob(f"./label" + "/*_label.csv")[0]
    loc_data = glob.glob(f"./label" + "/location.csv")[0]

    csi_data = pd.read_csv(csi_data, encoding='utf-8')
    loc_data = pd.read_csv(loc_data, encoding='utf-8')


    csi_data['start_time'] = pd.to_datetime(csi_data['start_time'])
    csi_data['end_time'] = pd.to_datetime(csi_data['end_time'])
    loc_data['timestamp'] = pd.to_datetime(loc_data['timestamp'])

    # # for save N of human csv (labeling) #
    csvFileName = str(csi_data['start_time'][0])[:-7] 
    csvFile = open('./label/' + csvFileName + '_label.csv', 'w', newline='', encoding='utf-8') # 2024-06-21 10:23:55_label
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['start_time','end_time','human' ,'none', 'sit', 'stand', 'label']) # set columns   

    os.remove("./label/location.csv") # delete location.csv 

    inf_cnt = 0
    while True:
        if inf_cnt == len(csi_data):
            break
        start_time = csi_data.loc[inf_cnt, 'start_time']
        end_time = csi_data.loc[inf_cnt, 'end_time']
        none = csi_data.loc[inf_cnt, 'none']
        sit = csi_data.loc[inf_cnt, 'sit']
        stand = csi_data.loc[inf_cnt, 'stand']
        label = csi_data.loc[inf_cnt, 'label']

        a = [humans for loc_time, humans in zip(loc_data['timestamp'], loc_data['nums']) if pd.to_datetime(start_time) <= pd.to_datetime(loc_time) <= pd.to_datetime(end_time)]
        human_value = int(round(sum(a) / len(a),0)) # calculate N of Human (Data range 2 Sec)

        inf_cnt += 1
        
        csvWriter.writerow([start_time, end_time, human_value, none, sit, stand, label])


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='path', action='store', default='./data')
    parser.add_argument('-s', '--size', dest='size', action='store', default=120)
    args = parser.parse_args()
    path = args.path # path of csv data files

    data = CSVLabelConverter(path) # init

    data.load_data(args.size) # forward,

    merge_data()

     