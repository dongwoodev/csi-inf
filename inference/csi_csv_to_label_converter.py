## YOU NEED MODEL_R.pt and widar_model.py + locationfile.csv
# Standard
import os
import glob
import argparse
import csv
import json
import ast

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
        self.datas = glob.glob(f"{path}" + "/*_.csv")
        self.checked = glob.glob(f"{path}" + "/*_r.csv")

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
        for data, checked in zip(self.datas, self.checked): # O(num of CSVs)
            df = pd.read_csv(data) # load csv file.

            ck_column_names = ['timestamp', 'nums', 'N1', 'N2', 'location', 'skeleton']
            checked = pd.read_csv(checked, header=None, names=ck_column_names) # load checked file (N of humans, Location)
            checked = checked.reset_index()
            data_L = locationLR_preproceesing(checked, '_L')
            data_R = locationLR_preproceesing(checked, '_R')
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


def locationLR_preproceesing(raw_data, side):
    data = raw_data[raw_data['timestamp'].str.contains(side)] # side : '_L' or '_R'
    data.loc[:, 'timestamp'] = data.loc[:, 'timestamp'].str[-29:-7]
    data.loc[:, 'timestamp'] = data.loc[:, 'timestamp'].str.replace('_',' ')

    data = data[['timestamp', 'nums', 'location', 'skeleton']]

    # data.loc[:, 'skeleton'] = data.loc[:, 'skeleton']
    data['skeleton'] = data['skeleton'].astype(str)

    def get_key_point(tensor_str):
        """
        ### get_key_point
        - Get Keypoint of pelvis(Hip) from skeleton data
        """
        if tensor_str == 'nan':
            # There is No Bounding Box
            return [0,0]
        else:
            tensor_list = ast.literal_eval(tensor_str.replace('tensor', ''))
            if (tensor_list[9][0] == 0) and (tensor_list[10][0] == 0):
                # There is No Pelvis(Hip) Key Point
                return [0,0]
            elif tensor_list[9][0] == 0:
                # There is No Right Pelvis(Hip) Key Point
                return [int(tensor_list[10][0]), int(tensor_list[10][1])]
            elif tensor_list[10][0] == 0:
                # There is No Left Pelvis(Hip) Key Point
                return [int(tensor_list[9][0]), int(tensor_list[9][1])]
            else:
                # There is Both of Pelvis Key Point
                return [int(tensor_list[9][0]), int(tensor_list[9][1])]
                
    
    def location_ratio(skel_data, side):
        """
        ### Location Labeling
        
        """
        def line_slope_intercept(x1, y1, x2, y2):
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return slope, intercept
        
        if side == '_L':
            # base line f(x), g(x)
            x1, y1, x2, y2 = 230, 300, 570, 255 # f
            x3, y3, x4, y4 = 255, 600, 885, 450 # g

        else:
            # base line f(x), g(x)
            x1, y1, x2, y2 = 695, 175, 1035, 250 # f
            x3, y3, x4, y4 = 350, 400, 975, 595 # g
            
        slope_f , intercept_f = line_slope_intercept(x1, y1, x2, y2)
        slope_g , intercept_g = line_slope_intercept(x3, y3, x4, y4)
        
        def yf(x):
            return (slope_f*x) + intercept_f

        def yg(x):
            return (slope_g*x) + intercept_g
        

        def point_position(x, y):
            y_f = yf(x)
            y_g = yg(x)

            position = {
                "line1": "Mid" if y > y_f  else "AP",
                "line2": "ESP" if y > y_g else "Mid"
            }

            if position['line1'] == 'Mid' and position['line2'] == 'ESP':
                return 'ESP'
            if position['line1'] == 'AP' and position['line2'] == 'Mid':
                return 'AP'
            if position['line1'] == 'Mid' and position['line2'] == 'Mid':
                return 'Mid'        

        
        if skel_data[0] + skel_data[1] == 0:
            return 'None'
        else:
            return point_position(skel_data[0], skel_data[1])



    data['skeleton'] = data['skeleton'].apply(get_key_point)
    if side == '_L':
        data['location'] = data['skeleton'].apply(location_ratio, args=(side,))
    else:
        data['location'] = data['skeleton'].apply(location_ratio, args=(side,))

    data = data.sort_values(by=['timestamp'])
    data.to_csv(f'./label/location{side}.csv', index=False, encoding='utf-8')
    return data



def merge_data():
    """
    ### merge_data
    - csi_data + location data (N of Human, location)
    - return : csv data including the Number of Human data and LocationL, LocationR
    """
    csi_data = glob.glob(f"./label" + "/*_label.csv")[0]
    loc_data_L = glob.glob(f"./label" + "/location_L.csv")[0]
    loc_data_R = glob.glob(f"./label" + "/location_R.csv")[0]

    csi_data = pd.read_csv(csi_data, encoding='utf-8')
    loc_data_L = pd.read_csv(loc_data_L, encoding='utf-8')
    loc_data_R = pd.read_csv(loc_data_R, encoding='utf-8')

    # csi_data['start_time'] = pd.to_datetime(csi_data['start_time'])
    # csi_data['end_time'] = pd.to_datetime(csi_data['end_time'])
    loc_data_L['timestamp'] = pd.to_datetime(loc_data_L['timestamp'])
    loc_data_R['timestamp'] = pd.to_datetime(loc_data_R['timestamp'])


    # # for save N of human csv (labeling) #
    csvFileName = str(csi_data['start_time'][0])[:-7] 
    csvFile = open('./label/' + csvFileName + '_label.csv', 'w', newline='', encoding='utf-8') # 2024-06-21 10:23:55_label
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['start_time','end_time','human' , 'locL', 'locR', 'none', 'sit', 'stand', 'label']) # set columns   

    os.remove("./label/location_L.csv") # delete location.csv 
    os.remove("./label/location_R.csv") # delete location.csv 


    inf_cnt = 0
    while inf_cnt < len(csi_data):
        # CSI Data Columns
        start_time = csi_data.loc[inf_cnt, 'start_time']
        end_time = csi_data.loc[inf_cnt, 'end_time']
        none = csi_data.loc[inf_cnt, 'none']
        sit = csi_data.loc[inf_cnt, 'sit']
        stand = csi_data.loc[inf_cnt, 'stand']
        label = csi_data.loc[inf_cnt, 'label']

        # Filter YoloCSV based on CSI data time range.
        filtered_data = [
            (humans, locL, locR) for loc_time, humans, locL, locR 
            in zip(loc_data_L['timestamp'], loc_data_L['nums'], loc_data_L['location'], loc_data_R['location']) 
            if pd.to_datetime(start_time) <= pd.to_datetime(loc_time) <= pd.to_datetime(end_time)
        ]

        if filtered_data:
            # N of Humans
            humans_list = [data[0] for data in filtered_data]
            human_value = int(round(sum(humans_list) / len(humans_list), 0))

            # Location
            locL_list = [data[1] for data in filtered_data]
            locR_list = [data[2] for data in filtered_data]

            locationL = next((loc for loc in locL_list if loc), None)
            locationR = next((loc for loc in locR_list if loc), None)

        else:
            human_value = 0
            locationL = None
            locationR = None

        csvWriter.writerow([start_time, end_time, human_value, locationL, locationR, none, sit, stand, label])

        print(f"{round(inf_cnt / (len(csi_data) * 0.01))}%") 
        inf_cnt += 1
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='path', action='store', default='./data')
    parser.add_argument('-s', '--size', dest='size', action='store', default=120)
    args = parser.parse_args()
    path = args.path # path of csv data files

    data = CSVLabelConverter(path) # init

    data.load_data(args.size) # forward,

    merge_data()

     