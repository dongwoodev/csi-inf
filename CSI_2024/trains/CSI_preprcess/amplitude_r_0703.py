# Import Library

import ast
import datetime
import glob
import os

import pandas as pd
import numpy as np


"""

진행 전, 2가지 이상의 파일이 필요합니다.
1. RAW한 CSI 데이터(1개)
2. inference.py를 거친 라벨링 데이터들 (1개 이상)

"""


def amplitude(path: str):
    df = pd.read_csv(path)

    ## EDA
    # df = df[df['bandwidth'] == 1] # ✔️ get length of features == 384
    

    df['data'] = df['data'].apply(ast.literal_eval) # Convert data to list by columns
    df_data = pd.DataFrame(df['data'].tolist(), index=df.index)
    timeseries = df['Timestamp'] # TimeSeries Columns
    df_data.iloc[:,130:] = df_data.iloc[:,130:].fillna(0) # 254 length of features data -> fill 0 padding 
    ## Calculate Amplitude Value
    new_df = {} # Dict for New Dataframe
    n = 0 # N of Columns

    for i in range(0, len(df_data.columns), 2):
        n += 1
        new_df['timestamp'] = df['Timestamp']
        new_df[n] = np.sqrt(np.square(df_data.iloc[:, i]) + np.square(df_data.iloc[:, i+1])) # Amplitude^2 = a^2 + b^2   
        
    
    
    new_df = pd.DataFrame(new_df)
    
    ## Save Dataframe in 2s intervals
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    interval = 2 # intervals(2 seconds)
    # split_dfs = []

    start_time = new_df['timestamp'].min().replace(microsecond=0)
    end_time = new_df['timestamp'].max().replace(microsecond=0)

    ## Save CSV File in "StartTime" Directory
    if not os.path.exists(f'{start_time}'):
        os.mkdir(f'{start_time}') # Create Directory
    global dirs
    dirs = start_time
    current_time = start_time
    if int(current_time.time().strftime("%S")) % 2 != 0:
        current_time = current_time + datetime.timedelta(seconds=1)
    while current_time <= end_time:
        next_time = current_time + pd.Timedelta(seconds=interval)
        split_df = new_df[(new_df['timestamp'] >= current_time) & (new_df['timestamp'] < next_time)]
        split_df.to_csv(f"./{start_time}/{current_time}.csv", sep=',', header=False, index=False) # Save start_time_{n}.csv
        # split_dfs.append(split_df)
        current_time = next_time

def preprocessing(path: str):

    df = pd.read_csv(path)
    interval = 2 # 2 sec

    ## TimeStamp EDA
    df['timestamp'] = df['timestamp'].str.replace('a/sit/','')
    df['timestamp'] = df['timestamp'].str.replace('a/stand/','')
    df['timestamp'] = df['timestamp'].str.replace('_',' ')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    start_time = df['timestamp'].min().replace(microsecond=0) + pd.Timedelta(seconds=interval)
    end_time = df['timestamp'].max().replace(microsecond=0) + pd.Timedelta(seconds=interval)  

    current_time = start_time

    if not os.path.exists(f'processed_R'):
        os.mkdir(f'processed_R') # Create Directory
        os.mkdir(f'processed_R/sit')
        os.mkdir(f'processed_R/stand')

    if int(current_time.time().strftime("%S")) % 2 != 0:
        current_time = current_time + datetime.timedelta(seconds=1)
    while current_time <= end_time:
        next_time = current_time + pd.Timedelta(seconds=interval)
        split_df = df[(df['timestamp'] >= current_time) & (df['timestamp'] < next_time)]
        if split_df['labels'].nunique() == 1:
            label = str(split_df['labels'].mode()[0])
            print(label, current_time)
            if label == "stand":        
                csi_df = pd.read_csv(f"./{dirs}/{current_time}.csv")
                csi_df.insert(193, '', label)
                csi_df.to_csv(f"./processed_R/stand/{current_time}.csv", sep=',', header=False, index=False)
            else:
                csi_df = pd.read_csv(f"./{dirs}/{current_time}.csv")
                csi_df.insert(193, '', label)
                csi_df.to_csv(f"./processed_R/sit/{current_time}.csv", sep=',', header=False, index=False)                    
        current_time = next_time

# ----

paths = glob.glob('./*_.csv')

for path in paths:
    amplitude(path)    

print("50% - Amplitide 추출 값 완료")

paths = glob.glob('./*_R.csv')

for path in paths:
    preprocessing(path)  

print("100% - 전처리 완료") 
