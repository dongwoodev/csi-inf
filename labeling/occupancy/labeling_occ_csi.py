import ast
import argparse
import csv
import glob
import os

import pandas as pd
import numpy as np

class CsiLabeling:
    def __init__(self):
        self.occ_df = pd.read_csv('occ.csv', encoding='utf-8')
        self.csifiles = sorted(glob.glob("./csi/*.csv"))
        os.makedirs('./complete', exist_ok=True)
        os.makedirs('./complete/zero', exist_ok=True)
        os.makedirs('./complete/multi', exist_ok=True)

    def saved_csi_label(self, grid, stride):
        # Define Grid time(interval) and Stride(Sliding window)
        grid = pd.Timedelta(seconds=grid)
        stride = pd.Timedelta(milliseconds=stride)

        self.occ_df['timestamp'] = self.occ_df['timestamp'].str.replace('_', ' ')
        self.occ_df['timestamp'] = pd.to_datetime(self.occ_df['timestamp'])

        try:
            for csi_file in self.csifiles:

                # Load the CSI Data
                csi_df = pd.read_csv(csi_file)
                csi_df['Timestamp'] = pd.to_datetime(csi_df['Timestamp']) # csi data를 한번에 잇는게 좋을래나...

                start_time = self.occ_df['timestamp'].min()
                end_time = self.occ_df['timestamp'].max()
                csi_end_time = csi_df['Timestamp'].max()
                current_start = start_time

                # Sliding Window
                while current_start < end_time:
                    current_end = current_start + grid # Start + 1 sec = end
                    if current_start > csi_end_time:
                        break
                    print(current_start, current_end, start_time, end_time)

                    # Extract CSI data
                    grid_df = csi_df[(csi_df['Timestamp'] >= current_start) & (csi_df['Timestamp'] < current_end)]

                    if grid_df.empty:
                        current_start += stride
                        continue
                    
                    grid_df.loc[:, 'data'] = grid_df['data'].apply(ast.literal_eval)  # Convert data to list by columns
                    data_to_cols = pd.DataFrame(grid_df['data'].tolist(), index=grid_df.index)
                    grid_df = grid_df.drop(columns=['data']).join(data_to_cols) 
                    grid_df = pd.concat([grid_df, data_to_cols], axis=1) # Remove unnecessary columns

                    grid_df.iloc[:,130:] = grid_df.iloc[:,130:].fillna(0)
                    filtered_csi_data = grid_df.iloc[:, [0] + list(range(25, 409))] # Timestamp + data


                    # Extract Occupancy data
                    occ_df = self.occ_df[(self.occ_df['timestamp'] >= current_start) & (self.occ_df['timestamp'] < current_end)]
                    if not occ_df.empty:
                        occ_df = occ_df['occ'].iloc[0]
                    else:
                        # 해당 시간 내 액션 데이터가 없는 경우
                        current_start += stride
                        continue
                    filtered_csi_data = filtered_csi_data.assign(occ=occ_df)
                    # # pose
                    # occupancy
                    occ = str(filtered_csi_data[:1]['occ'].values[0])
                    if occ == 'zero':
                        save_filename = f'./complete/zero/{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv'
                    elif occ == 'multi':
                        save_filename = f'./complete/multi/{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv'            

                    # Save the file
                    filtered_csi_data.to_csv(save_filename, index=False, header=None)
                    
                    print(f'{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv')
                    # Move to next stride
                    current_start += stride
        except pd.errors.ParserError as e:
            print(f"Error parsing {csi_file}: {e} csi 데이터 파일 구조를 다시 한번 확인해주세요.")
            return None
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grid', dest='grid', action='store')
    parser.add_argument('-s', '--strid', dest='stride', action='store')
    args = parser.parse_args()
    grid = int(args.grid) # path of occupancy class csv data files
    stride = int(float(args.stride) * 1000)

    data = CsiLabeling()
    data.saved_csi_label(grid, stride)
