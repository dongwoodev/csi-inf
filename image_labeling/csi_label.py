import ast
import argparse
import csv
import glob
import os

import pandas as pd
import numpy as np

class CsiLabeling:
    def __init__(self):
        self.location_df = pd.read_csv('location.csv', encoding='utf-8')
        # # pose
        # static action
        self.csifiles = glob.glob("./csi/*.csv")
        os.makedirs('./complete', exist_ok=True)

    def saved_csi_label(self, grid, stride):
        # Define Grid time(interval) and Stride(Sliding window)
        grid = pd.Timedelta(seconds=grid)
        stride = pd.Timedelta(milliseconds=stride)

        self.location_df['path'] = self.location_df['path'].str.replace('_', ' ')
        self.location_df['timestamp'] = pd.to_datetime(self.location_df['path'])
        # # pose
        # static action

        for csi_file in self.csifiles:

            # Load the CSI Data
            csi_df = pd.read_csv(csi_file)
            csi_df['Timestamp'] = pd.to_datetime(csi_df['Timestamp']) # csi data를 한번에 잇는게 좋을래나...

            start_time = self.location_df['timestamp'].min()
            end_time = self.location_df['timestamp'].max()

            current_start = start_time

            # Sliding Window
            while current_start <= end_time:
                current_end = current_start + grid # Start + 1 sec = end

                # Extract CSI data
                grid_df = csi_df[(csi_df['Timestamp'] >= current_start) & (csi_df['Timestamp'] < current_end)]
                grid_df.loc[:, 'data'] = grid_df['data'].apply(ast.literal_eval)  # Convert data to list by columns
                data_to_cols = pd.DataFrame(grid_df['data'].tolist(), index=grid_df.index)
                grid_df = grid_df.drop(columns=['data']).join(data_to_cols) 
                grid_df = pd.concat([grid_df, data_to_cols], axis=1) # Remove unnecessary columns

                grid_df.iloc[:,130:] = grid_df.iloc[:,130:].fillna(0)
                filtered_csi_data = grid_df.iloc[:, [0] + list(range(25, 409))] # Timestamp + data

                # Extract Location data
                location_data = self.location_df[(self.location_df['timestamp'] >= current_start) & (self.location_df['timestamp'] < current_end)]['location'].iloc[0]
                filtered_csi_data = filtered_csi_data.assign(location=location_data)
                # # pose
                # static action

                # Save the file
                save_filename = f'./complete/{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv'
                filtered_csi_data.to_csv(save_filename, index=False)

                print(f'{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv')
                # Move to next stride
                current_start += stride
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grid', dest='grid', action='store')
    parser.add_argument('-s', '--strid', dest='stride', action='store')
    args = parser.parse_args()
    grid = int(args.grid) # path of location csv data files
    stride = int(float(args.stride) * 100)

    data = CsiLabeling()
    data.saved_csi_label(grid, stride)
