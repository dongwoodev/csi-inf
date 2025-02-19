import ast
import glob
import argparse
import os
import argparse

import pandas as pd

class PeopleLabeling:
    def __init__(self):
        self.people_df = pd.read_csv('0809_ppl_L_test.csv', encoding='utf-8')
        self.csifiles = sorted(glob.glob("./csi/*.csv"))
        os.makedirs('./complete', exist_ok=True) # RESULT DIRS
        os.makedirs('./complete/1', exist_ok=True)
        os.makedirs('./complete/0', exist_ok=True)
        os.makedirs('./complete/other', exist_ok=True)

    def saved_csi_label(self, grid, stride):
        # Define Grid Time(interval) and Stride(Sliding window)
        grid = pd.Timedelta(seconds=grid)
        stride = pd.Timedelta(milliseconds=stride)

        # Timestamps preprocessing
        self.people_df['timestamp'] = self.people_df['path'].str.replace('__R.jpg', '')
        self.people_df['timestamp'] = self.people_df['timestamp'].str.replace('__L.jpg', '')
        self.people_df['timestamp'] = self.people_df['timestamp'].str.replace('_', ' ')
        self.people_df['timestamp'] = pd.to_datetime(self.people_df['timestamp'])

        try:
            for csi_file in self.csifiles:
                # Load the CSI Data
                csi_df = pd.read_csv(csi_file)
                csi_df['Timestamp'] = pd.to_datetime(csi_df['Timestamp'])
                
                start_time = self.people_df['timestamp'].min() # Label file Min & Max Time
                end_time = self.people_df['timestamp'].max()

                csi_end_time = csi_df['Timestamp'].max()
                current_start = start_time # Init time

                print(current_start, end_time)
                # Sliding Window
                while current_start < end_time:
                    # terminate programe
                    current_end = current_start + grid
                    if current_start > csi_end_time:
                        break

                    print(f"CSI : {current_start} - {current_end}, LABEL : {start_time}, {end_time}")

                    # Extract CSI data
                    grid_df = csi_df[(csi_df['Timestamp'] >= current_start) & (csi_df['Timestamp'] < current_end)]

                    if grid_df.empty:
                        current_start += stride
                        continue
                
                    # CSI DATA COLUNM, LIST TO COLUNMS
                    grid_df.loc[:, 'data'] = grid_df['data'].apply(ast.literal_eval)  # Convert data to list by columns
                    data_to_cols = pd.DataFrame(grid_df['data'].tolist(), index=grid_df.index)
                    grid_df = grid_df.drop(columns=['data']).join(data_to_cols) 
                    grid_df = pd.concat([grid_df, data_to_cols], axis=1) # Remove unnecessary columns

                    grid_df.iloc[:,130:] = grid_df.iloc[:,130:].fillna(0)
                    filtered_csi_data = grid_df.iloc[:, [0] + list(range(25, 409))] # Timestamp + data

                    # Extract Location data
                    people_data = self.people_df[(self.people_df['timestamp'] >= current_start) & (self.people_df['timestamp'] < current_end)]

                    if not people_data.empty:
                        people_data = people_data['people'].iloc[0]
                    else:
                        # 해당 시간 내 위치 데이터가 없는 경우
                        current_start += stride
                        continue

                    filtered_csi_data = filtered_csi_data.assign(people=people_data)

                    people = int(filtered_csi_data[:1]['people'].values[0])
                    print(people == 0)
                    if people == 1:
                        save_filename = f'./complete/1/{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv'
                    elif people == 0:
                        save_filename = f'./complete/0/{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv'
                    else:
                        save_filename = f'./complete/other/{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv'                    

                    # Save the file
                    filtered_csi_data.to_csv(save_filename, index=False, header=None)
                    
                    print(f'{current_start.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]}.csv is successfully saved.')  
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
    grid = int(args.grid) # path of location csv data files
    stride = int(float(args.stride) * 1000)

    data = PeopleLabeling()
    data.saved_csi_label(grid, stride)

