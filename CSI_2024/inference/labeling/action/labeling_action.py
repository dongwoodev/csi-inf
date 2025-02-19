import os
import glob
import argparse
import csv


import pandas as pd



class ActionLabeling:
    def __init__(self, action):
        self.images_L = sorted(glob.glob(f"./{action}/*__L.jpg"))
        self.images_R = sorted(glob.glob(f"./{action}/*__R.jpg"))
        self.action = action
    
    def action_label(self):

        def make_df(images, action):
            side = images[0].split('__')[-1][0]
            # './sit/2024-08-07_15_58_18.32__L.jpg' => '2024-08-07_15:58:32.55'
            timestamps = [image.split('/')[-1].replace(f'__{side}.jpg', '').replace('_',' ', 1).replace('_',':').replace(' ', '_') for image in images]
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'action' : self.action
            })

            return df


        df_L = make_df(self.images_L, self.action)
        df_R = make_df(self.images_R, self.action)

        merged_df = pd.merge(df_L, df_R, on='timestamp', suffixes=('_R', '_L'), how='outer')
        merged_df['action'] = self.action
        merged_df = merged_df.iloc[:,[0,3]]

        fileName = self.images_L[0].split('/')[-1].split('__L.jpg')[0]
        merged_df.to_csv(fileName + f'_{self.action}.csv', index=False)

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', dest='action', action='store')
    args = parser.parse_args()
    action = args.action

    data = ActionLabeling(action)
    data.action_label()
