import os
import glob
import argparse


import pandas as pd



class OccLabeling:
    def __init__(self, occ):
        self.images_L = sorted(glob.glob(f"./{occ}/*__L.jpg"))
        self.images_R = sorted(glob.glob(f"./{occ}/*__R.jpg"))
        self.occ = occ
    
    def occ_label(self):

        def make_df(images, occ):
            side = images[0].split('__')[-1][0]
            # './zero/2024-08-07_15_58_18.32__L.jpg' => '2024-08-07_15:58:32.55'
            timestamps = [image.split('/')[-1].replace(f'__{side}.jpg', '').replace('_',' ', 1).replace('_',':').replace(' ', '_') for image in images]
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'occ' : self.occ
            })

            return df


        df_L = make_df(self.images_L, self.occ)
        df_R = make_df(self.images_R, self.occ) # single cam

        merged_df = pd.merge(df_L, df_R, on='timestamp', suffixes=('_R', '_L'), how='outer')
        merged_df['occ'] = self.occ
        merged_df = merged_df.iloc[:,[0,3]]

        fileName = self.images_L[0].split('/')[-1].split('__L.jpg')[0] # single cam
        
        merged_df.to_csv(fileName + f'_{self.occ}.csv', index=False) # single cam, modify merged_df => df_L

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--occ', dest='occ', action='store')
    args = parser.parse_args()
    occ = args.occ

    data = OccLabeling(occ)
    data.occ_label()
