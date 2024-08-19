import pandas as pd
import glob

fileList = glob.glob(f"./*_complete.csv")

# LOADING COMPLETED IMAGE LOCATION DATA L R
if fileList[0].split('/')[-1].split('_complete.csv')[0][-1] == 'R':
    loc_R_df = pd.read_csv(fileList[0].split('/')[-1])
    loc_L_df = pd.read_csv(fileList[1].split('/')[-1])
else:
    loc_L_df = pd.read_csv(fileList[0].split('/')[-1])
    loc_R_df = pd.read_csv(fileList[1].split('/')[-1])

# MERGED DATA
loc_L_df['path'] = loc_L_df['path'].str.replace('__L.jpg', '')
loc_R_df['path'] = loc_R_df['path'].str.replace('__R.jpg', '')
merged_df = pd.merge(loc_R_df, loc_L_df, on='path', suffixes=('_R', '_L'), how='outer')
merged_df['location'] = merged_df.apply(lambda row: row['location_L'] if row['location_L'] == row['location_R'] else None, axis=1)
merged_df = merged_df.sort_values(by='path')

# EMPTY OR DIFF DATA
lack_df = merged_df[(merged_df['location_L'].isna()) | (merged_df['location_R'].isna()) | (merged_df['location_L'] != merged_df['location_R'])]

# SAVING DATA
fileName = fileList[0].split('/')[-1].split('_complete.csv')[0][:-2]
lack_df.to_csv(fileName + '_lack.csv', index=False)
merged_df.to_csv(fileName + '_merged.csv', index=False)