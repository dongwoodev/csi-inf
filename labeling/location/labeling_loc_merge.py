"""
code update 22 Aug 15:28
"""

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

# SETTING LOCATION COLUMN BASED ON CONDITIONS
def determine_location(row):
    if row['location_L'] == row['location_R']:
        return row['location_L']
    elif pd.notna(row['location_L']) and pd.isna(row['location_R']):
        return row['location_L']
    elif pd.notna(row['location_R']) and pd.isna(row['location_L']):
        return row['location_R']
    else:
        return None

merged_df['location'] = merged_df.apply(determine_location, axis=1)
merged_df = merged_df.sort_values(by='path')

# EMPTY OR DIFF DATA
lack_df = merged_df[(merged_df['location'].isna())]
clean_df = merged_df[(merged_df['location'].notna())]
clean_df = clean_df[["path", "location"]]

# SAVING DATA
fileName = fileList[0].split('/')[-1].split('_complete.csv')[0][:-2]
lack_df.to_csv(fileName + '_lack.csv', index=False)
merged_df.to_csv(fileName + '_merged.csv', index=False)
clean_df.to_csv(fileName + '_cleaned.csv', index=False)