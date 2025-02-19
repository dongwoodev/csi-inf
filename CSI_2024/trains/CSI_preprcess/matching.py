import glob
import pandas as pd


csv_files = glob.glob('./stand/*.csv')
for file in csv_files:
    df = pd.read_csv(file)
    if len(df) >= 120:
        df = df.iloc[:121, :]
        df.to_csv(file, index=False, header=False)
    else:
        print(len(df), file)


csv_files = glob.glob('./sit/*.csv')
for file in csv_files:
    df = pd.read_csv(file)
    if len(df) >= 120:
        df = df.iloc[:121, :]
        df.to_csv(file, index=False, header=False)
    else:
        print(len(df), file)

