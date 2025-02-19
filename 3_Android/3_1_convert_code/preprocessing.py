import pandas as pd
import numpy as np
import argparse
from scipy.signal import butter, filtfilt

csv_file = "./data/AP_SIT.csv"
file_name = csv_file.split('/')[-1].split(".")[0]
df = pd.read_csv(csv_file, header=None)
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', dest='mode', type=str, required=False, default="empty") # EMPTY, PREV
args = parser.parse_args()

# 80번째부터 행부터 140번째 행까지 선택 (임의로 선택)
# raw_data = df.iloc[80:140]
# raw_data = np.array(raw_data)
raw_data = np.array(df) # Version2 
print("0. csi: ", raw_data.shape)


# 1. Amplitude Calculation

def amplitude_calculation(data):
    even_elements = data[:, ::2]
    odd_elements = data[:, 1::2]
    amplitude = np.sqrt(np.square(even_elements) + np.square(odd_elements))
    return amplitude

amplitude = amplitude_calculation(raw_data)
print("1. amplitude: ", amplitude.shape)


# 2. ButterWorth Filter
def butterworth_filter(data, cutoff, fs, order=5, filter_type='low', prev_data=None):
    nyquist = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = filtfilt(b, a, data, axis=0) # 필터 적용
    filtered_data = np.ascontiguousarray(filtered_data) # 음수 스트라이드를 방지하기 위해 복사
    return filtered_data

butterworth = butterworth_filter(amplitude, cutoff=0.4, fs=5, order=1, filter_type='low') / 20.0
butterworth = butterworth[5:-5,:] # 3.Side Remove 앞뒤 5개 데이터 제거
print("2~3. butterworth_data: ", butterworth.shape)


# 4. EMPTY or PREV
def emptyorprev(data=None):
    # spare_data = np.array(data.iloc[20:80])
    spare_data = np.array(data) # version 2
    amplitude_spr = amplitude_calculation(spare_data) # (1) amplitude
    butterworth_spr = butterworth_filter(amplitude, cutoff=0.4, fs=5, order=1, filter_type='low') / 20.0 # (2). Butterworth
    butterworth_spr = butterworth[5:-5,:] # (3). Side Remove

    averaged_amplitude = np.mean(np.array(butterworth_spr), axis=0)
    preprocessed_data = np.subtract(butterworth, averaged_amplitude) * 3 
    return preprocessed_data
    

if args.mode == "empty":
    df_spr = pd.read_csv('./data/EMPTY.csv')
    preprocessed_data = emptyorprev(data=df_spr) 
else:
    df_spr = pd.read_csv(csv_file)    
    preprocessed_data = emptyorprev(data=df)

print("4. Subtract empty or prev data: ", preprocessed_data.shape)


# 5. Remove Null Data
csi_data = []
remove_indices = np.concatenate((np.arange(0,6), np.arange(32,33), np.arange(59, 66), np.arange(123,134), np.arange(191,192)))
csi_data = np.delete(preprocessed_data, remove_indices, axis=1)
print("5. Remove Null data: ", csi_data.shape)

# Save array to csv file
df = pd.DataFrame(csi_data)
df.to_csv(f"{file_name}_output.csv", index=False, header=False)