import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D

import argparse
from scipy.signal import butter, filtfilt

output = os.popen("ls -l").read()
print(output)

file_name = input()
file_name = "/Users/username/visual/" + file_name # File path!!!
inf_class = file_name.split("_")[0].split("/")[-1]
csv_file = f"{file_name}"
df = pd.read_csv(csv_file, header=None)
raw_data = np.array(df)
raw_data = raw_data[:300] # 180개의 데이터만 사용
print("0. csi: ", raw_data.shape)




def amplitude_calculation(data):
    even_elements = data[:, ::2]
    odd_elements = data[:, 1::2]
    print(even_elements)
    print(odd_elements)

    # # Phase
    # csi_complex = even_elements + 1j * odd_elements
    # phase = np.angle(csi_complex)
    # return phase

    # Amplitude
    amplitude = np.sqrt(np.square(even_elements) + np.square(odd_elements))
    return amplitude

def butterworth_filter(data, cutoff, fs, order, filter_type='low'):
    nyquist = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False, output="ba")
    filtered_data = filtfilt(b, a, data, axis=0) # 필터 적용
    filtered_data = np.ascontiguousarray(filtered_data) # 음수 스트라이드를 방지하기 위해 복사
    return filtered_data


remove_indices = np.concatenate((np.arange(0,6), np.arange(32,33), np.arange(59, 66), np.arange(123,134), np.arange(191,192)))

# remove_indices = np.arange(0, 6)  # np.arange는 1차원 배열을 반환합니다.


amplitude = amplitude_calculation(raw_data)
print("1. amplitude: ", amplitude.shape)
# amplitude = np.delete(amplitude, remove_indices, axis=1)
# print("2. remove_null: ", amplitude.shape)
butterworth = butterworth_filter(amplitude, cutoff=0.3, fs=10, order=1, filter_type='low')
print("3. 2butterworth: ", butterworth.shape)


# Visualization

df = pd.DataFrame(butterworth)
df_sampled = df.iloc[:, :]

X = np.arange(df_sampled.shape[1]) 
Y = np.arange(df_sampled.shape[0]) 
X, Y = np.meshgrid(X, Y)

Z = df_sampled.to_numpy()


# 3D Surface Plot 생성
fig = go.Figure(data=[go.Surface(z=Z, x=Y, y=X,cmin=0, cmax=30)])

z_min, z_max = 0, 30.0

# 그래프 레이아웃 설정
fig.update_layout(
    title=f"Amplitude Plot of CSI Data({inf_class})",
    scene=dict(
        zaxis=dict(range=[z_min, z_max]),
        xaxis_title="Sequence",
        yaxis_title="Subcarriers",
        zaxis_title="Amplitude",
        aspectmode="manual",
        aspectratio=dict(x=4, y=2, z=0.5)
    ),
)

# 그래프 표시
fig.show()


# butterworth = butterworth[:, 66 ] # [Sequence, Subcarriers]
# # Plotly 시각화
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=butterworth, mode='lines', name='67th Column'))

# # 그래프 레이아웃 설정
# fig.update_layout(
#     title="67th Column Visualization",
#     xaxis_title="Sequence",
#     yaxis_title="Amplitude",
#     yaxis=dict(range=[0, 30]),
#     template="plotly_white"
# )

# fig.show()