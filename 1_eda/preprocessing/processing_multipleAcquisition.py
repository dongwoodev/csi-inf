import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def amplitude_calculation(data):
    data = np.array(data, dtype=float)
    even_elements = data[:, ::2]
    odd_elements = data[:, 1::2]
    print(even_elements, type(even_elements))
    print(odd_elements, type(odd_elements))

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

# /Users/dongwookang/visual
def process_csv_files(file_name, slide, window, prevtime):

    # Load the data
    result = []
    load_file_path = os.path.join("")  # 파일 경로
    extract_file_path = os.path.join(f"/{file_name}") # 저장될 디렉토리 경로
    os.makedirs(extract_file_path, exist_ok=True)

    df = pd.read_csv(load_file_path + file_name + ".csv", header=None)
    raw_data = np.array(df)

    # 0. remove unnecessary columns (meta data)
    raw_data = raw_data[:, 5:]
    raw_data = np.array(raw_data, dtype=float)
    # 1. Amplitude Calculation
    amplitude_data = amplitude_calculation(raw_data)

    num_windows = (amplitude_data.shape[0] - prevtime) // slide  # 생성될 파일 개수 계산
    result = []

    for i in range(num_windows):
        start_prev = i * slide
        end_prev = start_prev + prevtime
        start_win = end_prev
        end_win = start_win + window

        if end_win > len(amplitude_data):
            break

        print(i, amplitude_data.shape[0])
        print("- Previous Data Range : ",start_prev, end_prev)
        print("- Window Data Range : ",start_win, end_win)

        # Butterworth 필터 적용
        prev_data = butterworth_filter(amplitude_data[start_prev:end_prev, :],cutoff=0.3, fs=10, order=1, filter_type='low')
        prev_data_mean = np.mean(prev_data, axis=0)

        window_data = butterworth_filter(amplitude_data[start_win:end_win, :],cutoff=0.3, fs=10, order=1, filter_type='low')
        diff_data = window_data - prev_data_mean

        result.append(diff_data)

        # 파일 저장
        save_path = os.path.join(extract_file_path, f"{file_name}_{i+1}.csv")
        np.savetxt(save_path, diff_data, delimiter=",", fmt='%f')
        print(f"저장 완료: {save_path}, Shape: {diff_data.shape}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("width", type=int)
    parser.add_argument("prevtime", type=int)
    args = parser.parse_args()

    file_name = input("Enter the file name: ")
    process_csv_files(file_name, slide=args.slide, window=args.width, prevtime=args.prevtime)



