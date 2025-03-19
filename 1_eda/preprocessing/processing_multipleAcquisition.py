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

# 디렉토리 내 모든 CSV 파일 처리
def process_all_csv_files(directory, slide, window, prevtime):
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]  # .csv 파일 리스트 가져오기
    print(csv_files)
    output_dir = os.path.join(directory, "output")  # 결과 저장 폴더

    os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성

    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        process_csv_files(file_path, output_dir, slide, window, prevtime)

def process_csv_files(file_path, output_dir, slide, window, prevtime):
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # 확장자 제거한 파일명
    extract_file_path = os.path.join(output_dir, file_name)  # 개별 파일 저장 디렉토리
    os.makedirs(extract_file_path, exist_ok=True)  # 파일별 디렉토리 생성

    print(f"📂 Processing: {file_name}")

    # Load the data
    df = pd.read_csv(file_path, header=None)
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
        diff_data = np.subtract(window_data, prev_data_mean) * 3 # 현재 데이터와 이전 데이터와 차이를 계산

        # Remove Null Subcarriers
        remove_indices = np.concatenate((np.arange(0,6), np.arange(32,33), np.arange(59, 66), np.arange(123,134), np.arange(191,192)))
        csi_data = np.delete(diff_data, remove_indices, axis=1) 
        result.append(csi_data)

        # 파일 저장
        save_path = os.path.join(extract_file_path, f"{file_name}_{i+1}.csv")
        np.savetxt(save_path, csi_data, delimiter=",", fmt='%f')
        print(f"저장 완료: {save_path}, Shape: {csi_data.shape}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=int)
    parser.add_argument("width", type=int)
    parser.add_argument("prevtime", type=int)
    args = parser.parse_args()

    dir_name = input("데이터가 있는 절대 경로 (/Users/csi/): ")
    process_all_csv_files(dir_name, slide=args.slide, window=args.width, prevtime=args.prevtime)



