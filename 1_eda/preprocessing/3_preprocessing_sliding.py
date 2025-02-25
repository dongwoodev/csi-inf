import argparse
import pandas as pd
import os
from glob import glob

"""
# Sliding Window CSI Data
- 만약 S를 0으로 하고 60W로 하면 슬라이딩 윈도우를 적용하지 않는다.
1. dataset_3090 디렉토리 내부에서 input_dir(Argument) 내 데이터 파일을 불러온다.
2. 불러온 데이터 파일을 width(Argument) 크기만큼 stride(Argument) 간격으로 슬라이딩 윈도우를 적용한다.
"""


def sliding_window_csi(input_dir, output_dir,width, stride):
    # 입력 디렉토리 내부의 모든 하위 디렉토리와 파일 탐색
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                
                # 파일 읽기
                data = pd.read_csv(input_file)
                total = len(data)
                
                # 입력 파일명과 상대 경로 추출
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                
                # 출력 디렉토리 생성 (존재하지 않는 경우)
                os.makedirs(output_subdir, exist_ok=True)
                
                # 파일명과 확장자 분리
                file_name = os.path.splitext(file)[0]
                
                # 파일 카운트 초기화
                file_count = 1

                if stride == 0:
                    # 슬라이딩 윈도우를 적용하지 않는 경우
                    for start_row in range(0, total, width):
                        end_row = start_row + width
                        window_data = data.iloc[start_row:end_row]
                        if len(window_data) == width:
                            output_filename = f"{output_subdir}/{file_name}_{file_count}.csv"
                            window_data.to_csv(output_filename, index=False, header=None)
                        file_count += 1

                else:
                    # 슬라이딩 윈도우를 통한 데이터 저장
                    for start_row in range(0, total - width + 1, stride): 
                        end_row = start_row + width 
                        window_data = data.iloc[start_row:end_row] # 0 ~ 0+60
                        
                        # 출력 파일 경로 및 저장
                        output_filename = f"{output_subdir}/{file_name}_{file_count}_{width}.csv"
                        if len(window_data) == width:
                            window_data.to_csv(output_filename, index=False, header=None)
                        file_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("width", type=int)
    parser.add_argument("stride", type=int)

    args = parser.parse_args()
        
    input_dir = f"/csi/datasets_3090/{args.input_dir}"
    output_dir = f"/csi/datasets_3090/{args.input_dir}_{args.width}W_{args.stride}S"

    sliding_window_csi(input_dir, output_dir, args.width, args.stride)
