import os
import shutil
import argparse
import csv

"""
# 정리되지 않은 데이터들을 넘버링을 통해 디렉토리 별로 분류하는 코드

-d : 1115, 1116 등 디렉토리 여부
-q : QUEUE 여부

1. 수집한 데이터(datasets/1115)들을 넘버링을 부여하여 분류하는 코드
return datasets/target_dir/L-sit
2. 이때 Queue Argument 유무에 따라 한 스팟(AP-L-SIT)에 있는 데이터를 한 파일로 묶을 건지 선택
"""

def move_csv_files(start, end, target_dir):
    global load_dir
    # h 디렉토리 경로 (csv 파일들이 있는 경로)

    # 대상 디렉토리가 없으면 생성
    os.makedirs(f"/csi/datasets/{load_dir}/{target_dir}", exist_ok=True) # 이미 디렉토리가 생성되있어도 오류없이 그 디렉토리에 담을 수 있음.
    print(f"디렉토리 생성: /csi/datasets/{load_dir}/{target_dir}")
    
    if queue == "queue":
        # 대상 디렉토리가 없으면 생성
        os.makedirs(f"/csi/datasets/{load_dir}_queue/{target_dir}", exist_ok=True) # 이미 디렉토리가 생성되있어도 오류없이 그 디렉토리에 담을 수 있음.
        print(f"Queue 디렉토리 생성: /csi/datasets/{load_dir}_queue/{target_dir}")        

    times = 0
    all_data = []

    # 시작 번호부터 끝 번호까지 반복
    for i in range(start, end + 1):
        filename = f"{i}.csv"
        source_path = os.path.join(f"/csi/datasets/{load_dir}", filename)  # 원본 파일 경로
        target_path = os.path.join(f"/csi/datasets/{load_dir}/{target_dir}", filename)  # 이동할 대상 경로

        # if queue == "queue":
        #     target_path_queue = os.path.join(f"/csi/datasets/{load_dir}_queue/{target_dir}", filename) 

        # 파일이 존재하는지 확인 후 이동
        if os.path.exists(source_path):
            print()
            times += 1
            shutil.copy(source_path, f"{target_path}")
            print(f"파일 이동: {filename}")

            if queue=="queue":
                with open(source_path, newline='') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for row in csvreader:
                        all_data.append(row)

        else:
            print(f"경고: {filename} 파일이 존재하지 않습니다.")

    if queue=="queue":
        with open(f"/csi/datasets/{load_dir}_queue/{target_dir}/{start}_{end+1}.csv", mode='w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(all_data)

    print(f"총 {times}개의 데이터가 이동했습니다.")


# 사용자 입력

def main():
    while True:
        try:
            start_num = int(input("시작 번호를 입력하세요: "))
            end_num = int(input("끝 번호를 입력하세요: "))
            target_directory = input("수집한 위치를 입력하세요(ex. esp-L-sit, ap-walk, empty) ")

            move_csv_files(start_num, end_num, target_directory)

        except ValueError:
            print("잘못된 입력입니다. 숫자를 입력하세요.")
        
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', dest='dir', action='store', required=True)
    parser.add_argument('-q', '--queue', type=str, default="general")
    args = parser.parse_args()
    load_dir = args.dir
    queue = args.queue

    print(f"분류되는 데이터는 {load_dir} 위치에서 확인할 수 있습니다.")
    main()
