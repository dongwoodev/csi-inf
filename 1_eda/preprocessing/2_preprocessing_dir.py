import argparse
import os
from glob import glob
import shutil

"""
# 날짜를 받아 dataset_3090 디렉토리에 클래스 분류
"""



# 각 분류별 대상 디렉토리 설정
act_map = {
    'sit': ['ap-r-sit', 'ap-l-sit', 'esp-r-sit', 'esp-l-sit'],
    'std': ['ap-r-std', 'ap-l-std','esp-r-std', 'esp-l-std'],
    'walk': ['ap-l-walk', 'ap-r-walk','esp-l-walk', 'esp-r-walk']
}
loc_map = {
    'esp': ['esp-l-sit', 'esp-l-std','esp-l-walk', 'esp-r-sit', 'esp-r-std','esp-r-walk'],
    'ap': ['ap-r-sit', 'ap-r-std','ap-r-walk','ap-l-sit', 'ap-l-std','ap-l-walk']
}
occ_map = {
    'occ': ['esp-l-sit', 'esp-l-std','esp-l-walk', 'esp-r-sit', 'esp-r-std','esp-r-walk','ap-r-sit', 'ap-r-std','ap-r-walk','ap-l-sit', 'ap-l-std','ap-l-walk'],
    'emp': ['empty']
}

def copy_files(src_dir, dest_dir):
    """src_dir의 모든 파일을 dest_dir로 복사합니다."""
    os.makedirs(dest_dir, exist_ok=True)
    
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)

def copy_by_mapping(mapping, category, source_root, target_root):
    """지정된 맵핑에 따라 파일을 복사합니다."""
    for key, dirs in mapping.items():
        for dir_name in dirs:
            src_dir = os.path.join(f"/csi/datasets/{source_root}", dir_name)
            dest_dir = os.path.join(target_root, category, key)

            print(f"Copying files from {src_dir} to {dest_dir}...")
            copy_files(src_dir, dest_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=str, help="복사할 파일이 있는 소스 디렉토리 (예: '1111')")

    args = parser.parse_args()
    # 소스 디렉토리 이름을 추출하여 타겟 디렉토리 설정
    #source_dir_name = os.path.basename(args.source_root.rstrip("/"))
    target_root = os.path.join("datasets_3090", args.source_root)

    copy_by_mapping(act_map, 'act', args.source_root, target_root)
    copy_by_mapping(loc_map, 'loc', args.source_root, target_root)
    copy_by_mapping(occ_map, 'occ', args.source_root, target_root)
    print("파일 복사가 완료되었습니다!")
