from ultralytics import YOLO
from utils import norm_kpts
import pandas as pd
import cv2
import os
import glob
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pose", type=str,
                choices=[
                    'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 
                    'yolov8l-pose', 'yolov8x-pose', 'yolov8x-pose-p6'
                ],
                default='yolov8n-pose',
                help="choose type of yolov8 pose model")
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to data/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save csv file, eg: dir/data.csv")
args = vars(ap.parse_args())

col_names = [
    '0_X', '0_Y','0_conf', '1_X', '1_Y','1_conf', '2_X', '2_Y','2_conf', '3_X', '3_Y','3_conf', '4_X', '4_Y','4_conf', '5_X', '5_Y','5_conf', 
    '6_X', '6_Y','6_conf', '7_X', '7_Y','7_conf', '8_X', '8_Y','8_conf', '9_X', '9_Y','9_conf', '10_X', '10_Y','10_conf', '11_X', '11_Y', '11_conf',
    '12_X', '12_Y','12_conf', '13_X', '13_Y','13_conf', '14_X', '14_Y','14_conf', '15_X', '15_Y','15_conf', '16_X', '16_Y','16_conf',
]

# YOLOv8 Pose Model
model = YOLO(f"{args['pose']}.pt")

full_lm_list = []
target_list = []
class_names = sorted(os.listdir(args['data']))
for class_name in class_names:
    path_to_class = os.path.join(args['data'], class_name)
    img_list = glob.glob(path_to_class + '/*.jpg') + \
        glob.glob(path_to_class + '/*.jpeg') + \
        glob.glob(path_to_class + '/*.png')
    img_list = sorted(img_list)

    for img_path in img_list:
        img = cv2.imread(img_path)
        lm_list = []
        if img is None:
            print(
                f'[ERROR] Error in reading {img_path} -- Skipping.....\n[INFO] Taking next Image')
            continue
        else:
            results = model.predict(img)
            for result in results:
                poses = result.keypoints
                for pose in poses:
                    for pnt in pose:
                        confi = pnt.conf[0]
                        for ppnt, z in zip(pnt.xy[0], confi.tolist()):
                            x = ppnt[0]
                            y = ppnt[1]
                            lm_list.append([int(x), int(y), z])
        
        if len(lm_list) == 17:
            pre_lm = norm_kpts(lm_list)
            full_lm_list.append(pre_lm)
            target_list.append(class_name)

        print(f'{os.path.split(img_path)[1]} Landmarks added Successfully')
    print(f'[INFO] {class_name} Successfully Completed')
print('[INFO] Landmarks from Dataset Successfully Completed')

# to csv
data_x = pd.DataFrame(full_lm_list, columns=col_names)
data = data_x.assign(Pose_Class=target_list)
data.to_csv(args['save'], encoding='utf-8', index=False)
print(f"[INFO] Successfully Saved Landmarks data into {args['save']}")
