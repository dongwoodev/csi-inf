"""
YOU NEED THESE FILES
- images in /image
- Labeled csi_data in /data

and these images datas will saved /video

"""


# Standard
import os
import glob
import argparse


# third-party
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont # pillow


# torch
from widar_model import *


# Create Video Directory
videoFolderPath = "./video"
if not os.path.exists(videoFolderPath):
    os.makedirs(videoFolderPath)


class ImageLabeling:
    def __init__(self, data, path):
        self.data = glob.glob(f"{data}" + "/*.csv") # csv
        self.path = glob.glob(f"{path}/*.jpg") # images

    @staticmethod
    def find_image_in_timestamp(start_time, end_time, image_files):
        """
        ### find_image timestamps that fits csi timestamps
        - args
            - start_time(string)
            - end_time(string)
            - image_files(list)
        - return
            - fitting images ts file name list
        """
        return [image_file for image_file in image_files if pd.to_datetime(start_time) <= pd.to_datetime(image_file.split('/')[3][:-7].replace('_',' ')) <= pd.to_datetime(end_time)] # 2024-06-21 10:25:48.560000
    
    @staticmethod
    def draw_in_picture(image_list, action, values, count):
        """
        ### labeling on picture
        - args
            - image_list(list) : images in csi ts
            - action(str) : action 
            - values(int) : action softmax
            - count(int) : inference counting

        - return
            - saved img to `/video`
        """
        for image in image_list:
            saved_file_name = image.split('/')[3].split('__')[0] # 2024-06-21_10:23:56.04
            pil_image = Image.open(image)
            text = f'{count} : {action}({(values * 100)}%)'
            font = ImageFont.load_default(40)

            # drawing
            draw = ImageDraw.Draw(pil_image)

            # Text background
            text_bbox = draw.textbbox((0,0), text, font=font) 
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3]+ 10 - text_bbox[1]
            draw.rounded_rectangle([0, 0, text_width, text_height], fill=(0,0,0))

            # Text
            draw.text((0, 0), text, (255,255,255), font) # position , text, color
            pil_image.save("./video/" + f"{saved_file_name}.jpg")
            # pil_image.show()

    def save_image(self):
       for data in self.data: #csv in list
           df = pd.read_csv(data) # load csv file.
           inf_cnt = 0
           while True:
               if inf_cnt == len(df):
                   break
               start_time = df.loc[inf_cnt, 'start_time']
               end_time = df.loc[inf_cnt, 'end_time']
               label = df.loc[inf_cnt, 'label']
               label_value = df.loc[inf_cnt, label].round(2)
            #    print(label)
            #    print(start_time)
            #    print(end_time)
            #    print(label_value)
               image_list = self.find_image_in_timestamp(start_time, end_time, self.path)
               self.draw_in_picture(image_list, label, label_value, inf_cnt)
               inf_cnt += 1
               print(f"{round((inf_cnt / len(df)) * 100,1)}" + " % 진행!" )
               
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', dest='data', action='store', default= './label') # CSV data path
    parser.add_argument('-p', '--path', dest='image', action='store', default='./image') # Image path

    args = parser.parse_args()
    data = args.data
    path = args.image


    data = ImageLabeling(data, path) # init

    data.save_image() # forward