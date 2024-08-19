import argparse
import csv

# ----
"""
- file directory
  ./test/L folder - raw image 
  labeling_loc_no.py
  *_.csv - load using command

- execution command 
  python3 labeling_loc_no.py -f 0809_loc_R.csv -s R


- result
  ./loc_visual folder - visualized image
  *_complete.csv
"""

import glob
import os
from PIL import Image, ImageDraw, ImageFont # pillow 10.4.0

img_width, img_height = 1280, 720

class LocationLabeling:
    def __init__(self, file):
        self.csvfile = open(file, mode='r', newline='', encoding='utf-8')
        
    def location_label(self):
        csvreader = csv.DictReader(self.csvfile)
        savedcsvFile = open('./' + file[:-4] + '_complete.csv', 'w', newline='', encoding='utf-8') # 2024-06-21 10:23:55_label
        csvWriter = csv.writer(savedcsvFile)
        csvWriter.writerow(['path','location']) # set columns   

        for row in csvreader:
            skel_data = {'11': (float(row['11_X']), float(row['11_Y'])), '12': (float(row['12_X']), float(row['12_Y']))}
            
            loc = self.location_ratio(skel_data, side)
            csvWriter.writerow([row['path'], loc])
            

    @staticmethod
    def location_ratio(skel_data, side):
            
        # def normalize_x(x):
        #     return (x + 1) / 2 * img_width
        
        # def normalize_y(y):
        #     return (y + 1) / 2 * img_height
        
        def line_slope_intercept(x1, y1, x2, y2):
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return slope, intercept

        def yf(x):
            return (slope_f*x) + intercept_f
        
        def yg(x):
            return (slope_g*x) + intercept_g

        def point_position(x, y):
            y_f = yf(x)
            y_g = yg(x)

            position = {
                "line1": "Mid" if y > y_f  else "AP",
                "line2": "ESP" if y > y_g else "Mid"
            }

            if position['line1'] == 'Mid' and position['line2'] == 'ESP':
                return 'ESP'
            if position['line1'] == 'AP' and position['line2'] == 'Mid':
                return 'AP'
            if position['line1'] == 'Mid' and position['line2'] == 'Mid':
                return 'Mid'    

        x11, y11 = skel_data.get('11')
        x12, y12 = skel_data.get('12')


        if side == 'L':
            # base line f(x), g(x)
            x1, y1, x2, y2 = 230, 300, 570, 255 # f
            x3, y3, x4, y4 = 255, 600, 885, 450 # g


        else:
            # base line f(x), g(x)
            x1, y1, x2, y2 = 695, 175, 1035, 250 # f
            x3, y3, x4, y4 = 350, 400, 975, 595 # g

    
        slope_f, intercept_f = line_slope_intercept(x1, y1, x2, y2)
        slope_g, intercept_g = line_slope_intercept(x3, y3, x4, y4)


        if (x11, y11, x12, y12) == (0,0):
            return 'Nan'
        elif (x11, y11) == (0,0):
            return point_position(x12, y12)
        elif (x12, y12) == (0,0):
            return point_position(x11, y11)
        else:
            mid_x, mid_y = (x11 + x12)/2, (y11 + y12)/2 
            return point_position(mid_x, mid_y)

class LocationVisualizer:

    def __init__(self, file):
        filename = glob.glob(f"*{side}_complete.csv")[0]
        self.csvfile = open(filename, mode='r', newline='', encoding='utf-8')
        self.images = glob.glob(f"./test/{side}/*")
        self.modified_paths = [path.replace(f"./test/{side}/", "") for path in self.images]


        self.visualFolderPath = "./loc_visual"
        os.makedirs(self.visualFolderPath, exist_ok=True)
        os.makedirs(self.visualFolderPath + "/L", exist_ok=True)
        os.makedirs(self.visualFolderPath + "/R", exist_ok=True)
        self.find_data_in_images()

        

    def find_data_in_images(self):
        csvreader = csv.DictReader(self.csvfile)
        for row in csvreader:
            path = row['path']
            if path in self.modified_paths:
                pil_image = Image.open(f"./test/{side}/{path}")
                draw = ImageDraw.Draw(pil_image)

                if side == 'L':
                    draw.line((230, 300, 570, 255), fill='red', width=5)
                    draw.line((255, 600, 885, 450), fill='red', width=5)
                else:
                    draw.line((695, 175, 1035, 250), fill='red', width=5)
                    draw.line((350, 400, 975, 595), fill='red', width=5) 

                text = f"{row['location']}"
                font = ImageFont.load_default(size=40)

                # Text background
                
                text_bbox = draw.textbbox((0,0), text=text, font=font) 
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3]+ 10 - text_bbox[1]
                draw.rounded_rectangle([0, 0, text_width, text_height], fill=(0,0,0))

                # Text
                draw.text((0, 0), text, (255,255,255), font) # position , text, color
                pil_image.save(f"{self.visualFolderPath}/{side}/" + f"{path[:-4]}.jpg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', action='store')
    parser.add_argument('-s', '--side', dest='side', action='store')
    args = parser.parse_args()
    file = args.file # path of location csv data files
    side = args.side

    data = LocationLabeling(file) # init
    data.location_label()

    data = LocationVisualizer(file) # Visualize
    data.find_data_in_images()
