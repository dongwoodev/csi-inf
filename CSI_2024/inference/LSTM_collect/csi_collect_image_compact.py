# Standard Library
import sys, os
from io import StringIO
import csv
import json
import argparse
import datetime
import multiprocessing

# Third-party
import numpy as np
import cv2
import serial

# GUI library
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.Qt import *
import pyqtgraph as pq # Data Visualizer 

# YOLOv8
"""
installation : `pip install ultralytics`
ref : https://docs.ultralytics.com/tasks/pose/
"""
from ultralytics import YOLO

# GLOBAL VARIABLES
csi_total_subcarrier_index = [i for i in range(0, 384)]
csi_valid_subcarrier_index = [i for i in range(0, 192)]

CSI_DATA_INDEX = 1000  # buffer size
CSI_DATA_LLFT_COLUMNS = 64 # size of columns
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]
CSI_DATA_COLUMNS = len(csi_valid_subcarrier_index)

## initial data(zero data)
csi_ht_data_array = np.zeros( [CSI_DATA_INDEX, 128], dtype=np.complex64)
csi_lt_data_array = np.zeros( [CSI_DATA_INDEX, 64], dtype=np.complex64)



class csi_data_graphical_window(QMainWindow):
    def __init__(self, rec):
        super().__init__()
        self.rec = rec

        self.setWindowTitle("Data Acquirement Tool")
        self.setGeometry(500, 500, 1200, 800) # location(x, y), width, height

        # SETTING MAIN WIDGET & LAYOUT
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.layout = QVBoxLayout()
        self.mainWidget.setLayout(self.layout)

        # SETTING PYQTGRAPH
        self.graphWidget = pq.GraphicsLayoutWidget()

        self.plotItem_ht = self.graphWidget.addPlot(row=0, col=0, title="CSI Data(HT)") 
        self.plotItem_ht.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_lt = self.graphWidget.addPlot(row=1, col=0, title="CSI Data(LT)")
        self.plotItem_lt.setLabels(left='CSI Carrier Number', bottom='Time')

        ## SETTING HEAT MAP
        self.heatmap_ht = pq.ImageItem(border='w')
        self.heatmap_lt = pq.ImageItem(border='w')
        self.plotItem_ht.addItem(self.heatmap_ht)
        self.plotItem_lt.addItem(self.heatmap_lt)
        
        ## Color Scale(LUT)
        colormap_ht = pq.colormap.getFromMatplotlib('inferno')  # 'viridis', 'plasma', 'inferno', 'magma' 'coolwarm'
        colormap_lt = pq.colormap.getFromMatplotlib('inferno')  # 'viridis', 'plasma', 'inferno', 'magma' 'coolwarm'
        self.heatmap_ht.setLookupTable(colormap_ht.getLookupTable())
        self.heatmap_lt.setLookupTable(colormap_lt.getLookupTable())
        
        ## Color Scale Min&Max
        self.absScaleMin = 0
        self.absScaleMax = 80
        self.phaseScaleMin = -3
        self.phaseScaleMax = 3 

        # SETTING START BUTTON
        self.pushButton = QPushButton("No data currently being collected.")
        self.pushButton.setStyleSheet("background-color: gray; color: black;")
        self.pushButton.setMaximumHeight(100)
        self.pushButton.setMinimumHeight(100)
        self.pushButton.clicked.connect(self.toggleButtonState)
        self.hLayout = QHBoxLayout()
        self.hLayout.addWidget(self.pushButton)

        self.layout.addWidget(self.graphWidget) # Graph
        self.layout.addLayout(self.hLayout) # Start Button

        # Set up CSI initial Data
        self.csi_ht_abs_array = np.abs(csi_ht_data_array)
        self.csi_ht_phase_array = np.angle(csi_ht_data_array)
        self.csi_lt_abs_array = np.abs(csi_lt_data_array)
        self.csi_lt_phase_array = np.angle(csi_lt_data_array)
        self.heatmap_ht.setImage(self.csi_ht_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        self.heatmap_lt.setImage(self.csi_lt_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        
        # timer
        self.timer = QTimer()
        self.timer.setInterval(0.01)  # update per 0.1s
        self.timer.timeout.connect(self.update_data)
        self.timer.start(0)	# 0 -> 100

        # CSI Datasets folder
        os.makedirs("/data/csi-data/Dataset", exist_ok=True) # PATH      

        self.isButtonStopped = False  # Variable for Button behaviour state.

    def closeEvent(self, event):
        # Terminate the camera process when the window is closed
        if self.rec.is_alive():
            self.rec.terminate()
            self.rec.join()  # Ensure the process has finished
        event.accept()  # Accept the close event to close the window


    def update_data(self):
        """
        # ON
        1. In case of human detection,  activate isStarted = True. # Camera class
        2. In update_data func, isProcess.value = True + PUSH BUTTON.

        # OFF
        1. In case of None detection, activate isStarted = False. #Camera class
        2. In update_data func, isProcess.value = Fasle + PUSH BUTTON.
        """

        # Visualize colleted CSI data
        self.csi_ht_abs_array = np.abs(csi_ht_data_array)
        self.csi_ht_phase_array = np.angle(csi_ht_data_array)
        self.csi_lt_abs_array = np.abs(csi_lt_data_array)
        self.csi_lt_phase_array = np.angle(csi_lt_data_array)

        self.heatmap_ht.setImage(self.csi_ht_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        self.heatmap_lt.setImage(self.csi_lt_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        
        if  isStarted.value == True and isProcess.value == False:
            print(f'▶️ [{datetime.datetime.now()}] Acquiring data have been activated')
            isProcess.value = True
            self.pushButton.click()

        elif isStarted.value == False and isProcess.value == True:
            print(f'⏹️ [{datetime.datetime.now()}] Acquiring data have been deactivated')
            isProcess.value = False
            self.pushButton.click()

    def toggleButtonState(self):
        # Just Button Design.
        if self.isButtonStopped:
            # State for not being acquired.
            self.pushButton.setText("No data currently being collected.")
            self.pushButton.setStyleSheet("background-color: gray; color: black;")   

        else:
            # State for being acquired.
            self.pushButton.setText("Currently Collecting data")
            self.pushButton.setStyleSheet("background-color: blue; color: white;")
            
        self.isButtonStopped = not self.isButtonStopped

def csi_data_read_parse(ser, isCollect):
    while True:
        strings = str(ser.readline()) # byte → strings 
        if not strings:
            break

        strings = strings.lstrip('b\'').rstrip('\\r\\n\'') # cleaning
        index = strings.find('CSI_DATA') # 'CSI_DATA' must be exist in the string

        if index == -1:
            # Not exist CSI_DATA in string
            continue  
        csv_reader = csv.reader(StringIO(strings)) # string to CSV file
        csi_data = next(csv_reader)

        # exception #
        if len(csi_data) != len(DATA_COLUMNS_NAMES):
            print(f"⏸️ The length of the collected csi-data does not match the expected columns, expected columns: {len(DATA_COLUMNS_NAMES)}, current columns: {len(csi_data)}")
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1]) 
        except json.JSONDecodeError:
            print(f"⏸️ Unable to convert JSON data to a Python object. The data is incomplete.")
            continue

        if len(csi_raw_data) not in [128, 256, 384]:
            print(f"⏸️ The number of columns in the converted data is not valid (128, 256, 384 expected). Current number of columns: {len(csi_raw_data)}")
            continue


        # Writing CSI-data in CSV file
        if isStarted.value == True and isCollect == False:
            csvFileName = f"/data/csi-data/Dataset/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_.csv"
            csvFile = open(csvFileName, 'w', newline='', encoding='utf-8')
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(["Timestamp"] + DATA_COLUMNS_NAMES)      
            isCollect = True
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            csvWriter.writerow([timestamp] + csi_data)

        # Writing CSI-data (Progressing)
        elif isStarted.value == True and isCollect == True: 
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            csvWriter.writerow([timestamp] + csi_data)
        
        # Done Writing CSI-data
        elif isStarted.value == False and isCollect == True:
            isCollect = False
            csvFile.close()

        csi_ht_data_array[:-1] = csi_ht_data_array[1:]
        csi_lt_data_array[:-1] = csi_lt_data_array[1:]

        if len(csi_raw_data) == 384:
            csi_valid_subcarrier_len = CSI_DATA_COLUMNS
            csi_ht_subcarrier_en = 1
        else:
            if len(csi_raw_data) == 128:
                csi_valid_subcarrier_len = CSI_DATA_LLFT_COLUMNS
                csi_ht_subcarrier_en = 0
            else:
                csi_valid_subcarrier_len = 128
                csi_ht_subcarrier_en = 2 


        if csi_ht_subcarrier_en == 2:
            continue
        else:
            for i in range(csi_valid_subcarrier_len):	# HT or LT
                if i < 64:                              # LT Received : -32:32
                    csi_lt_data_array[-1][i] = complex(csi_raw_data[csi_total_subcarrier_index[i]*2  ],
                                                csi_raw_data[csi_total_subcarrier_index[i*2+1]  ]) *2
                    #    csi_lt_data_array[-1][i] = complex(csi_raw_data[csi_total_subcarrier_index[i]  ],
                    #                            csi_raw_data[csi_total_subcarrier_index[i]+ 33*2  ]) *2
                if (i< 192) and (i >= 64):                              # LT Received : -32:32
                    csi_ht_data_array[-1][i-64] = complex(csi_raw_data[csi_total_subcarrier_index[i]*2  ],
                                                csi_raw_data[csi_total_subcarrier_index[i*2+1]  ])   
            continue

    ser.close()
    return

class SubThread(QThread):
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.ser = serial.Serial(port=self.serial_port, baudrate=921600, bytesize=8, parity='N', stopbits=1)

        if self.ser.isOpen():
            print("Serial Port OPEN SUCCESS")
        else:
            return

        # Collect data Flag
        self.isCollect = False
        

    def run(self):
        csi_data_read_parse(self.ser, self.isCollect)

class Camera():
    def __init__(self):
        self.photosFolderPath = "/data/csi-data/Photos" # PATH
        os.makedirs(self.photosFolderPath, exist_ok=True)

        # Camera Settings
        self.camA= cv2.VideoCapture('/dev/video0') # CamA # CAMPATH
        self.camB= cv2.VideoCapture('/dev/video1') # CamB
        # If the Cams doesn't open due to a error, put in VideoCapture '1' or '/dev/video1' instead of '0'.

        self.set_camera_resolution(camera=self.camA)      
        self.set_camera_resolution(camera=self.camB)      

        self.record = False
        self.model = YOLO("./yolov8s-pose.pt") # YOLOv8 Model PATH

    def set_camera_resolution(self, camera, width=1280, height=720):
        """### Camera FRAME WIDTH, HEIGHT(1280x720)"""
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    def recording(self, isStarted, isProcess):
        while True:
            # Frames continued collect
            retA, frameA = self.camA.read()
            retB, frameB = self.camB.read()

            resultA = self.model.predict(frameA, iou=0.5, conf=0.5)[0]
            resultB = self.model.predict(frameB, iou=0.5, conf=0.5)[0]

            current_minute = datetime.datetime.now().minute
            current_hour = datetime.datetime.now().hour

            # Condition : The number of bounding boxes is more than 1, or at the passive time
            condition = ((len(resultA.boxes) >= 1 or len(resultB.boxes) >= 1) or (current_minute in [30, 31] and current_hour in [9, 13, 17, 20]))

            # Start Collecting Image data
            if condition and (isProcess.value == False):
                isStarted.value = True # Start Collecting CSI data through img
                
                # Create Detailed Image Directory
                timestamp_dirs = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                os.makedirs(self.photosFolderPath + f"/{timestamp_dirs}", exist_ok=True) # PATH     

            # Collecting data..
            elif condition and (isProcess.value == True):
                timestamp_image = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-4]

                # Save Raw image data
                cv2.imwrite(os.path.join(self.photosFolderPath + f"/{timestamp_dirs}",f'{timestamp_image}__L.jpg'), frameA) # PATH
                cv2.imwrite(os.path.join(self.photosFolderPath + f"/{timestamp_dirs}", f'{timestamp_image}__R.jpg'), frameB) # PATH         

            # Stop Collecting image data
            elif (len(resultA.boxes) == 0 and len(resultB.boxes) == 0) or (current_minute not in [30, 31] and current_hour not in [9, 13, 17, 20]) and (isProcess.value == True):
                isStarted.value = False # Stop Collecting CSI data            

if __name__ == '__main__':

    # MORE THAN PYTHON 3.6
    if sys.version_info < (3, 6):
        print("Python version should be > 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")

    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device") # ESP32s3 Module Port for connecting 

    args = parser.parse_args()
    serial_port = args.port # /dev/ttyACM0

    app = QApplication(sys.argv) # Create App Objects with Command Arguments

    # SHARING VARIABLES
    isStarted = multiprocessing.Value('b', False) # Human detection(CAMERA PART)
    isProcess = multiprocessing.Value('b', False) # Collecting CSI Data

    # CAMERA SETTINGS (multiprocessing)
    camera = Camera()
    rec = multiprocessing.Process(target=camera.recording, args=(isStarted, isProcess))
    rec.start()

    # STARTS UP WINDOW
    window = csi_data_graphical_window(rec)
    window.show() 
    
    # SUB THREAD
    subthread = SubThread(serial_port)
    subthread.start()
    sys.exit(app.exec())

