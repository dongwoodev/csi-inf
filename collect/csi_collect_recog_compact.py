"""
CSI-Data Collecting Program

- author
    - https://github.com/espressif/esp-idf
    - https://github.com/espressif/esp-csi
"""

# Standard Library
import sys, os
import csv
import json
import argparse
import serial
import datetime
import multiprocessing

# Third-party
import numpy as np
import cv2

# GUI library
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QRadioButton, QPushButton, QLabel, QButtonGroup
from PyQt5.Qt import *

import pyqtgraph as pq
 

# YOLOv8
"""
installation : `pip install ultralytics`
ref : https://docs.ultralytics.com/tasks/pose/

"""
from ultralytics import YOLO

# GLOBAL VARIABLES #

csi_total_subcarrier_index = [i for i in range(0, 384)]  # 28  White
csi_valid_subcarrier_index = [i for i in range(0, 192)]  # 28  White

CSI_DATA_INDEX = 1000  # buffer size
CSI_DATA_LLFT_COLUMNS = 64 # size of columns
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]
CSI_DATA_COLUMNS = len(csi_valid_subcarrier_index)

# 초기 데이터(zero)
csi_ht_data_array = np.zeros( [CSI_DATA_INDEX, 128], dtype=np.complex64)
csi_lt_data_array = np.zeros( [CSI_DATA_INDEX, 64], dtype=np.complex64)

# Butterworth Filter
cutoff = 10
order = 8

# 사인 파형과 노이즈 생성
fs = 200  # 샘플링 주파수
t = np.arange(0, 2, 1/fs)  # 시간 벡터
freq = 10  # 사인 파형의 주파수
x = np.sin(2 * np.pi * freq * t)  # 사인 파형
x_noise = x + 0.5 * np.random.normal(size=len(t))  # 노이즈 추가

class csi_data_graphical_window(QMainWindow):
    def __init__(self):
        super().__init__() # Inheritance QMainWindow
        """
        GUI Window와 관련한 변수를 생성합니다.

        - 위젯 UI와 관련된 변수
        - 초기 버튼 값 상태 (isButtonStopped, startTime, stopTime)
        - 초기 제로 데이터 설정
        - 타이머 설정
        - CSV 파일 설정
        """

        self.setWindowTitle("Real-time CSI-data Heatmap")
        self.setGeometry(500, 500, 1200, 800) # location(x, y), width, height

        # SETTING MAIN WIDGET & LAYOUT
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.layout = QVBoxLayout() # 세로 레이아웃
        self.mainWidget.setLayout(self.layout)

        # SETTING PYQTGRAPH
        self.graphWidget = pq.GraphicsLayoutWidget()

        self.plotItem_ht = self.graphWidget.addPlot(row=0, col=0, title="CSI Data(HT)") 
        self.plotItem_ht.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_lt = self.graphWidget.addPlot(row=1, col=0, title="CSI Data(LT)")
        self.plotItem_lt.setLabels(left='CSI Carrier Number', bottom='Time')

        # SETTING HEAT MAP
        self.heatmap_ht = pq.ImageItem(border='w')
        self.heatmap_lt = pq.ImageItem(border='w')
        self.plotItem_ht.addItem(self.heatmap_ht)
        self.plotItem_lt.addItem(self.heatmap_lt)
        
        # 컬러 스케일 (LUT) 설정
        colormap_ht = pq.colormap.getFromMatplotlib('inferno')  # 'viridis', 'plasma', 'inferno', 'magma' 'coolwarm'등 사용 가능
        colormap_lt = pq.colormap.getFromMatplotlib('inferno')  # 'viridis', 'plasma', 'inferno', 'magma' 'coolwarm'등 사용 가능
        self.heatmap_ht.setLookupTable(colormap_ht.getLookupTable())
        self.heatmap_lt.setLookupTable(colormap_lt.getLookupTable())
        
        # 컬러 스케일의 최소값과 최대값 정의
        self.absScaleMin = 0
        self.absScaleMax = 80
        self.phaseScaleMin = -3
        self.phaseScaleMax = 3 

        # SETTING START BUTTON
        self.pushButton = QPushButton("START")
        self.pushButton.setStyleSheet("background-color: blue; color: white;")  # 초기 색상 설정
        self.pushButton.setMaximumHeight(80)
        self.pushButton.clicked.connect(self.toggleButtonState)

        # SETTING TEXT OUTPUT SPACE
        self.textLabel = QLabel("Selected option: None")
        self.textLabel.setWordWrap(True)  # 긴 텍스트를 위한 자동 줄바꿈 활성화
        self.textLabel2 = QLabel("TIME INFO")  # 오른쪽에 추가할 새로운 텍스트 출력창 
        self.textLabel3 = QLabel("STATE")

        # ADD UI IN LAYOUT
        self.wLayout = QVBoxLayout() 
        self.wLayout.addWidget(self.radioButton1) # 라디오 버튼 추가
        self.wLayout.addWidget(self.radioButton2)
        self.wLayout.addWidget(self.radioButton3)

        self.hLayout = QHBoxLayout()
        self.hLayout.addLayout(self.wLayout)
        self.hLayout.addWidget(self.pushButton)

        self.hLayout2 = QHBoxLayout()
        self.hLayout2.addWidget(self.textLabel) 
        self.hLayout2.addWidget(self.textLabel2)

        self.layout.addWidget(self.graphWidget) # 그래프
        self.layout.addLayout(self.hLayout) # 라디오버튼, 시작 버튼
        self.layout.addLayout(self.hLayout2) # 기록 라벨

        # Set up CSI initial Data #
        """
        - 초기 데이터의 절댓값, 각도를 반환합니다.
        - 타이머를 설정합니다.
        - 데이터셋 디렉토리를 생성합니다.
        - 생성한 디렉토리에 넣을 CSV파일을 설정합니다. 
        """

        # 초기 데이터 설정
        self.csi_ht_abs_array = np.abs(csi_ht_data_array)
        self.csi_ht_phase_array = np.angle(csi_ht_data_array)
        self.csi_lt_abs_array = np.abs(csi_lt_data_array)
        self.csi_lt_phase_array = np.angle(csi_lt_data_array)
        self.heatmap_ht.setImage(self.csi_ht_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        self.heatmap_lt.setImage(self.csi_lt_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        
        # 타이머 설정
        self.timer = QTimer()
        self.timer.setInterval(0.01)  # 0.1초마다 업데이트
        self.timer.timeout.connect(self.update_data)
        self.timer.start(0)	# 0 -> 100

        # Datasets 폴더 생성 확인
        self.datasetFolderPath = "Dataset"
        if not os.path.exists(self.datasetFolderPath):
            os.makedirs(self.datasetFolderPath)       
            


        self.isButtonStopped = False  # 버튼 상태 추적을 위한 변수

        self.startTime = None   # 타이머 변수
        self.stopTime = None

        # label -> key(For Record)
        self.labelDict = {'':0, 'Steady State':1, 'Stand':2, 'Sit':3}


    def update_data(self):
        """
        - 0.1초마다 데이터의 변환을 히트맵으로 나타냅니다.
        - `Datasets` 디렉토리에 저장할 CSI 데이터를 저장합니다.
            - `isStarted`가 True의 경우, 데이터를 수집합니다.
            - `isStarted`이 False인 경우 데이터를 수집을 멈춥니다.


        Returns:
            - Datasets/csi_data_with_labels_2024-05-16 10:49:04.csv
        """
        self.csi_ht_abs_array = np.abs(csi_ht_data_array)
        self.csi_ht_phase_array = np.angle(csi_ht_data_array)
        self.csi_lt_abs_array = np.abs(csi_lt_data_array)
        self.csi_lt_phase_array = np.angle(csi_lt_data_array)

        # 변경되는 데이터 시각화
        self.heatmap_ht.setImage(self.csi_ht_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        self.heatmap_lt.setImage(self.csi_lt_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        
        # 0분 3분마다 스위치 컨트롤
        current_minutes = datetime.datetime.now().minute
        current_hour = datetime.datetime.now().hour
        
        if (current_minutes in [30,31] and current_hour in [9, 13, 17, 20])  and isStarted.value == False:
            print(f'⏰ [{datetime.datetime.now()}] Activate collecting CSI Data')
            self.pushButton.click()
        elif (current_minutes in [32] and current_hour in [9, 13, 17, 20]) and isStarted.value == True:
            print(f'⏰ [{datetime.datetime.now()}] Deactivate collecting CSI Data')
            self.pushButton.click()


    def toggleButtonState(self):
        """
        버튼의 상태에 따라 적절한 액션을 수행하는 메서드
        - `if self.isButtonStopped` : 버튼 상태가 STOP인 상태
        - `else`: START버튼을 눌러 STOP을 대기하는 상태
        """
        if self.isButtonStopped:
            # STOP ~ (START를 대기하는 상태)
            isStarted.value = False
            self.stopTime = QDateTime.currentDateTime()
            self.pushButton.setText("Start")
            self.pushButton.setStyleSheet("background-color: blue; color: white;")  # "Start" 상태의 색상
            # 여기에 "Start" 상태일 때 수행할 추가 동작을 구현할 수 있습니다.
            self.textLabel2.setText(f"Started at: {self.startTime.toString()} || Stopped at: {self.stopTime.toString()}")
            self.startTime = None  # 다음 시작을 위해 초기화

        else:
            # START ~ (STOP을 대기하는 상태)
            isStarted.value = True
            self.startTime = QDateTime.currentDateTime()
            #self.textLabel2.setText(f"Started at: {self.startTime.toString()}")
            self.pushButton.setText("Stop")
            self.pushButton.setStyleSheet("background-color: red; color: black;")  # "Stop" 상태의 색상
            # 여기에 "Stop" 상태일 때 수행할 추가 동작을 구현할 수 있습니다.
            
        self.isButtonStopped = not self.isButtonStopped  # 상태 토글


def csi_data_read_parse(ser, isCollect):

    while True:
        strings = str(ser.readline())
        if not strings:
            break

        strings = strings.lstrip('b\'').rstrip('\\r\\n\'') # 포트를 통해 수집한 데이터 (가장 아래 주석 참조)
        index = strings.find('CSI_DATA') # 'CSI_DATA'가 있는 문자열 인덱스, 0이 나와야 정상

        if index == -1:
            continue  
        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)

        # exception #
        if len(csi_data) != len(DATA_COLUMNS_NAMES):
            print(f"해당 데이터의 컬럼의 수가 상이합니다. {len(DATA_COLUMNS_NAMES)} != {len(csi_data)}")

        try:
            csi_raw_data = json.loads(csi_data[-1]) # JSON 객체를 파이썬 객체로 읽어옵니다.
        except json.JSONDecodeError:
            print(f"JSON 객체를 파이썬 객체로 읽어오기에 데이터가 불완전(incomplete)합니다.")
            continue

        if len(csi_raw_data) != 128 and len(csi_raw_data) != 256 and len(csi_raw_data) != 384:
            print(f"파이썬 객체로 변환 후 데이터의 컬럼 수가 상이합니다.: {len(csi_raw_data)}")
            continue


        # 데이터를 수집하기 시작
        if isStarted.value == True and isCollect == False:
            print(f"    📝 [{datetime.datetime.now()}] CSI 데이터 작성을 시작합니다.")
            # CSV 파일 설정
            csvFileName = f"/data/csi-data/Dataset/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_.csv" # 파일 생성
            csvFile = open(csvFileName, 'w', newline='', encoding='utf-8') # csv파일 설정
            csvWriter = csv.writer(csvFile) # 파일 객체를 csv.writer 객체로 변환
            csvWriter.writerow(["Timestamp", "Label"] + DATA_COLUMNS_NAMES) # 데이터셋 컬럼        
            isCollect = True
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            csvWriter.writerow([timestamp] + csi_data) # 데이터 csv 파일에 작성하기

        # 데이터를 수집하고 있는 경우
        elif isStarted.value == True and isCollect == True: 
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            csvWriter.writerow([timestamp] + csi_data) # 데이터 csv 파일에 작성하기
        
        # 데이터 수집이 끝난 경우
        elif isStarted.value == False and isCollect == True:
            print(f"    📄 [{datetime.datetime.now()}] CSI 데이터 작성을 종료합니다.")
            isCollect = False
            csvFile.close()

        csi_ht_data_array[:-1] = csi_ht_data_array[1:]
        csi_lt_data_array[:-1] = csi_lt_data_array[1:]

        # 데이터가 어떻게 들어오는지 확인

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
    """
    ### Thread for Collecting Data with GUI
    """
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.ser = serial.Serial(port=self.serial_port, baudrate=921600, bytesize=8, parity='N', stopbits=1)
        if self.ser.isOpen():
            print("💡 ESP Open Success!")
        else:
            return
        # 데이터 수집 시작 플래그
        self.collectingData = False
    def run(self):
        csi_data_read_parse(self.ser, self.collectingData)



class Camera():
    """
    ### Img data have collect, but inferenced data haven't collect.
    - Active Acquire
    - Passive Acqire (09:30, 13:30, 17:30, 20:30)
    """
    def __init__(self):
        # Image Dirs Settings
        self.photosFolderPath = "/data/csi-data/Photos"
        os.makedirs(self.photosFolderPath, exist_ok=True)

        # Camera Settings
        self.camA= cv2.VideoCapture('/dev/video0') # CamA
        self.camB= cv2.VideoCapture('/dev/video2') # CamB
        # If the Cams doesn't open due to a error, put in VideoCapture '1' or '/dev/video1' instead of '0'.

        self.set_camera_resolution(self.camA)
        self.set_camera_resolution(self.camB)        

        self.record = False
        self.model = YOLO("./yolov8x-pose.pt") # YOLOv8 Model

    def set_camera_resolution(self, camera, width=1280, height=720):
        """### Camera FRAME WIDTH, HEIGHT(1280x720)"""
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    def recording(self, isStarted, isProcess):
        try:
            while True:
                # Frames continued collect
                retA, frameA = self.camA.read()
                retB, frameB = self.camB.read()

                # Inference Human's Existion.
                resultA = self.model.predict(frameA, iou=0.5, conf=0.5)[0]
                resultB = self.model.predict(frameB, iou=0.5, conf=0.5)[0]

                current_minute = datetime.datetime.now().minute
                current_hour = datetime.datetime.now().hour

                # Condition : The number of bounding boxes is more than 1, or at the passive time
                condition = ((len(resultA.boxes) >= 1 or len(resultB.boxes) >= 1) or (current_minute in [30, 31] and current_hour in [9, 13, 17, 20]))

                # Start Collecting Image data
                if condition and not isProcess.value:
                    isStarted.value = True # Start Collecting CSI data through img
                    
                    # Create Detailed Image Directory
                    timestamp_dirs = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                    os.makedirs(self.photosFolderPath + f"/{timestamp_dirs}", exist_ok=True)       

                # Collecting data..
                elif condition and isProcess.value:
                    timestamp_image = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-4]

                    # Save Raw image data
                    cv2.imwrite(os.path.join(self.photosFolderPath + f"/{timestamp_dirs}",f'{timestamp_image}__L.jpg'), frameA)
                    cv2.imwrite(os.path.join(self.photosFolderPath + f"/{timestamp_dirs}", f'{timestamp_image}__R.jpg'), frameB)              

                # Stop Collecting image data
                elif not condition and isProcess.value:
                    isStarted.value = False # Stop Collecting CSI data

                # if Activate 'ESC' Key, 
                if cv2.waitKey(1) & 0xFF == 27:
                    break
 
 
        except KeyboardInterrupt:
            pass

        finally:
            self.camA.release()
            self.camB.release()
            cv2.destroyAllWindows()        

if __name__ == '__main__':

    # MORE THAN PYTHON 3.6
    if sys.version_info < (3, 6):
        print("Python version should be > 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device") # ESP32s3 Module Port  for connecting 
    
    args = parser.parse_args()
    serial_port = args.port # /dev/ttyACM0

    app = QApplication(sys.argv) # 명령어를 인수로 앱 객체 생성

    # STARTS UP WINDOW
    window = csi_data_graphical_window()
    window.show() 

    # SHARING VARIABLES
    isStarted = multiprocessing.Value('b', False) # 스위치가 켜졌느냐 안켜졌느냐
    isProcess = multiprocessing.Value('b', False) # 사람이 있는지 없는지
    labelkey = multiprocessing.Value('i', 0)
    
    # SUB THREAD
    subthread = SubThread(serial_port)
    subthread.start()

    # CAMERA SETTINGS (multiprocessing)
    camera = Camera()
    rec = multiprocessing.Process(target=camera.recording, args=(isStarted, isProcess))
    rec.start()

    sys.exit(app.exec())

