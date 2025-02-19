# Standard Library
import sys, os
from os import path
from io import StringIO
import csv
import json
import argparse
import serial
import datetime
import multiprocessing
import glob

# Third-party
import pandas as pd
import numpy as np
import cv2
import subprocess as sp

# GUI library
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QRadioButton, QPushButton, QLabel, QButtonGroup
from PyQt5.Qt import *

import pyqtgraph as pq
from pyqtgraph import PlotWidget
 
# Filtering Library
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter

# YOLOv8
"""
installation : `pip install ultralytics`
ref : https://docs.ultralytics.com/tasks/pose/

"""
from ultralytics import YOLO
import torch
from widar_model import *





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

padd = [0 for _ in range(128)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model_R.pt")
model = model.eval()


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

        # SETTING RADIO BUTTONS
        self.buttonGroup = QButtonGroup() # 버튼 그룹 설정
        self.radioButton1 = QRadioButton("Steady State")
        self.buttonGroup.addButton(self.radioButton1)
        self.radioButton2 = QRadioButton("Stand")
        self.buttonGroup.addButton(self.radioButton2)
        self.radioButton3 = QRadioButton("Sit")
        self.buttonGroup.addButton(self.radioButton3)
        
        self.buttonGroup.buttonClicked.connect(self.onRadioButtonClicked)

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
            
        # CSV 파일 설정
        self.csvFileName = "./Dataset/csi_data_with_labels_" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".csv"
        # self.csvFile = open(self.csvFileName, 'w', newline='', encoding='utf-8')
        # self.csvWriter = csv.writer(self.csvFile)
        # CSV 파일 헤더 작성
        # self.csvWriter.writerow(["Timestamp", "Label", "Data"] )#+ [f"Data{i}" for i in range(CSI_DATA_COLUMNS)])
        


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

        # 필터 적용
        b, a = butter_lowpass(cutoff, fs, order)
        filtered_ht_abs_array = filtfilt(b, a, self.csi_ht_abs_array)
        filtered_lt_abs_array = filtfilt(b, a, self.csi_lt_abs_array)

        # 가우시안 스무딩 적용
        sigma = 1 # 가우시안 커널의 표준편차
        self.smoothed_ht_abs_array = gaussian_filter(filtered_ht_abs_array, sigma)
        self.smoothed_lt_abs_array = gaussian_filter(filtered_lt_abs_array, sigma)

        # 변경되는 데이터 시각화
        self.heatmap_ht.setImage(self.csi_ht_abs_array, levels=(self.absScaleMin, self.absScaleMax))
        self.heatmap_lt.setImage(self.csi_lt_abs_array, levels=(self.absScaleMin, self.absScaleMax))

        # 0분 3분마다 스위치 컨트롤
        currentTime = datetime.datetime.now().minute
        
        if currentTime == 30  and isStarted.value == False:
            print(f'⏰ [{datetime.datetime.now()}] 데이터 기록 스위치가 활성화 되었습니다.')
            self.pushButton.click()
        elif currentTime == 32 and isStarted.value == True:
            print(f'⏰ [{datetime.datetime.now()}] 데이터 기록 스위치가 비활성화 되었습니다.')
            self.pushButton.click()

        """
        if currentTime in [0, 5, 10, 15, 20, 25, 30]  and isStarted.value == False:
            print(f'⏰ [{datetime.datetime.now()}] 데이터 기록 스위치가 활성화 되었습니다.')
            self.pushButton.click()
        elif currentTime in [2, 7, 12, 17, 22, 27, 32] and isStarted.value == True:
            print(f'⏰ [{datetime.datetime.now()}] 데이터 기록 스위치가 비활성화 되었습니다.')
            self.pushButton.click()

        """

    def onRadioButtonClicked(self, button):
        """
        변경된 라디오 버튼에 대해 핸들링하는 메서드
            - 텍스트 출력 레이아웃에 현재 선택된 옵션을 표시

        Args:
            `button` : 라디오버튼 객체
        
        """
        self.selectedLabel = button.text()
        self.textLabel.setText(f'Selected option: {button.text()}')
        labelkey.value = self.labelDict[self.selectedLabel] 

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

def butter_lowpass(cutoff, fs, order=5):
    """
    Butterworkth 필터 계수 계산

    - `nyq` : 샘플링 주파수 * 0.5
    - `normal_cutoff` : 컷오프 주파수 / nyq

    Arg: 
        -`cutoff` : 컷오프 주파수를 의미합니다. (기본값 10)
        - `fs` : 샘플링 주파수를 의미합니다. (기본값 8)
        - `order` : 필터의 차수를 의미합니다. (기본값 200)

    Returns:
        b, a : 필터의 분자(b)와 분모(a) 반환
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def csi_preprocessing(raw_data: str):
    """
    ### INFERENCE - PREPROCESSING ###
    Ampulitude를 적용해 196차원의 데이터를 만들어 시퀀스에 추가할 준비를 합니다. 
    
    """

    raw_data = json.loads(raw_data) # str -> list
    # ampulitude

    n = 0

    if len(raw_data) < 384:
        raw_data = raw_data + padd # 256 + 128 = 384

    csi_inf_data = []


    for i in range(0, len(raw_data), 2):
        # 384 -> 192
        csi_inf_data.append(np.sqrt(np.square(raw_data[i]) + np.square(raw_data[i+1]))) # Amplitude^2 = a^2 + b^2
        n += 1
    
    csi_inf_data = csi_inf_data + [0,0,0,0] # 192 -> 196

    return np.array([csi_inf_data])

def csi_data_read_parse(ser, isCollect, labelDict):

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
            csvFileName = f"/data/csi-data/Dataset/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{labelDict[labelkey.value]}.csv" # 파일 생성
            csvFile = open(csvFileName, 'w', newline='', encoding='utf-8') # csv파일 설정
            csvWriter = csv.writer(csvFile) # 파일 객체를 csv.writer 객체로 변환
            csvWriter.writerow(["Timestamp", "Label"] + DATA_COLUMNS_NAMES) # 데이터셋 컬럼        
            isCollect = True
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            label = labelDict[labelkey.value]
            csvWriter.writerow([timestamp, label] + csi_data) # 데이터 csv 파일에 작성하기

            ### INFERENCE ###
            # 객체 생성 및 시퀀스에 데이터 추가 
            x_data = csi_preprocessing(csi_data[24])
            # print(x_data)


        # 데이터를 수집하고 있는 경우
        elif isStarted.value == True and isCollect == True: 
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            label = labelDict[labelkey.value]
            csvWriter.writerow([timestamp, label] + csi_data) # 데이터 csv 파일에 작성하기

            ### INFERENCE ###
            inf_data = csi_preprocessing(csi_data[24])
            # print(inf_data.shape[1])s
            if inf_data.shape[1] == 196:
                if len(x_data) == 120:
                    x = pd.DataFrame(x_data)
                    a = x.iloc[:,6:32]
                    b = x.iloc[:,33:59]
                    c = x.iloc[:,66:125]    
                    d = x.iloc[:,134:192]          
                    x = pd.concat([a,b,c,d], axis=1)
                    x_data = x.to_numpy()
                    

                    x_data = x_data.reshape(120, 13, 13)
                    x_data = torch.Tensor(x_data)
                    # print(type(x_data), x_data.shape)
                    x_data = torch.as_tensor(x_data, device=device)
                    ## 추론
                    output = model(x_data)
                    result = torch.softmax(output,dim=1)
                    result = result.tolist()
                    print("\n")
                    print(f""" - none: {result[0][0]} \n - sit: {result[0][1]} \n - stand: {result[0][2]}""")
                    print(f"{max(result[0][0:3])}") #modify 0723

                    # if abs(result[0][0] - result[0][1]) > 0.7:
                    #     print("stand" if result[0][0] < result[0][1] else "sit")
                    # else:
                    #     print("none")

                    
                    


                    ## 다시 채우기
                    # 여기서 60개 제거
                    x_data = inf_data

                else:
                    x_data = np.append(x_data, inf_data, axis=0)
        
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
    GUI를 실행하면서 데이터를 수집할 스레드

    Args:
        - serial_port : 연결할 포트 /dev/ttyACM0
        - save_file_name : 저장할 파일 명칭

    run() :
        - csi 데이터 작성하기
    """
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.ser = serial.Serial(port=self.serial_port, baudrate=921600, bytesize=8, parity='N', stopbits=1)
        self.labelDict = {0:'', 1:'Steady State', 2:'Stand', 3:'Sit'}

        if self.ser.isOpen():
            print("OPEN SUCCESS")
        else:
            return

        # 데이터 수집 시작 플래그
        self.collectingData = False
        

    def run(self):
        csi_data_read_parse(self.ser, self.collectingData, self.labelDict)

class Camera():
    """
    이미지 수집을 위한 카메라 객체

    - 카메라 객체에 관한 정보(코덱, 너비, 높이, fps)
    - 라벨 딕셔너리
    """
    def __init__(self):

        # 이미지 디렉토리 설정
        self.photosFolderPath = "/data/csi-data/Photos"
        self.CheckedFolderPath = "/data/csi-data/Checked"
        if not os.path.exists(self.photosFolderPath) and not os.path.exists(self.CheckedFolderPath):
            os.makedirs(self.photosFolderPath)
            os.makedirs(self.CheckedFolderPath)

        # 카메라 설정
        self.camA= cv2.VideoCapture('/dev/video0') # CamA
        self.camB= cv2.VideoCapture('/dev/video2') # CamB
        # 만약 이전 오류로 인해 캠이 열리지 않는다면 0 대신 1(외부 카메라) 혹은 '/dev/video1'(문자열)로 넣어주면 됩니다.
        
        
        # 카메라 코덱 설정
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 코덱
        # self.out = None # Video


        # self.frameWidth = int(self.camA.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        # self.frameHeight = int(self.camA.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.frameRate = int(self.camA.get(cv2.CAP_PROP_FPS))
        self.record = False # 녹화 여부
        self.labelDict = {0:'', 1:'Steady State', 2:'Stand', 3:'Sit'} 

        # Yolov8 Modeling
        self.model = YOLO("./yolov8s-pose.pt")
        
    def recording(self, isStarted, isClosed):
        """
        카메라 객체의 촬영을 위한 메서드

        1. `Start`버튼을 누른 경우
            - `isStarted`가 True, `self.record`가 False 인 상태
        2. `Stop`버튼을 누른 경우
            - `isStarted`가 False, `self.record`가 True 인 상태
        3. 녹화 중인 상태
            - `isStarted`가 True, `self.record`가 True 인 상태

        Args:
            - `isStarted` : Start 버튼에 대한 불린 값
        
        
        """
        try:
            while True:
                # key = cv2.waitKey(1)
                retA, frameA = self.camA.read() # ret(boolean), frame(ndarray)
                retB, frameB = self.camB.read()
                # cv2.imshow('Recording Video NOW',frame) # 현재 프레임 보여줌  

                # 사진 데이터 수집 시작
                """
                - 사진 디렉토리 생성
                - 1개 사진 저장
                """
                if isStarted.value == True and self.record == False:
                    print(f'    📸 [{datetime.datetime.now()}] 이미지 데이터 수집을 시작합니다.')
                    self.record = True
                    timestamp_dirs = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                    os.makedirs(self.photosFolderPath + f"/{timestamp_dirs}")
                    os.makedirs(self.CheckedFolderPath + f"/{timestamp_dirs}")
                    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') <- 세부디렉토리 설정 가능
                    # self.out = cv2.VideoWriter(os.path.join(self.videosForlderPath, f'{timestamp}_{self.labelDict[labelkey.value]}.mp4'),cv2.CAP_FFMPEG, self.fourcc, self.frameRate, (self.frameWidth, self.frameHeight))

                # 사진 중지
                elif isStarted.value == False and self.record == True:
                    print(f'    📷 [{datetime.datetime.now()}] 영상 데이터 수집을 종료합니다.')
                    self.record = False
                    self.processing(self.model, timestamp_dirs)
                    # self.out.release()
                    # self.cmd = ''
                    # sp.call(self.cmd, shell=True)

                # 녹화중 : 녹화중인 상태일 때, 비디오 객체에 프레임을 담습니다.
                # 사진 시간별로 저장
                elif isStarted.value == True and self.record == True:
                    timestamp_image = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-4]
                    cv2.imwrite(os.path.join(self.photosFolderPath + f"/{timestamp_dirs}",f'{timestamp_image}_{self.labelDict[labelkey.value]}_L.jpg'), frameA)
                    cv2.imwrite(os.path.join(self.photosFolderPath + f"/{timestamp_dirs}", f'{timestamp_image}_{self.labelDict[labelkey.value]}_R.jpg'), frameB)


                    frameA, frameB = None, None
                    # print(f'        🎞️ [{datetime.datetime.now()}] {id(frame)} 영상 데이터 녹화중...')

                # 종료 : q버튼을 누르면 종료됩니다.
                elif isClosed == 1:
                    break

            self.cam.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            self.camA.release()
            self.camB.release()
            cv2.destroyAllWindows()      

    def processing(self, model, timestamp_dirs):
        """
        Yolov8 Image Processing

        - 인지 모델 처리된 이미지 저장
        - 라벨링 된 데이터 저장
            - 사람 수가 1명의 경우, 인물의 위치 정보와 관절 정보도 추가로 데이터 입력

        `원본사진경로 | # Person | Action | Location | 사람 좌표 | 사람 관절 좌표값`  

        """
        print(f'         🤖 [{datetime.datetime.now()}] 원본 사진을 검사하면서 예측을 시작합니다! ')
        writed = csv.writer(open(f"{self.CheckedFolderPath}/{timestamp_dirs}/{timestamp_dirs}.csv",'w')) # create csv file for labeling
        img_list = glob.glob(f"{self.photosFolderPath}/{timestamp_dirs}" + "/*.jpg")
        for img_dir in img_list:
            results = model.predict(img_dir, iou=0.5)
            # USE_CUDA = torch.cuda.is_available()
            # device = torch.device('cuda:0' if USE_CUDA else 'cpu')
            # print('현재 사용 device :', torch.cuda.get_device_name())
            for result in results:
                result.save(filename=f"{self.CheckedFolderPath}/{timestamp_dirs}/{result.path[-27:]}") # Save Processed Image File

                if len(result.boxes) == 1:
                    writed.writerow([result.path, len(result.boxes), "", ""]+ list(result.boxes[0].xyxy) + list(result.keypoints[0].xy)) # Write Row Data in CSV File
                else:
                    writed.writerow([result.path, len(result.boxes), "", ""])
        print(f'         👏 [{datetime.datetime.now()}] 예측이 종료되었습니다.')

if __name__ == '__main__':

    # MORE THAN PYTHON 3.6
    if sys.version_info < (3, 6):
        print("Python version should be > 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device") # ESP32s3 Module Port  for connecting 
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv',
                        help="Save the data printed by the serial port to a file")
    
    args = parser.parse_args()
    serial_port = args.port # /dev/ttyACM0
    file_name = args.store_file # ./csi_data.csv

    app = QApplication(sys.argv) # 명령어를 인수로 앱 객체 생성

    # STARTS UP WINDOW
    window = csi_data_graphical_window()
    window.show() 


    # SHARING VARIABLES
    isStarted = multiprocessing.Value('b', False) # 스위치가 켜졌느냐 안켜졌느냐
    isClosed = multiprocessing.Value('b', False)
    labelkey = multiprocessing.Value('i', 0)
    
    # SUB THREAD
    subthread = SubThread(serial_port)
    subthread.start()

    # CAMERA SETTINGS (multiprocessing)
    # camera = Camera()
    # rec = multiprocessing.Process(target=camera.recording, args=(isStarted,isClosed))
    # rec.start()


    sys.exit(app.exec())



"""
- csi_data_read_parse 함수에 strings 데이터 정보

CSI_DATA,
516197,
70:5d:cc:2d:fe:bc,
-42,11,1,5,0,0,1,0,0,0,0,-93,0,1,1,-1058868037,0,83,0,256,0,
"[0,0,0,0,0,0,0,0,0,0,0,0,-1,-10,-2,-10,-2,-11,-3,-11,-3,-11,-3,-11,-4,-11,-4,-11,-5,-11,-5,-11,-6,-11,-6,-11,-6,-10,-7,-10,-7,-10,-7,-10,-7,-10,-7,-10,-7,-9,-8,-9,-8,-9,-8,-8,-8,-9,-8,-9,-8,-9,-8,-9,0,0,-8,-10,-8,-11,-7,-11,-7,-12,-7,-12,-7,-12,-6,-13,-6,-13,-5,-14,-5,-14,-5,-14,-6,-14,-5,-14,-5,-14,-5,-15,-4,-14,-4,-14,-5,-14,-6,-13,-6,-13,-6,-13,-6,-13,-4,-14,-3,-14,-2,-14,-2,-14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-9,-1,-10,-1,-10,-2,-10,-3,-10,-3,-11,-3,-11,-4,-11,-5,-11,-5,-11,-6,-11,-6,-11,-6,-11,-6,-11,-7,-10,-7,-10,-7,-10,-7,-10,-8,-10,-8,-9,-8,-9,-8,-9,-8,-8,-8,-8,-8,-9,-8,-9,-8,-9,-9,-8,0,0,-8,-10,-8,-10,-7,-11,-7,-12,-7,-12,-7,-12,-7,-13,-6,-13,-5,-14,-6,-14,-5,-14,-6,-14,-5,-14,-6,-14,-5,-14,-5,-14,-5,-14,-5,-14,-6,-13,-6,-13,-6,-13,-6,-13,-4,-14,-3,-14,-3,-14,-2,-14,-1,-14,-1,-14,0,0,0,0,0,0]"

"""
