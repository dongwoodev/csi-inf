# Standard Libraty
from io import StringIO
import sys, os
import csv
import json
import re
import argparse
import datetime
import multiprocessing
import time
import warnings

# Third-party
from scipy.linalg import svd
from sklearn.decomposition import NMF
from tqdm import tqdm
import serial
import numpy as np
import pandas as pd
import torch

# GUI library
from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# MODEL
from models.model.CNN import CNN
from models.model.transformer import Transformer

# butterworth library
from scipy.signal import butter, filtfilt

# MQTT Library
from mqtt_config import BROKER_ADDRESS, PORT, TOPIC, create_mqtt_message
import paho.mqtt.client as mqtt

warnings.filterwarnings("ignore", category=DeprecationWarning)

processes = [] # Subprocess List

# GLOBAL VARIABLES & Initialize #
MAX_VALUE = 0 
MIN_VALUE = 0
CSI_DATA_INDEX = 1000
CSI_DATA_LLFT_COLUMNS = 64
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"] # 3(rssi), 6(mcs), 14(noise_floor), 18(local_timestamp)

CSI_DATA_COLUMNS = 384
GET_START_TIME = True
GET_EMPTY_INFO_START_TIME = True
FILENAME_TIMES = 0 
CURRENT_TIME = datetime.datetime.now()
LABELS = {"file":"", "occ": "", "act": "", "loc": ""}


garbage_cnt1 = multiprocessing.Value('i', 0)
garbage_cnt2 = multiprocessing.Value('i', 0)
garbage_cnt3 = multiprocessing.Value('i', 0)
garbage_cnt4 = multiprocessing.Value('i', 0)
simple_cnt =  0
T_type = "all"
sequence_len = 60 # sequence length of time series #MODIFY
sequence_len_prev = 50
isStarted = multiprocessing.Value('b', False)   
labelkey = multiprocessing.Value('i', 0)


csi_bt_array_shared = [multiprocessing.Array('d', 1000 * 114) for _ in range(4)]
csi_raw_array_shared = [multiprocessing.Array('d', 1000 * 114) for _ in range(4)]
csi_prev_array_shared = [multiprocessing.Array('d', 1000 * 114) for _ in range(4)]
locks = [multiprocessing.Lock() for _ in range(4)]


# csi_raw_array0 = np.zeros([CSI_DATA_INDEX, 114])
# csi_bt_array0 = np.zeros([CSI_DATA_INDEX, 114])
# csi_prev_array0 = np.zeros([CSI_DATA_INDEX, 114])
# csi_raw_array1 = np.zeros([CSI_DATA_INDEX, 114])
# csi_bt_array1 = np.zeros([CSI_DATA_INDEX, 114])
# csi_prev_array1 = np.zeros([CSI_DATA_INDEX, 114])
# csi_raw_array2 = np.zeros([CSI_DATA_INDEX, 114])
# csi_bt_array2 = np.zeros([CSI_DATA_INDEX, 114])
# csi_prev_array2 = np.zeros([CSI_DATA_INDEX, 114])
# csi_raw_array3 = np.zeros([CSI_DATA_INDEX, 114])
# csi_bt_array3 = np.zeros([CSI_DATA_INDEX, 114])
# csi_prev_array3 = np.zeros([CSI_DATA_INDEX, 114])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prev_data_BT  = np.zeros([sequence_len,192])
prev_amplitude0 = None # PREV DIFF AMPLITUDE
prev_amplitude1 = None # PREV DIFF AMPLITUDE
prev_amplitude2 = None # PREV DIFF AMPLITUDE
prev_amplitude3 = None # PREV DIFF AMPLITUDE

def parse_argument():
    parser = argparse.ArgumentParser(description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-m', '--model', dest='model', type=str, required=False, default='CNN')
    parser.add_argument('-a', '--acquire', dest='acq', action='store_true', required=False, default=False)
    parser.add_argument('-d', '--dir', dest='dir', type=str, required=False, default=datetime.datetime.now().strftime("%m%d"))
    args = parser.parse_args()
    return args.acq, args.dir, args.model

acq_bool, csi_dir, model_type = parse_argument()

# MQTT Configuration
broker_address = "localhost"
port = 1883
topic = "test/csi"
client = mqtt.Client() # Initialize MQTT client
client.connect(BROKER_ADDRESS, PORT)

# Load Model #
model_name = "1111_1118_queue_60W_20S_PREV" 
def load_model(path, n_classes):
    path = path.split("/")
    if model_type == "CNN":
        model = CNN(n_classes=n_classes)
        path[-2] = "CNN"
    else:
        model = Transformer(
            feature=114 if T_type == "HT" else 52 if T_type=="LT" else 166,
            d_model=64,
            n_head=4,
            max_len=(sequence_len-10) * 2,
            ffn_hidden=32,
            n_layers=2,
            drop_prob=0.1,
            n_classes=n_classes,
            device=device).to(device=device)
        path[-2] = "Transformer"

    path = "/".join(path)      
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

model_occ = load_model(f"/csi/weights/{model_name}/model/occ", n_classes=2).to(device)
model_loc = load_model(f"/csi/weights/{model_name}/model/loc", n_classes=2).to(device)
model_act = load_model(f"/csi/weights/{model_name}/model/act", n_classes=3).to(device)

CSI_SAVE_PATH = f"/csi/datasets/{csi_dir}"
os.makedirs(CSI_SAVE_PATH, exist_ok=True)

# GUI #
class csi_data_graphical_window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Quad-chips CSI SENSING")
        self.setGeometry(0, 0, 1500, 1400) # location(x, y), width, height

        # SETTING MAIN WIDGET & LAYOUT
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.layout = QVBoxLayout()
        self.mainWidget.setLayout(self.layout)


        # SETTING PYQTGRAPH
        self.graphWidget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphWidget)  # 그래프 위젯 추가

        self.plotItem_p1 = self.graphWidget.addPlot(row=0, col=0, title="PORT 1") 
        self.plotItem_p1.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p0 = self.graphWidget.addPlot(row=0, col=1, title="PORT 0") 
        self.plotItem_p0.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p1_prev = self.graphWidget.addPlot(row=0, col=2, title="PORT 1 (Previous)") 
        self.plotItem_p1_prev.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p0_prev = self.graphWidget.addPlot(row=0, col=3, title="PORT 0 (Previous)") 
        self.plotItem_p0_prev.setLabels(left='CSI Carrier Number', bottom='Time')

        self.plotItem_p3 = self.graphWidget.addPlot(row=1, col=0, title="PORT 3") 
        self.plotItem_p3.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p2 = self.graphWidget.addPlot(row=1, col=1, title="PORT 2") 
        self.plotItem_p2.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p3_prev = self.graphWidget.addPlot(row=1, col=2, title="PORT 3 (Previous)") 
        self.plotItem_p3_prev.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_p2_prev = self.graphWidget.addPlot(row=1, col=3, title="PORT 2 (Previous)") 
        self.plotItem_p2_prev.setLabels(left='CSI Carrier Number', bottom='Time')

        # SETTING HEATMAP
        self.heatmap_p1 = pg.ImageItem(border='w')
        self.plotItem_p1.addItem(self.heatmap_p1)
        self.heatmap_p0 = pg.ImageItem(border='w')
        self.plotItem_p0.addItem(self.heatmap_p0)
        self.heatmap_p1_prev = pg.ImageItem(border='w')
        self.plotItem_p1_prev.addItem(self.heatmap_p1_prev)
        self.heatmap_p0_prev = pg.ImageItem(border='w')
        self.plotItem_p0_prev.addItem(self.heatmap_p0_prev)

        self.heatmap_p3 = pg.ImageItem(border='w')
        self.plotItem_p3.addItem(self.heatmap_p3)
        self.heatmap_p2 = pg.ImageItem(border='w')
        self.plotItem_p2.addItem(self.heatmap_p2)
        self.heatmap_p3_prev = pg.ImageItem(border='w')
        self.plotItem_p3_prev.addItem(self.heatmap_p3_prev)
        self.heatmap_p2_prev = pg.ImageItem(border='w')
        self.plotItem_p2_prev.addItem(self.heatmap_p2_prev)   

        # COLOR SCALE(LUT)
        colors = np.array([[0,255,0,255],[0,0,0,255],[255,0,0,255]],dtype=np.ubyte)
        colormap = pg.colormap.getFromMatplotlib('inferno')
        colormap_prev = pg.ColorMap(pos=np.array([-1.0,0.0,1.0]), color=colors)

        self.heatmap_p1.setLookupTable(colormap.getLookupTable())
        self.heatmap_p0.setLookupTable(colormap.getLookupTable())
        self.heatmap_p3.setLookupTable(colormap.getLookupTable())
        self.heatmap_p2.setLookupTable(colormap.getLookupTable())

        self.heatmap_p1_prev.setLookupTable(colormap_prev.getLookupTable(-1.0,1.0,256))
        self.heatmap_p0_prev.setLookupTable(colormap_prev.getLookupTable(-1.0,1.0,256))
        self.heatmap_p3_prev.setLookupTable(colormap_prev.getLookupTable(-1.0,1.0,256))
        self.heatmap_p2_prev.setLookupTable(colormap_prev.getLookupTable(-1.0,1.0,256))

        self.absScaleMin = 0
        self.absScaleMax = 1
        self.absScaleMin_prev = -1
        self.absScaleMax_prev = 1

        # SETTING LAYOUT
        self.BottomLayout = QHBoxLayout()
        self.radioLayout = QVBoxLayout() # 라디오 버튼 그룹을 위한 세로 라벨 레이아웃
        self.labelGroupLayout = QVBoxLayout() # 클래스 라벨 그룹을 위한 세로 라벨 레이아웃
        
        ## RADIO BUTTON GROUP
        self.radioGroupBox = QGroupBox("Port Type")
        self.BottomLayout.addWidget(self.radioGroupBox)
        self.radioButton0 = QRadioButton("Raw Mode")
        self.radioButton1 = QRadioButton("Butterworth Mode")
        self.radioButton1.setChecked(True)  # 기본 선택값 설정
        self.radioLayout.addWidget(self.radioButton0)
        self.radioLayout.addWidget(self.radioButton1)
        self.radioGroupBox.setLayout(self.radioLayout)

        ## File Num Label
        self.labelGroupBox = QGroupBox("Labels")
        self.labelGroupLayout = QVBoxLayout() # 그룹박스 내부 레이아웃


        self.label1 = QLabel("")
        font = QFont("Arial", 30)
        self.label1.setFont(font)
        self.label1.setAlignment(Qt.AlignCenter)
        self.labelGroupLayout.addWidget(self.label1)

        ## LABEL GROUP
        self.label2 = QLabel("Occ")
        self.label3 = QLabel("Loc")
        self.label4 = QLabel("Act")
        self.label2.setFont(font)
        self.label3.setFont(font)
        self.label4.setFont(font)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label3.setAlignment(Qt.AlignCenter)
        self.label4.setAlignment(Qt.AlignCenter)
        self.fileLayout = QVBoxLayout() 
        self.occLayout = QVBoxLayout()
        self.locLayout = QVBoxLayout()
        self.actLayout = QVBoxLayout()
        self.labelGroupLayout.addWidget(self.label2)
        self.labelGroupLayout.addWidget(self.label3)
        self.labelGroupLayout.addWidget(self.label4)
        self.labelGroupBox.setLayout(self.labelGroupLayout)
        self.BottomLayout.addWidget(self.labelGroupBox)


        ## BUTTON
        self.pushButton = QPushButton("정지 상태")
        self.pushButton.setStyleSheet("background-color: gray; color: black;")
        self.pushButton.setMaximumHeight(400)
        self.pushButton.setMinimumHeight(100)        
        self.pushButton.clicked.connect(self.toggleButtonState)
        self.BottomLayout.addWidget(self.pushButton)
        self.isButtonStopped = False
        self.layout.addLayout(self.BottomLayout)

        # QTimer
        self.timer = QTimer()
        self.timer.setInterval(1.0) # update per 0.1s
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(0) # 0 >> 100


        # ======================================================= # 
        # Visualize initial data   
        # self.csi_raw_array_p0 = csi_raw_array0
        # self.csi_raw_array_p1 = csi_raw_array1
        # self.csi_raw_array_p2 = csi_raw_array2
        # self.csi_raw_array_p3 = csi_raw_array3

        # self.csi_bt_array_p0 = csi_bt_array0
        # self.csi_bt_array_p1 = csi_bt_array1
        # self.csi_bt_array_p2 = csi_bt_array2
        # self.csi_bt_array_p3 = csi_bt_array3

        # self.csi_prev_array_p0 = csi_prev_array0
        # self.csi_prev_array_p1 = csi_prev_array1
        # self.csi_prev_array_p2 = csi_prev_array2
        # self.csi_prev_array_p3 = csi_prev_array3

        # self.heatmap_p0.setImage(self.csi_bt_array_p0, levels=(self.absScaleMin, self.absScaleMax))
        # self.heatmap_p1.setImage(self.csi_bt_array_p1, levels=(self.absScaleMin, self.absScaleMax))
        # self.heatmap_p2.setImage(self.csi_bt_array_p2, levels=(self.absScaleMin, self.absScaleMax))
        # self.heatmap_p3.setImage(self.csi_bt_array_p3, levels=(self.absScaleMin, self.absScaleMax))

        # self.heatmap_p0_prev.setImage(self.csi_prev_array_p0, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
        # self.heatmap_p1_prev.setImage(self.csi_prev_array_p1, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
        # self.heatmap_p2_prev.setImage(self.csi_prev_array_p2, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
        # self.heatmap_p3_prev.setImage(self.csi_prev_array_p3, levels=(self.absScaleMin_prev, self.absScaleMax_prev))

    def update_graph(self):
        """시각화 업데이트"""

        for port_num in range(4):
            with csi_raw_array_shared[port_num].get_lock():  # 배열 잠금
                data_for_vis_raw = np.array(csi_raw_array_shared[port_num]).reshape(1000, 114)  # 공유 배열에서 데이터 읽기
            with csi_bt_array_shared[port_num].get_lock():  # 배열 잠금
                data_for_vis_bt = np.array(csi_bt_array_shared[port_num]).reshape(1000, 114)  # 공유 배열에서 데이터 읽기
            with csi_prev_array_shared[port_num].get_lock():
                data_for_vis_prev = np.array(csi_prev_array_shared[port_num]).reshape(1000, 114)  # 공유 배열에서 데이터 읽기

            # 데이터를 시각화에 반영
            if port_num == 0:
                self.heatmap_p0_prev.setImage(data_for_vis_prev, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
                if self.radioButton1.isChecked():
                    self.heatmap_p0.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p0.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
            elif port_num == 1:
                if self.radioButton1.isChecked():
                    self.heatmap_p1.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p1.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
                self.heatmap_p1_prev.setImage(data_for_vis_prev, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
            elif port_num == 2:
                if self.radioButton1.isChecked():
                    self.heatmap_p2.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p2.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
                self.heatmap_p2_prev.setImage(data_for_vis_prev, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
            elif port_num == 3:
                if self.radioButton1.isChecked():
                    self.heatmap_p3.setImage(data_for_vis_bt, levels=(self.absScaleMin, self.absScaleMax))
                elif self.radioButton0.isChecked():
                    self.heatmap_p3.setImage(data_for_vis_raw, levels=(self.absScaleMin, self.absScaleMax))
                self.heatmap_p3_prev.setImage(data_for_vis_prev, levels=(self.absScaleMin_prev, self.absScaleMax_prev))
     
        self.label1.setText(str(LABELS['file']))
        self.label2.setText(LABELS['occ'])
        self.label3.setText(LABELS['loc'])
        self.label4.setText(LABELS['act'])

    def toggleButtonState(self):
        if self.isButtonStopped:
            isStarted.value = False
            self.pushButton.setText("정지 상태")
            self.pushButton.setStyleSheet("background-color: gray; color: white;")
        else:
            isStarted.value = True
            # if empty_process:_init
            #     isEmpty.value = True # Empty
            # else:
                # print("\n ⏰ ")
            print("\n ⏰ ")
            self.pushButton.setText("취득중 상태")
            self.pushButton.setStyleSheet("background-color: blue; color: black;")
        self.isButtonStopped = not self.isButtonStopped

    def closeEvent(self, event):
        terminate_process() # 프로세스 종료
        event.accept() # 창닫기 이벤트

# ======================================================= #
            #  F   U   N   C    T   I  O  N #
# ======================================================= #

def butterworth_filter(data, cutoff, fs, order=5, filter_type='low', prev_data=None):
    nyquist = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)

    if prev_data is not None: # BT
        global sequence_len
        global prev_data_BT
        bt_data = np.zeros([sequence_len*2,192])
        bt_data[:sequence_len,:] = prev_data
        bt_data[sequence_len:,:] = data
        prev_data_BT = data
        filtered_data = filtfilt(b, a, bt_data, axis=0)
        filtered_data = np.ascontiguousarray(filtered_data)
        return filtered_data[sequence_len:, :]
    else: # INPUT
        filtered_data = filtfilt(b, a, data, axis=0) # 필터 적용
        filtered_data = np.ascontiguousarray(filtered_data) # 음수 스트라이드를 방지하기 위해 복사
        return filtered_data

def butterworth_filter_BT(data, cutoff, fs, order=5, filter_type='low'):
    filtered_data = butterworth_filter(data, cutoff, fs, order, filter_type, prev_data_BT)
    return filtered_data

def butterworth_filter_INPUT(data, cutoff, fs, order=5, filter_type='low'):
    return butterworth_filter(data, cutoff, fs, order, filter_type)

def update_csi_data(port_num, new_data, sequence_len, flag="bt"):
    """
    - Update Csi data for Visualization 
    """
    with locks[port_num]:  # 포트별 잠금
    # 공유 배열을 2차원 배열로 변환
        if flag == "raw":
            shared_array = np.frombuffer(csi_raw_array_shared[port_num].get_obj()).reshape(1000, 114)
        elif flag == "bt":
            shared_array = np.frombuffer(csi_bt_array_shared[port_num].get_obj()).reshape(1000, 114)
        else:
            shared_array = np.frombuffer(csi_prev_array_shared[port_num].get_obj()).reshape(1000, 114)

        # 기존 데이터를 앞으로 이동
        if flag in ["raw", "bt"]: # 114 * 60
            shared_array[:-sequence_len] = shared_array[sequence_len:]
        else: # 114 * 50
            shared_array[:-sequence_len_prev] = shared_array[sequence_len_prev:]

        # 새로운 데이터를 끝에 추가
        if flag in ["raw", "bt"]:
            shared_array[-sequence_len:] = new_data
        else:
            shared_array[-sequence_len_prev:] = new_data

def get_amplitude(csi, is_sequence=False):
    csi = np.array(csi)
    if is_sequence==True: 
        even_elements = csi[:,::2]
        odd_elements = csi[:,1::2]
    else:
        even_elements = csi[::2]
        odd_elements = csi[1::2]
    amplitude = np.sqrt(np.square(even_elements) + np.square(odd_elements))
    return amplitude

def remove_null_csi(csi_input):
    csi_data = []
    remove_indices = np.concatenate((np.arange(0,6), np.arange(32,33), np.arange(59, 66), np.arange(123,134), np.arange(191,192)))
    csi_data = np.delete(csi_input, remove_indices, axis=1)
    return csi_data

def terminate_process():
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

def csi_preprocessing(raw_data, port_num, return_type = "raw", empty_csi = None):
    global prev_amplitude0
    global prev_amplitude1
    global prev_amplitude2
    global prev_amplitude3

    amplitude = get_amplitude(raw_data, True) # 1. Amplitude

    if return_type == "raw":
        amplitude = amplitude /20.0
        return amplitude 
    
    if return_type == "bt":
        amplitude = butterworth_filter_BT(amplitude, cutoff=0.7, fs=5, order=1, filter_type='low') /20.0
        return amplitude 

    if return_type == "prev": 
        # 2. Butterworth
        amplitude = butterworth_filter_INPUT(amplitude, cutoff=0.7, fs=5, order=1, filter_type='low') /20.0

        # 3. Side Remove
        amplitude = amplitude[5:-5,:]

        # 4. Previous Subtract
        if prev_amplitude0 is None and port_num == 0:
            prev_amplitude0 = amplitude  # 처음 들어오는 데이터는 prev_amplitude에 저장
            amplitude = amplitude - prev_amplitude0
            return amplitude  # 그대로 반환
        elif prev_amplitude1 is None and port_num == 1:
            prev_amplitude1 = amplitude  # 처음 들어오는 데이터는 prev_amplitude에 저장
            amplitude = amplitude - prev_amplitude1
            return amplitude  # 그대로 반환
        elif prev_amplitude2 is None and port_num == 2:
            prev_amplitude2 = amplitude  # 처음 들어오는 데이터는 prev_amplitude에 저장
            amplitude = amplitude - prev_amplitude2
            return amplitude  # 그대로 반환
        elif prev_amplitude3 is None and port_num == 3:
            prev_amplitude3 = amplitude  # 처음 들어오는 데이터는 prev_amplitude에 저장
            amplitude = amplitude - prev_amplitude3
            return amplitude  # 그대로 반환

        if port_num == 0:
            averaged_amplitude = np.mean(np.array(prev_amplitude0), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3 # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude0 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude
        elif port_num == 1:
            averaged_amplitude = np.mean(np.array(prev_amplitude1), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude)  * 3 # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude1 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude
        elif port_num == 2:
            averaged_amplitude = np.mean(np.array(prev_amplitude2), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3 # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude2 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude
        elif port_num == 3:
            averaged_amplitude = np.mean(np.array(prev_amplitude3), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3 # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude3 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude

def predict(model, X, classes:list):
    softmax = torch.nn.Softmax(dim=1)
    
    X = torch.unsqueeze(X, dim=0)
    
    output = softmax(model(X))
    result = output.detach().cpu().numpy()
    

    pred_index = result.argmax(axis=1)[0]
    pred_class = classes[pred_index]
    pred_confidence = float(result[0][pred_index])
    return pred_class, pred_confidence


# -----------------------------------------------------------
# # # # CODE
# -----------------------------------------------------------
def csi_data_read_parse(ser):

    global GET_START_TIME
    global GET_EMPTY_INFO_START_TIME
    global CURRENT_TIME
    global MAX_VALUE
    global MIN_VALUE
    global T_type
    global sequence_len
    global acq_bool
    global LABELS
    

    global simple_cnt
    total_data = [] 


    ser_str = str(ser)
    match = re.search(r"port='(/dev/tty\w+)'", ser_str)
    port_name = match.group(1).split("/")[-1]
    port_num = int(port_name.replace("ttyACM", "")) % 4

    FILENAME_TIMES = 0
    start_time = None  # 초기화

    ## CSV FILE ##
    filename_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if acq_bool:
        csvFileName = f"{CSI_SAVE_PATH}/{filename_time}_PORT{port_num}.csv"
        csvFile = open(csvFileName, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)


    prev_csi_raw_data = None
    while True: 
        strings = ser.readline() # READ CSI DATA

        if not strings:
            break
        result = re.findall(r"-?\d+", strings.decode("utf-8")) # Demical Number Extract(String Type) 
        csi_raw_data = list(map(int, result))[28:] # Int list type

        if len(csi_raw_data) not in [384]: 
            if port_num == 0:
                with garbage_cnt1.get_lock():
                    garbage_cnt1.value += 1
            if port_num == 1:
                with garbage_cnt2.get_lock():
                    garbage_cnt2.value += 1
            if port_num == 2:
                with garbage_cnt3.get_lock():
                    garbage_cnt3.value += 1
            if port_num == 3:
                with garbage_cnt4.get_lock():
                    garbage_cnt4.value += 1
            print(f"[{port_num}] {len(csi_raw_data)}")
            continue

        if csi_raw_data[0] != 0:
            print(f"{port_num} NZERO ERROR")
            continue

        meta_data = [port_num, result[6], result[9], result[17], result[21]]
        # 3(rssi), 6(mcs), 14(noise_floor), 18(local_timestamp) index +3

        # RAW DATA ACQUISITION
        acquisition_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
        if (acq_bool == True) and (isStarted.value == True): 
            zeroed_data = np.zeros(384)
            zeroed_data[132:246] = csi_raw_data[132:246]
            zeroed_data[254:368] = csi_raw_data[254:368]
            final_data = np.concatenate(([acquisition_time], meta_data, zeroed_data))
            csvWriter.writerow(final_data)
            print(f"[{acquisition_time}] data collecting")

        if isStarted.value == True:

            GET_START_TIME = True 
            total_data.append(csi_raw_data) # FOR PREDICT & VISUALIZE

            if GET_START_TIME == True:
                start_time = datetime.datetime.now()
                GET_START_TIME = False
                
            # 3. PREPROCESSING #
            if len(total_data) == sequence_len: # 60
                if start_time is not None and (datetime.datetime.now() - start_time).total_seconds() <= 0.5:
                    GET_START_TIME = True

                    # PREPROCESSING 
                    vis_data_raw = csi_preprocessing(total_data, port_num)
                    vis_data_bt = csi_preprocessing(total_data, port_num, 'bt')
                    vis_data_prev = csi_preprocessing(total_data, port_num, 'prev')
                    vis_prev = np.zeros_like(vis_data_raw)

                    total_data = []
                    print("\n")
                    FILENAME_TIMES += 1

                    # VISUALIZATION
                    selected_data_raw = np.concatenate((vis_data_raw[:, 127:184], vis_data_raw[:, 66:123]), axis=1)
                    selected_data_bt = np.concatenate((vis_data_bt[:, 127:184], vis_data_bt[:, 66:123]), axis=1)
                    selected_data_prev = np.concatenate((vis_data_prev[:, 127:184], vis_data_prev[:, 66:123]), axis=1)

                    update_csi_data(port_num, selected_data_raw, sequence_len, flag="raw")
                    update_csi_data(port_num, selected_data_bt, sequence_len)
                    update_csi_data(port_num, selected_data_prev, sequence_len, flag="prev")

                    # PREDICTION
                    # print(port_num)
                    # input_data = remove_null_csi(vis_data_prev) #or vis_data_emp or vis_data_diff
                    # inf_data = torch.tensor(input_data, dtype=torch.float32).to(device)
                    # inf_time = str(datetime.datetime.now())
                    # occ, occ_score = predict(model_occ, inf_data, ["EMPTY", "OCCUPIED"])
                    # loc, loc_score = predict(model_loc, inf_data, ["LEFT", "RIGHT"])
                    # act, act_score = predict(model_act, inf_data, ["SIT", "STAND", "WALK"])
                    # print(f"----------------------------------\n{port_num} inf time : {inf_time} \n OCC:{occ} ({round(occ_score,2)}) \n LOC: {loc} ({round(loc_score,2)}) \n ACT: {act} ({round(act_score,2)})\n----------------------------------")
                    # if port_num == 0:
                    #     LABELS = dict(zip(['file','occ', 'loc', 'act'], [FILENAME_TIMES, occ, loc, act]))
                    
                    # MQTT
                    # message = create_mqtt_message(port=port_num, cc=occ, occ_score=round(occ_score,2), loc=loc, loc_score=round(loc_score,2), act=act, act_score=round(act_score,2), timestamp=inf_time)
                    # client.publish(TOPIC, message)

                    total_data = [] 
                else:
                    total_data = [] 
                    GET_START_TIME = True    

    ser.close()

class SubProcess():

    def __init__(self, port_num):
        port_num = str(port_num)
        self.ser = serial.Serial(port=f'/dev/ttyACM{port_num}', baudrate=921600, bytesize=8, parity='N', stopbits=1)
        
        if self.ser.isOpen():
            print("OPEN SUCCESS")
        else:
            return


    def start(self):
        ser_str = str(self.ser)
        match = re.search(r"port='(/dev/tty\w+)'", ser_str)
        port_name = match.group(1).split("/")[-1]
        port_num = int(port_name.replace("ttyACM", "")) % 4
        if port_num == 0:
            csi_data_read_parse(self.ser)
        elif port_num == 1:
            csi_data_read_parse(self.ser)
        elif port_num == 2:
            csi_data_read_parse(self.ser)
        elif port_num == 3:
            csi_data_read_parse(self.ser)

if __name__ == '__main__':

    # MORE THAN PYTHON 3.6
    if sys.version_info < (3, 6):
        print("Python version should be > 3.6")
        exit()

    app = QApplication(sys.argv)

    # SHARING VALIABLES
    isStarted = multiprocessing.Value('b', False)   
    labelkey = multiprocessing.Value('i', 0)

    # STARTS UP WINDOWS(GUI)
    window = csi_data_graphical_window()
    window.show()
    
    # SUB PROCESS
    # subprocess0 = multiprocessing.Process(target=SubProcess(0).start)
    # subprocess0.start()

    # subprocess1 = multiprocessing.Process(target=SubProcess(1).start)
    # subprocess1.start()

    # subprocess2 = multiprocessing.Process(target=SubProcess(2).start)
    # subprocess2.start()

    # subprocess3 = multiprocessing.Process(target=SubProcess(3).start)
    # subprocess3.start()
    
    for port_num in range(4):
        process = multiprocessing.Process(target=SubProcess(port_num).start)
        processes.append(process)  # 프로세스 리스트에 추가
        process.start()


    sys.exit(app.exec())
