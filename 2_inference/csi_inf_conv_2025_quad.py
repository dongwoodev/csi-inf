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

# GLOBAL VARIABLES & Initialize #
MAX_VALUE = 0 
MIN_VALUE = 0
CSI_DATA_INDEX = 1000
CSI_DATA_LLFT_COLUMNS = 64
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]
# 3(rssi), 6(mcs), 14(noise_floor), 18(local_timestamp)

CSI_DATA_COLUMNS = 384
GET_START_TIME = True
GET_EMPTY_INFO_START_TIME = True
FILENAME_TIMES = 0 
CURRENT_TIME = datetime.datetime.now()
LABELS = {"file":"", "occ": "", "act": "", "loc": ""}


T_type = "all"
sequence_len = 60 # sequence length of time series #MODIFY

# PORT 0
csi_raw_data_array0 = np.zeros([CSI_DATA_INDEX, 192])
csi_bt_data_array0 = np.zeros([CSI_DATA_INDEX, 192])
csi_emp_data_array0 = np.zeros([CSI_DATA_INDEX, 192])
csi_diff_data_array0 = np.zeros([CSI_DATA_INDEX, 192])

# PORT 1
csi_raw_data_array1 = np.zeros([CSI_DATA_INDEX, 192])
csi_bt_data_array1 = np.zeros([CSI_DATA_INDEX, 192])
csi_emp_data_array1 = np.zeros([CSI_DATA_INDEX, 192])
csi_diff_data_array1 = np.zeros([CSI_DATA_INDEX, 192])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prev_data_BT  = np.zeros([sequence_len,192])
prev_data_INPUT  = np.zeros([sequence_len,192])
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

# CSI_SAVE_PATH #
# CSI_SAVE_PATH_0 = f"/csi/datasets/{csi_dir}_PORT0"
# CSI_SAVE_PATH_1 = f"/csi/datasets/{csi_dir}_PORT1"
# os.makedirs(CSI_SAVE_PATH_0, exist_ok=True)
# os.makedirs(CSI_SAVE_PATH_1, exist_ok=True)
CSI_SAVE_PATH = f"/csi/datasets/{csi_dir}"
os.makedirs(CSI_SAVE_PATH, exist_ok=True)


# GUI #
class csi_data_graphical_window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multiple CSI SENSING")
        self.setGeometry(0, 0, 1500, 1400) # location(x, y), width, height

        # SETTING MAIN WIDGET & LAYOUT
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.layout = QVBoxLayout()
        self.mainWidget.setLayout(self.layout)

        # SETTING PYQTGRAPH
        self.graphWidget = pg.GraphicsLayoutWidget()
        self.plotItem_raw0 = self.graphWidget.addPlot(row=0, col=0, title="PORT 0 (RAW)") 
        self.plotItem_raw0.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_bt0 = self.graphWidget.addPlot(row=0, col=1, title="PORT 0 (Butter)") 
        self.plotItem_bt0.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_diff0 = self.graphWidget.addPlot(row=0, col=2, title="PORT 0 (Previous)") 
        self.plotItem_diff0.setLabels(left='CSI Carrier Number', bottom='Time')

        self.plotItem_raw1 = self.graphWidget.addPlot(row=1, col=0, title="PORT 1 (RAW)") 
        self.plotItem_raw1.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_bt1 = self.graphWidget.addPlot(row=1, col=1, title="PORT 1 (Butter)") 
        self.plotItem_bt1.setLabels(left='CSI Carrier Number', bottom='Time')
        self.plotItem_diff1 = self.graphWidget.addPlot(row=1, col=2, title="PORT 1 (Previous)") 
        self.plotItem_diff1.setLabels(left='CSI Carrier Number', bottom='Time')

        # SETTING HEATMAP
        self.heatmap_raw0 = pg.ImageItem(border='w')
        self.plotItem_raw0.addItem(self.heatmap_raw0)
        self.heatmap_bt0 = pg.ImageItem(border='w')
        self.plotItem_bt0.addItem(self.heatmap_bt0)
        self.heatmap_diff0 = pg.ImageItem(border='w')
        self.plotItem_diff0.addItem(self.heatmap_diff0)

        self.heatmap_raw1 = pg.ImageItem(border='w')
        self.plotItem_raw1.addItem(self.heatmap_raw1)
        self.heatmap_bt1 = pg.ImageItem(border='w')
        self.plotItem_bt1.addItem(self.heatmap_bt1)
        self.heatmap_diff1 = pg.ImageItem(border='w')
        self.plotItem_diff1.addItem(self.heatmap_diff1)    

        # COLOR SCALE(LUT)
        colors = np.array([[0,255,0,255],[0,0,0,255],[255,0,0,255]],dtype=np.ubyte)
        colormap_raw = pg.colormap.getFromMatplotlib('inferno')
        colormap_bt = pg.colormap.getFromMatplotlib('inferno')
        colormap_diff = pg.ColorMap(pos=np.array([-1.0,0.0,1.0]), color=colors)

        ## PORT 0
        self.heatmap_raw0.setLookupTable(colormap_raw.getLookupTable())
        self.heatmap_bt0.setLookupTable(colormap_bt.getLookupTable())
        self.heatmap_diff0.setLookupTable(colormap_diff.getLookupTable(-1.0,1.0,256))

        ## PORT 1
        self.heatmap_raw1.setLookupTable(colormap_raw.getLookupTable())
        self.heatmap_bt1.setLookupTable(colormap_bt.getLookupTable())
        self.heatmap_diff1.setLookupTable(colormap_diff.getLookupTable(-1.0,1.0,256))

        self.absScaleMin_RAW = 0
        self.absScaleMax_RAW = 1
        self.absScaleMin_BT = 0
        self.absScaleMax_BT = 1
        self.absScaleMin_Input = -1
        self.absScaleMax_Input = 1
        self.absScaleMin_Diff = -1
        self.absScaleMax_Diff = 1

        # SETTING LAYOUT
        self.hLayout = QHBoxLayout()
        self.BottomLayout = QHBoxLayout()
        self.fileLayout = QVBoxLayout() 
        self.occLayout = QVBoxLayout()
        self.locLayout = QVBoxLayout()
        self.actLayout = QVBoxLayout()
        self.layout.addWidget(self.graphWidget) 
        self.layout.addLayout(self.BottomLayout) 
        self.BottomLayout.addLayout(self.fileLayout)
        self.BottomLayout.addLayout(self.occLayout) 
        self.BottomLayout.addLayout(self.locLayout)
        self.BottomLayout.addLayout(self.actLayout)

        self.timer = QTimer()
        self.timer.setInterval(1.0) # update per 0.1s
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(0) # 0 >> 100

        # SETTING BUTTON
        self.pushButton = QPushButton("정지 상태")
        self.pushButton.setStyleSheet("background-color: gray; color: black;")
        self.pushButton.setMaximumHeight(100)
        self.pushButton.setMinimumHeight(100)        
        self.pushButton.clicked.connect(self.toggleButtonState)
        self.BottomLayout.addWidget(self.pushButton)
        self.isButtonStopped = False

        # SETTING LABEL (NEW)
        self.label1 = QLabel("File")
        self.label2 = QLabel("Occ")
        self.label3 = QLabel("Loc")
        self.label4 = QLabel("Act")
        font = QFont("Arial", 30)
        self.label1.setFont(font)
        self.label2.setFont(font)
        self.label3.setFont(font)
        self.label4.setFont(font)
        self.fileLayout.addWidget(self.label1)
        self.occLayout.addWidget(self.label2)
        self.locLayout.addWidget(self.label3)
        self.actLayout.addWidget(self.label4)

        self.csi_raw_abs_array0 = csi_raw_data_array0
        self.csi_bt_abs_array0 = csi_bt_data_array0
        self.csi_diff_abs_array0 = csi_diff_data_array0
        self.heatmap_raw0.setImage(self.csi_raw_abs_array0, levels=(self.absScaleMin_RAW, self.absScaleMax_RAW))
        self.heatmap_bt0.setImage(self.csi_bt_abs_array0, levels=(self.absScaleMin_BT, self.absScaleMax_BT))
        self.heatmap_diff0.setImage(self.csi_diff_abs_array0, levels=(self.absScaleMin_Diff, self.absScaleMax_Diff))

        self.csi_raw_abs_array1 = csi_raw_data_array1
        self.csi_bt_abs_array1 = csi_bt_data_array1
        self.csi_diff_abs_array1 = csi_diff_data_array1
        self.heatmap_raw1.setImage(self.csi_raw_abs_array1, levels=(self.absScaleMin_RAW, self.absScaleMax_RAW))
        self.heatmap_bt1.setImage(self.csi_bt_abs_array1, levels=(self.absScaleMin_BT, self.absScaleMax_BT))
        self.heatmap_diff1.setImage(self.csi_diff_abs_array1, levels=(self.absScaleMin_Diff, self.absScaleMax_Diff))

    def update_graph(self):
        """시각화 업데이트"""

        # PORT 0
        self.csi_raw_abs_array0 = csi_raw_data_array0
        self.csi_bt_abs_array0 = csi_bt_data_array0
        self.csi_diff_abs_array0 = csi_diff_data_array0
        self.heatmap_raw0.setImage(self.csi_raw_abs_array0, levels=(self.absScaleMin_RAW, self.absScaleMax_RAW))
        self.heatmap_bt0.setImage(self.csi_bt_abs_array0, levels=(self.absScaleMin_BT, self.absScaleMax_BT))
        self.heatmap_diff0.setImage(self.csi_diff_abs_array0, levels=(self.absScaleMin_Diff, self.absScaleMax_Diff))

        # PORT 1
        self.csi_raw_abs_array1 = csi_raw_data_array1
        self.csi_bt_abs_array1 = csi_bt_data_array1
        self.csi_emp_abs_array1 = csi_emp_data_array1
        self.csi_diff_abs_array1 = csi_diff_data_array1
        self.heatmap_raw1.setImage(self.csi_raw_abs_array1, levels=(self.absScaleMin_RAW, self.absScaleMax_RAW))
        self.heatmap_bt1.setImage(self.csi_bt_abs_array1, levels=(self.absScaleMin_BT, self.absScaleMax_BT))
        self.heatmap_diff1.setImage(self.csi_diff_abs_array1, levels=(self.absScaleMin_Diff, self.absScaleMax_Diff))

        self.label1.setText(str(LABELS['file']))
        self.label2.setText(LABELS['occ'] + ", ")
        self.label3.setText(LABELS['loc'] + ", ")
        self.label4.setText(LABELS['act'] + ", ")

    def toggleButtonState(self):
        if self.isButtonStopped:
            isStarted.value = False
            self.pushButton.setText("정지 상태")
            self.pushButton.setStyleSheet("background-color: gray; color: white;")
        else:
            isStarted.value = True
            # if empty_process:
            #     isEmpty.value = True # Empty
            # else:
                # print("\n ⏰ ")
            print("\n ⏰ ")
            self.pushButton.setText("취득중 상태")
            self.pushButton.setStyleSheet("background-color: blue; color: black;")
        self.isButtonStopped = not self.isButtonStopped

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
        amplitude = butterworth_filter_BT(amplitude, cutoff=0.4, fs=5, order=1, filter_type='low') /20.0
        return amplitude

    if return_type == "diff": 
        # 2. Butterworth
        amplitude = butterworth_filter_INPUT(amplitude, cutoff=0.4, fs=5, order=1, filter_type='low') /20.0

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
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3  # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude0 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude
        elif port_num == 1:
            averaged_amplitude = np.mean(np.array(prev_amplitude1), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3  # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude1 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude
        elif port_num == 2:
            averaged_amplitude = np.mean(np.array(prev_amplitude1), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3  # 현재 데이터와 이전 평균 데이터의 차이 계산
            prev_amplitude2 = amplitude  # 현재 amplitude를 prev_amplitude에 저장
            return diff_amplitude
        elif port_num == 3:
            averaged_amplitude = np.mean(np.array(prev_amplitude1), axis=0)  # 이전 amplitude의 평균 계산
            diff_amplitude = np.subtract(amplitude, averaged_amplitude) * 3  # 현재 데이터와 이전 평균 데이터의 차이 계산
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


def csi_data_read_parse(ser):

    global GET_START_TIME
    global GET_EMPTY_INFO_START_TIME
    global CURRENT_TIME
    global MAX_VALUE
    global MIN_VALUE
    global T_type
    # global isEmpty
    global sequence_len
    global acq_bool
    global LABELS
    # global empty_process
    
    total_data = [] 
    total_acq_data = []


    ser_str = str(ser)
    match = re.search(r"port='(/dev/tty\w+)'", ser_str)
    port_name = match.group(1).split("/")[-1]
    port_num = int(port_name.replace("ttyACM", "")) % 4

    FILENAME_TIMES = 0
    start_time = None  # 초기화

    ## CSV FILE ##
    filename_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if acq_bool:
        if port_num == 0:
            csvFileName = f"{CSI_SAVE_PATH}/{filename_time}_PORT{port_num}.csv"
        elif port_num == 1:
            csvFileName = f"{CSI_SAVE_PATH}/{filename_time}_PORT{port_num}.csv"
        elif port_num == 2:
            csvFileName = f"{CSI_SAVE_PATH}/{filename_time}_PORT{port_num}.csv"
        elif port_num == 3:
            csvFileName = f"{CSI_SAVE_PATH}/{filename_time}_PORT{port_num}.csv"
        csvFile = open(csvFileName, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)

    prev_csi_raw_data = None

    while True: 
        strings = ser.readline() # READ CSI DATA
        if not str(strings):
            break

        result = re.findall(r"-?\d+", strings.decode("utf-8")) # Demical Number Extract(String Type) 
        csi_raw_data = list(map(int, result))[27:] # Int list type
        acquisition_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 

        if len(csi_raw_data) not in [384]: 
            continue

        if csi_raw_data[0] != 0:
            continue

        # if csi_raw_data[128] != 0:
        #     print(csi_raw_data)
        #     continue
        excep_amp = get_amplitude(csi_raw_data)
        if sum(excep_amp[128:132]) > 0.0 or excep_amp[6] == 0.0:
            csi_raw_data = prev_csi_raw_data

        meta_data = [result[6], result[9], result[17], result[21]]
        # 3(rssi), 6(mcs), 14(noise_floor), 18(local_timestamp) index +3

        
        prev_csi_raw_data = csi_raw_data
        # RAW DATA ACQUISITION
        if (acq_bool == True) and (isStarted.value == True): 
            csvWriter.writerow([acquisition_time] +  meta_data + csi_raw_data)

        if isStarted.value == True:

            GET_START_TIME = True 
            total_data.append(csi_raw_data) # FOR PREDICT & VISUALIZE
            total_acq_data.append([acquisition_time] +  meta_data + csi_raw_data) # FOR ACQUISTION

            if GET_START_TIME == True:
                start_time = datetime.datetime.now()
                GET_START_TIME = False
                
            # 3. PREPROCESSING #
            if len(total_data) == sequence_len:
                if start_time is not None and (datetime.datetime.now() - start_time).total_seconds() <= 0.5:
                    GET_START_TIME = True
                    # total_data = np.array(total_data)[:60,:]
                    # PREPROCESSING 
                    vis_data_raw = csi_preprocessing(total_data, port_num)
                    vis_data_bt = csi_preprocessing(total_data, port_num, 'bt')
                    vis_data_diff = csi_preprocessing(total_data, port_num, 'diff')

                    if port_num == 2:
                        print(2)
                        print(len(vis_data_raw[0]),len(vis_data_bt[0]))
                    elif port_num == 1:
                        print(1)
                        print(len(vis_data_raw[0]),len(vis_data_bt[0]))

                    # vis_data_emp = csi_preprocessing(total_data, port_num, 'empty', empty_feature) # 50 192
                    vis_diff = np.zeros_like(vis_data_raw)
                    # vis_emp[5:-5,:] = vis_data_emp
                    vis_diff[5:-5, :] = vis_data_diff


                    total_data = []
                    print("\n")
                    FILENAME_TIMES += 1

                    # VISUALIZATION
                    if port_num == 0:
                        csi_raw_data_array0[:-sequence_len] = csi_raw_data_array0[sequence_len:]
                        csi_raw_data_array0[-sequence_len:] = vis_data_raw[:, :]
                        csi_bt_data_array0[:-sequence_len] = csi_bt_data_array0[sequence_len:]
                        csi_bt_data_array0[-sequence_len:] = vis_data_bt[:, :]
                        # csi_emp_data_array0[:-sequence_len] = csi_emp_data_array0[sequence_len:]
                        # csi_emp_data_array0[-sequence_len:] = vis_emp[:, :]
                        csi_diff_data_array0[:-sequence_len] = csi_diff_data_array0[sequence_len:]
                        csi_diff_data_array0[-sequence_len:] = vis_diff[:, :]
                    elif port_num == 1:
                        csi_raw_data_array1[:-sequence_len] = csi_raw_data_array1[sequence_len:]
                        csi_raw_data_array1[-sequence_len:] = vis_data_raw[:, :]
                        csi_bt_data_array1[:-sequence_len] = csi_bt_data_array1[sequence_len:]
                        csi_bt_data_array1[-sequence_len:] = vis_data_bt[:, :]
                        # csi_emp_data_array1[:-sequence_len] = csi_emp_data_array1[sequence_len:]
                        # csi_emp_data_array1[-sequence_len:] = vis_emp[:, :]
                        csi_diff_data_array1[:-sequence_len] = csi_diff_data_array1[sequence_len:] 
                        csi_diff_data_array1[-sequence_len:] = vis_diff[:, :]                        
                    

                    # PREV_SENSING 
                    print(port_num)
                    input_data = remove_null_csi(vis_data_diff) #or vis_data_emp or vis_data_diff
                    inf_data = torch.tensor(input_data, dtype=torch.float32).to(device)
                    inf_time = str(datetime.datetime.now())
                    occ, occ_score = predict(model_occ, inf_data, ["EMPTY", "OCCUPIED"])
                    loc, loc_score = predict(model_loc, inf_data, ["LEFT", "RIGHT"])
                    act, act_score = predict(model_act, inf_data, ["SIT", "STAND", "WALK"])
                    print(f"----------------------------------\n{port_num} inf time : {inf_time} \n OCC:{occ} ({round(occ_score,2)}) \n LOC: {loc} ({round(loc_score,2)}) \n ACT: {act} ({round(act_score,2)})\n----------------------------------")
                    if port_num == 0:
                        LABELS = dict(zip(['file','occ', 'loc', 'act'], [FILENAME_TIMES, occ, loc, act]))
                    
                    # MQTT
                    # message = create_mqtt_message(port=port_num, cc=occ, occ_score=round(occ_score,2), loc=loc, loc_score=round(loc_score,2), act=act, act_score=round(act_score,2), timestamp=inf_time)
                    # client.publish(TOPIC, message)

                    total_data = [] 
                    total_acq_data = []
                else:
                    total_data = [] 
                    total_acq_data = []
                    GET_START_TIME = True    

    ser.close()

class SubThread(QThread):
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.ser = serial.Serial(port=self.serial_port, baudrate=921600, bytesize=8, parity='N', stopbits=1)
        
        if self.ser.isOpen():
            print("OPEN SUCCESS")
        else:
            return

    def run(self):
        csi_data_read_parse(self.ser)

if __name__ == '__main__':

    # MORE THAN PYTHON 3.6
    if sys.version_info < (3, 6):
        print("Python version should be > 3.6")
        exit()

    app = QApplication(sys.argv)

    # SHARING VALIABLES
    isStarted = multiprocessing.Value('b', False)
    # isEmpty = multiprocessing.Value('b', False)    
    labelkey = multiprocessing.Value('i', 0)

    # STARTS UP WINDOWS(GUI)
    window = csi_data_graphical_window()
    window.show()
    
    # SUB THREAD
    subthread0 = SubThread('/dev/ttyACM0')
    subthread0.start()

    subthread1 = SubThread('/dev/ttyACM1')
    subthread1.start()

    subthread2 = SubThread('/dev/ttyACM2')
    subthread2.start()

    subthread3 = SubThread('/dev/ttyACM3')
    subthread3.start()

    
    sys.exit(app.exec())
