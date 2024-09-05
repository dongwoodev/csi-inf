
# installation Process

- file path
```
esp/esp-csi/examples/get-started/tools/
```


## Conda Env Installation

```
conda deactivate
conda activate csi
pip install -r requirements.txt 
```

## MenuConfig

```
cd ~/esp/esp-idf
./install.sh esp32s3
. ./export.sh
```

```
cd ~/esp/esp-csi/examples/get-started/csi_recv_router
idf.py set-target esp32s3
idf.py menuconfig
```
- `set-target esp32s3` : Make sure to set the correct chip target
- `menuconfig` : project configuration to configure Wi-Fi or Ethernet

```
sudo chmod 777 /dev/ttyACM0
# ls -al /dev/ttyACM0
```
- ESP32s3 File Permissions Settings✨


## Building  and flash


```
idf.py build
idf.py flash -b 921600 -p /dev/ttyACM0
```
- build the project and flash it to the board.
- if it may be other computer's ESP, you should build and flash.


## Check!
- please checking path in 426 lines, 183 lines `csi_collec_recog.py` 


# Execute Collecting Tool
```
cd ../tools
python csi_data_read_parse.py -p /dev/ttyACM0 # example file
```

or

```
cd ../tools
python csi_collect_recog.py -p /dev/ttyACM0
python csi_collet_image_compact.py -p /dev/ttyACM0
```


---

## 1. Package Version Issue

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/(...)site-packages/cv2/qt/plugins" even though it was found.
```

- solution

```
pip uninstall opencv-python
pip install opencv-python-headless
```
https://github.com/NVlabs/instant-ngp/discussions/300

## 2. conda installation
```
# based on ubuntu 20.02
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

```

```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

```

```
# 새로운 채널을 추가합니다.
conda config --add channels conda-forge

# 채널의 우선순위를 정해, 높은 우선순위의 채널을 먼저 참조합니다.
conda config --set channel_priority strict

# 채널을 확인합니다.
conda config --show channels
```
https://dongwooblog.tistory.com/202


## 3. requirements_notcuda.txt
this txt is for different cuda version.

pip install torch
pip install trochvision
pip install ultralytics

https://github.com/ultralytics/ultralytics


## 4. esp-csi or esp-idf
https://github.com/espressif/esp-csi
https://github.com/espressif/esp-idf

## 5. download Yolo Model.

