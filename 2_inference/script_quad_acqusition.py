import subprocess
import keyboard
import time

script_path = "/home/keti/esp/esp-csi/examples/get-started/tools/csi_inf_conv_2025_single.py"

processes = []


for i in range(4):
    if i == 2:
        p = subprocess.Popen(["python", script_path, "-p", f"/dev/ttyACM{i}", "-a", "-x", "750", "-y", "800"])
    elif i == 3:
        p = subprocess.Popen(["python", script_path, "-p", f"/dev/ttyACM{i}", "-a","-y", "800"])        
    elif i == 0:
        p = subprocess.Popen(["python", script_path, "-p", f"/dev/ttyACM{i}", "-a","-x", "750"])
    elif i == 1:
        p = subprocess.Popen(["python", script_path, "-p", f"/dev/ttyACM{i}", "-a",])
    processes.append(p)


