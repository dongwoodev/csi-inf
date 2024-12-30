# mqtt_config.py

import json
from datetime import datetime

# MQTT 설정
BROKER_ADDRESS = "localhost"
PORT = 1883
TOPIC = "test/csi"

# 패킷 ID 관리
packet_id = -1

def get_next_packet_id():
    global packet_id
    packet_id += 1
    if packet_id > 50000:
        packet_id = 0
    return packet_id

def create_mqtt_message(occ=None, occ_score=None, loc=None, loc_score=None, act=None, act_score=None, timestamp=None):
    message = {
        "packet_id": get_next_packet_id(),
        "timestamp": timestamp,
        "occupancy": occ,
        "occ_conf" : occ_score,
        "loc" : loc,
        "loc_conf" : loc_score,
        "act" : act,
        "act_conf": act_score
    }
    return json.dumps(message)
