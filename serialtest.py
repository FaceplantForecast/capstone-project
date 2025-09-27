import serial
import time
import sys

# Adjust device names and baud rates
config_port = serial.Serial('/dev/ttyUSB0', 115200)   # for CLI commands
data_port   = serial.Serial('/dev/ttyUSB1', 3125000)   # for binary point cloud data

def InitiateRadar():
    # Send a config file line by line
    with open('profile_2025_09_26T22_19_26_970.cfg', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('%'):  # skip comments
                config_port.write((line.strip() + '\n').encode())
                time.sleep(0.01)  # small delay between commands

    print("Configuration sent. Radar should be running.")

def ReadRadarData():
    while True:
        data = data_port.read(1024)  # read 1024 bytes
        print(data)  # raw binary stream
