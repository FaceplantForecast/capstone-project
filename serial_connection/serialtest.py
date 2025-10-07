import serial
import time
import sys

# Adjust device names and baud rates
#config_port = serial.Serial('/dev/ttyUSB0', 115200)   # for CLI commands
#data_port   = serial.Serial('/dev/ttyUSB1', 115200)   # for binary point cloud data

#debugging on windows
config_port = serial.Serial('COM6', 115200)   # for CLI commands

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

def CLIController():
    while True:
        user_input = input("command:") + "\r\n"

        #exit loop
        if user_input.lower() == "exit":
            break
        
        #send command
        config_port.write(user_input.encode('utf-8'))
        time.sleep(0.1)

        #read once to get to the "meat"
        data = config_port.readline()

        #read until the terminal goes back to ready state
        while data.decode() != "R5F0> \n":
            if user_input.replace('\r\n', '') not in data.decode():
                print(data.decode().replace('\n', ''))
            data = config_port.readline()
        
        #read one more line to prep for next cycle
        data = config_port.readline()

def main():
    CLIController()