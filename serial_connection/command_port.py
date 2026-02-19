# ==================================================================================
# Faceplant Forecast, 2025-2026
# This code handles connecting to the command port and controlling the radar either
# with a CLI for debugging or by other python scripts
# ==================================================================================

import serial
import time
import os
import sys
import numpy as np
import multiprocessing.shared_memory as sm

#set parent directory so enums can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from enums import PACKET_DATA, DEBUG_LEVEL as DEBUG, BUFF_SIZES, CMD_INDEX, BOOT_MODE, PLATFORM

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global config_port


def bootstrapper():
    """
    This function handles the startup sequence for the process
    """
    global cmd_buffer
    global cmd_data

    #create the buffer, give it a name, set create to False, and give the size in bytes
    cmd_buffer = sm.SharedMemory("cmd_buffer", create=False)
    # Create the data, which is the array that is accessed by this script.
    # By setting the buffer, it can be accessed by other scripts as well
    cmd_data = np.ndarray(  shape=(BUFF_SIZES.CMD_BUFF,),
                            dtype=np.int8,
                            buffer=cmd_buffer.buf)

def InitiateRadar():
    # Send a config file line by line
    with open('radar_profile.cfg', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('%'):  # skip comments
                config_port.write((line.strip() + '\n').encode())
                time.sleep(0.01)  # small delay between commands

    print("Configuration sent. Radar should be running.")

def send_cfg(ser, cfg_file):


    if not os.path.exists(cfg_file):
        raise FileNotFoundError(cfg_file)

    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('%')]

    print(f"[CFG] Sending {len(lines)} commands from {cfg_file}...")
    for line in lines:
        ser.write((line + "\n").encode())
        ser.flush()
        time.sleep(0.05)
        resp = ser.read(ser.in_waiting or 1)
        if resp:
            try:
                txt = resp.decode(errors="ignore").strip()
                if txt:
                    print(f"[CFG RESP] {txt}")
            except Exception:
                pass
    ser.close()
    print("[CFG] Radar started.")

def CLIController(user_input):
        #send command
        config_port.write(user_input.encode('utf-8'))
        time.sleep(0.1)

        #read twice to get to the "meat"
        data = config_port.readline()
        print(data.decode())
        data = config_port.readline()

        #read until the terminal goes back to ready state
        output = ""
        i = 0
        while data.decode() != "mmwDemo:/>\n":
            if user_input.replace('\r\n', '') not in data.decode():
                output = output + data.decode()
            data = config_port.readline()
            i += 1
            if i > 10:
                print("ERROR")
                break
        
        #read one more line to prep for next cycle
        config_port.readline()

        return output

def UserCLI():
    global config_port

    if 'config_port' in globals(): #make sure to only run this if the port is actually defined
        while True:
            user_input = input("command:") + "\r\n"

            #exit loop
            if user_input.lower() == "exit\r\n":
                break
            
            #call CLI Controller function
            print(CLIController(user_input))
        

def main():
    global config_port

    bootstrapper()

    #don't start connection in the server test
    if cmd_data[CMD_INDEX.BOOT_MODE] != BOOT_MODE.DEMO_CONNECTION_TEST:

        match cmd_data[CMD_INDEX.PLATFORM]:
            case PLATFORM.RASPBERRY_PI:
                #Adjust device names and baud rates (deployment on Raspberry Pi)
                try:
                    config_port = serial.Serial('/dev/ttyACM0', 115200)   # for CLI commands
                except:
                    config_port = serial.Serial('/dev/ttyUSB0', 115200)   #sometimes switches, I don't know why
            case PLATFORM.FRITZ_LAPTOP:
                #debugging on laptop
                config_port = serial.Serial('COM6', 115200)   # for CLI commands
            case PLATFORM.FRITZ_DESKTOP:
                #debugging on desktop
                config_port = serial.Serial('COM3', 115200)   # for CLI commands
            case _:
                #default to raspberry pi
                try:
                    config_port = serial.Serial('/dev/ttyACM0', 115200)   # for CLI commands
                except:
                    config_port = serial.Serial('/dev/ttyUSB0', 115200)   #sometimes switches, I don't know why

        #call bootstrapper and initiate the radar
        send_cfg(config_port, 'radar_profile.cfg')
