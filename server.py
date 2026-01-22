# ==================================================================================
# Faceplant Forecast, 2026
# This code handles the server for connecting to the app. It is responsible for
# sending data and receiving commands
# ==================================================================================

import numpy as np
import multiprocessing.shared_memory as sm
import time
from enums import BUFF_SIZES
import websockets

#global variables
global cmd_buffer
global cmd_data
global radar_buffer
global radar_data

# ===============================================================
# SERVER SETUP
# GCP endpoint + auth token (WebSocket)
BASE_GCP_URL = "gcr-ws-482782751069.us-central1.run.app/ws"
# Original token you gave earlier (without the stray dash)
AUTH_TOKEN = "M8b4eFJHq2pI3V9nW5r0dE-PLZpQyX7uB1cTa9kN4mE"

# Full WSS URL with auth_token as query param
GCP_WSS_URL = f"wss://{BASE_GCP_URL}?role=pi&token={AUTH_TOKEN}"
# ===============================================================

def bootstrapper():
    """
    This function handles the startup sequence for the process
    """
    global cmd_buffer
    global cmd_data
    global radar_buffer
    global radar_data

    #create the buffer, give it a name, set create to False, and give the size in bytes
    cmd_buffer = sm.SharedMemory("cmd_buffer", create=False)
    # Create the data, which is the array that is accessed by this script.
    # By setting the buffer, it can be accessed by other scripts as well
    cmd_data = np.ndarray(  shape=(BUFF_SIZES.CMD_BUFF,),
                            dtype=np.int8,
                            buffer=cmd_buffer.buf)
    
    radar_buffer = sm.SharedMemory("radar_buffer", create=False)
    radar_data = np.ndarray(shape=(BUFF_SIZES.RADAR_BUFF,),
                            dtype=np.int8,
                            buffer=radar_buffer.buf)



def main():
    bootstrapper()
    print("FILLER BAYBEEEEEEEEEE\n")