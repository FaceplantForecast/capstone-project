# ==================================================================================
# Faceplant Forecast, 2026
# This code handles the server for connecting to the app. It is responsible for
# sending data and receiving commands
# ==================================================================================

import numpy as np
import multiprocessing.shared_memory as sm
import time
from enums import BUFF_SIZES, RADAR_DATA
import websockets
import asyncio
import json

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
    radar_data = np.ndarray(shape=(BUFF_SIZES.RADAR_LEN,),
                            dtype=np.int64,
                            buffer=radar_buffer.buf)

def control_loop():
    global cmd_data
    global radar_data

    FALL_DETECTED = 0

    while True:
        if radar_data[RADAR_DATA.FALL_DETECTED] == 1:
            #Only send once (first detection)
            if FALL_DETECTED == 0:
                print("Sending notification to server")
                send_fall_flag(probability=float(radar_data[RADAR_DATA.PROBABILITY])/100.0,
                               frame_id=radar_data[RADAR_DATA.FRAME_ID],
                               ts=radar_data[RADAR_DATA.TIMESTAMP])
            FALL_DETECTED = 1
        else:
            FALL_DETECTED = 0
def send_fall_flag(probability: float, frame_id: int, ts: float):
    """
    Public function used in live_loop.
    Wraps the async WebSocket sender in asyncio.run.
    """
    try:
        asyncio.run(_send_fall_flag_ws(probability, frame_id, ts))
    except RuntimeError as e:
        # In case there's already a running event loop (unlikely in this script),
        # you could adapt this to use create_task instead.
        print(f"[GCP] asyncio runtime error: {e}")

async def _send_fall_flag_ws(probability: float, frame_id: int, ts: float):
    """
    Internal coroutine: open WSS, send one JSON message, close.
    Includes auth_token both in query string and in the JSON body.
    """
    msg = {"fall_detected": 1, "ts": time.strftime("%m-%d-%Y %H:%M:%S", time.localtime())}

    print(f"[GCP] Connecting to {GCP_WSS_URL} ...")
    try:
        async with websockets.connect(GCP_WSS_URL) as ws:
            # Send payload
            await ws.send(json.dumps(msg))
            print("[GCP] Sent fall flag payload")

            # Optional: try to read a response (if server replies)
            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=3.0)
                print(f"[GCP] Received reply: {reply}")
            except asyncio.TimeoutError:
                print("[GCP] No reply from server (timeout), assuming fire-and-forget.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[GCP] WebSocket closed: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"[GCP] Failed to send fall flag via WSS: {e}")

def main():
    bootstrapper()
    control_loop()