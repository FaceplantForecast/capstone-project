# ==================================================================================
# Faceplant Forecast, 2026
# This code handles the server for connecting to the app. It is responsible for
# sending data and receiving commands
# ==================================================================================

import numpy as np
import multiprocessing.shared_memory as sm
import time
from enums import APP_CMD, BUFF_SIZES, CMD_INDEX, RADAR_DATA
import websockets
import asyncio
import json
import threading

#global variables
global cmd_buffer
global cmd_data
global radar_buffer
global radar_data

#configuration
DEBUG = False
POLL_SLEEP_SEC = 0.1
RECONNECT_SLEEP_SEC = 2.0

#===============================SERVER SETUP===============================
BASE_GCP_URL = "gcr-ws-482782751069.us-central1.run.app/ws"         #GCP endpoint + auth token (WebSocket)
AUTH_TOKEN = "M8b4eFJHq2pI3V9nW5r0dE-PLZpQyX7uB1cTa9kN4mE"          #Original token
GCP_WSS_URL = f"wss://{BASE_GCP_URL}?role=pi&token={AUTH_TOKEN}"    #Full WSS URL with auth_token as query param
#===============================================================

def _bootstrapper():
    """
    Handle the startup sequence for the process.
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

#===============================MESSAGING===============================
async def _send_message_ws(msg: dict):
    """
    Open WebSocket, send one JSON message, then close.
    """
    async with websockets.connect(GCP_WSS_URL) as ws:
        await ws.send(json.dumps(msg))

#TODO: change device and account ID away from hard-coded values
def _build_base_message(msg_type: str, payload: dict):
    """
    Build message into standard format. Valid message types are:
    fall_event -> fall flag
    system_event -> runtime information, like process states or failures
    """
    return {
        "msg_type": msg_type,
        "ts_send": time.strftime("%m-%d-%Y %H:%M:%S", time.localtime()),
        "device_id": "deployed-pi-01",
        "account_id": "account-1",
        "payload": payload,
    }

def send_system_event(event_type: str, message: str, **details):
    """
    Public helper used by bootloader to report daemon health events.
    """
    payload = {"event_type": event_type, "message": message, **details}
    msg = _build_base_message("system_event", payload)
    try:
        asyncio.run(_send_message_ws(msg))
        if DEBUG:
            print(f"[GCP] Sent system event: {event_type}")
    except Exception as err:
        print(f"[GCP] Failed to send system event {event_type}: {err}")

def send_fall_flag(probability: float, frame_id: int, ts: float):
    """
    Send one fall-detection event to cloud backend.
    """
    payload = {
        "fall_detected": 1,
        "probability": float(probability),
        "frame_id": int(frame_id),
        "ts_fall": time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(ts)),
    }
    msg = _build_base_message("fall_event", payload)

    try:
        asyncio.run(_send_message_ws(msg))
        if DEBUG:
            print("[GCP] Sent fall flag payload")
    except Exception as err:
        print(f"[GCP] Failed to send fall flag via WSS: {err}")

def _apply_server_command(raw_msg):
    """
    Parse incoming command JSON and map known commands onto APP_CMD flags.
    """
    global cmd_data

    try:
        data = json.loads(raw_msg)
    except json.JSONDecodeError:
        if DEBUG:
            print(f"[GCP] Non-JSON command ignored: {raw_msg}")
        return

    command = data.get("command")
    if not isinstance(command, str):
        return

    command = command.lower().strip()

    if command == "redo_background_scan":
        cmd_data[CMD_INDEX.APP_CMD] = APP_CMD.REDO_BACKGROUND_SCAN
        print("[SERVER] Queued command: REDO_BACKGROUND_SCAN")
    elif command in ("restart_failed_daemons", "retry_failed_daemons"):
        cmd_data[CMD_INDEX.APP_CMD] = APP_CMD.RESTART_FAILED_DAEMONS
        print("[SERVER] Queued command: RESTART_FAILED_DAEMONS")

async def _command_listener_loop():
    """
    Maintain a receive loop for server-to-pi commands over websocket.
    """
    while True:
        try:
            async with websockets.connect(GCP_WSS_URL) as ws:
                register_msg = _build_base_message("register", {"role": "pi"})
                await ws.send(json.dumps(register_msg))

                async for message in ws:
                    _apply_server_command(message)
        except Exception as err:
            print(f"[GCP] Command listener disconnected: {err}")
            await asyncio.sleep(RECONNECT_SLEEP_SEC)

def _start_command_listener_thread():
    """
    Run async command listener in a dedicated daemon thread.
    """
    thread = threading.Thread(
        target=lambda: asyncio.run(_command_listener_loop()),
        daemon=True,
        name="ServerCommandListener",
    )
    thread.start()
#===============================================================

def _control_loop():
    """
    Poll shared memory for fall flags and forward events once per trigger.
    """
    global radar_data

    fall_latched = 0

    while True:
        if radar_data[RADAR_DATA.FALL_DETECTED] == 1:
            #Only send once (first detection)
            if fall_latched == 0:
                send_fall_flag(probability=float(radar_data[RADAR_DATA.PROBABILITY])/100.0,
                               frame_id=radar_data[RADAR_DATA.FRAME_ID],
                               ts=radar_data[RADAR_DATA.TIMESTAMP])
            fall_latched = 1
            radar_data[RADAR_DATA.FALL_DETECTED] = 0 #reset buffer flag
        else:
            fall_latched = 0

        time.sleep(POLL_SLEEP_SEC)

def main():
    _bootstrapper()
    _start_command_listener_thread()
    _control_loop()

if __name__ == "__main__":
    main()