# ========================================================================================================================
# Faceplant Forecast, 2025-2026
# This is the script that must be run to start up all the other code blocks. It must be run whenever the Raspberry Pi
# boots up, and is also necessary if attempting to test integration between systems. This script creates shared memory
# buffers and handles the process of starting the other scripts as daemon processes.
# ========================================================================================================================

import multiprocessing as mp
from multiprocessing import shared_memory as sm
import numpy as np
import serial_connection.command_port as CmdPrt
import serial_connection.data_port as DatPrt
import server as Server
import time
import argparse
import platform
from signal import signal, SIGTERM

#import enums
from enums import BUFF_SIZES, CMD_INDEX, MAIN_STATUS, CMD_PORT_STATUS, AI_STATUS, BOOT_MODE, PLATFORM, APP_CMD

# Supervisory constants for daemon recovery behaviour.
PROCESS_MAX_RESTARTS = 3
PROCESS_MONITOR_SLEEP_SEC = 1.0
PROCESS_HEALTHY_RESET_SEC = 10.0
PROCESS_RESTART_DELAY_SEC = 0.75

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global radar_buffer
global radar_data

def _start_command_process():
    """
    Start the command-port process with a SIGTERM hook for clean radar shutdown.
    """
    signal(SIGTERM, CmdPrt.shutdown) #stops radar on process termination

    CmdPrt.main()

def _start_data_process():
    """
    Start the data process
    """

    DatPrt.main()

def _start_server_process():
    """
    Start the server process
    """

    Server.main()

# process table, keyed by logical process name
PROCESS_SPECS = {
    "server": {"target": _start_server_process, "name": "ServerProc"},
    "command_port": {"target": _start_command_process, "name": "CommandProc"},
    "data_port": {"target": _start_data_process, "name": "DataProc"},
}
PROCESS_STATE = {}

def _create_buffers():
    """
    Set up shared-memory buffers used by all service processes.
    """
    global cmd_buffer
    global cmd_data
    global radar_buffer
    global radar_data

    #================================================================
    #SHARED MEMORY BUFFER EXAMPLE

    #create the buffer, give it a name, set create to True, and give the size in bytes
    cmd_buffer = sm.SharedMemory("cmd_buffer", create=True, size=BUFF_SIZES.CMD_BUFF)
    # Create the data, which is the array that is accessed by this script.
    # By setting the buffer, it can be accessed by other scripts as well
    cmd_data = np.ndarray(  shape=(BUFF_SIZES.CMD_BUFF,),
                            dtype=np.int8,
                            buffer=cmd_buffer.buf)
    #populate the array with data. In this case, zeros.
    cmd_data[:] = np.zeros(BUFF_SIZES.CMD_BUFF) #populate the array with zeros
    #================================================================

    #create the rest of the shared memory buffers
    
    radar_buffer = sm.SharedMemory("radar_buffer", create=True, size=BUFF_SIZES.RADAR_BUFF)
    radar_data = np.ndarray(shape=(BUFF_SIZES.RADAR_LEN,),
                            dtype=np.int64,
                            buffer=radar_buffer.buf)
    radar_data[:] = np.zeros(BUFF_SIZES.RADAR_LEN) #populate the array with zeros

def _set_cmd_defaults():
    """
    Set default command-buffer values for normal boot operation.
    """
    global cmd_data

    cmd_data[CMD_INDEX.FRAMERATE] = 10 #set default framerate to 10
    cmd_data[CMD_INDEX.AI_STATUS] = AI_STATUS.OFFLINE
    cmd_data[CMD_INDEX.MAIN_STATUS] = MAIN_STATUS.RUNNING
    cmd_data[CMD_INDEX.CMD_PORT_STATUS] = CMD_PORT_STATUS.OFFLINE
    cmd_data[CMD_INDEX.APP_CMD] = APP_CMD.NONE

    #check for current platform to change serial port settings
    if platform.node() == "raspberrypi":
        cmd_data[CMD_INDEX.PLATFORM] = PLATFORM.RASPBERRY_PI
    elif platform.node() == "DESKTOP-QL18C7K":
        cmd_data[CMD_INDEX.PLATFORM] = PLATFORM.FRITZ_LAPTOP
    elif platform.node() == "DESKTOP-A8R7298":
        cmd_data[CMD_INDEX.PLATFORM] = PLATFORM.FRITZ_DESKTOP

def send_process_notification(event_type, process_name, message, **details):
    """
    Forward process health events to the cloud service via the server transport helper.
    """
    payload = {"process": process_name, **details}
    try:
        Server.send_system_event(event_type=event_type, message=message, **payload)
    except Exception as err:
        print(f"[BOOTLOADER] Failed to send {event_type} notification: {err}")



def _start_named_process(proc_key):
    """
    Create and start one daemon process from PROCESS_SPECS, then track lifecycle metadata.
    """
    spec = PROCESS_SPECS[proc_key]
    proc = mp.Process(target=spec["target"], daemon=True, name=spec["name"])
    proc.start()

    PROCESS_STATE[proc_key]["proc"] = proc
    PROCESS_STATE[proc_key]["last_start_ts"] = time.time()
    PROCESS_STATE[proc_key]["death_handled"] = False

def _initialise_process_table():
    """
    Prepare per-process supervisor state used by the crash-restart monitor.
    """
    PROCESS_STATE.clear()
    for proc_key in PROCESS_SPECS:
        PROCESS_STATE[proc_key] = {
            "proc": None,
            "restart_count": 0,
            "suspended": False,
            "last_start_ts": 0.0,
            "death_handled": False,
        }

def _handle_server_commands():
    """
    Consume server-driven APP_CMD requests relevant to the bootloader supervisor.
    """
    global cmd_data

    if cmd_data[CMD_INDEX.APP_CMD] == APP_CMD.RESTART_FAILED_DAEMONS:
        print("[BOOTLOADER] Received server request to retry failed daemons.")
        for proc_key, state in PROCESS_STATE.items():
            state["suspended"] = False
            state["restart_count"] = 0

            proc = state["proc"]
            if proc is None or not proc.is_alive():
                _start_named_process(proc_key)

        cmd_data[CMD_INDEX.APP_CMD] = APP_CMD.NONE

def _monitor_and_restart_processes():
    """
    Monitor child daemons and restart crashed children up to retry budget.
    """
    while True:
        _handle_server_commands()

        now = time.time()
        for proc_key, state in PROCESS_STATE.items():
            proc = state["proc"]
            if proc is None:
                continue

            if proc.is_alive():
                if (now - state["last_start_ts"]) >= PROCESS_HEALTHY_RESET_SEC:
                    state["restart_count"] = 0
                state["death_handled"] = False
                continue

            if state["death_handled"]:
                continue

            state["death_handled"] = True
            exit_code = proc.exitcode
            state["restart_count"] += 1

            send_process_notification(
                event_type="process_crashed",
                process_name=proc_key,
                message=f"Process {proc_key} crashed.",
                exit_code=exit_code,
                restart_attempt=state["restart_count"],
            )

            if state["restart_count"] >= PROCESS_MAX_RESTARTS:
                state["suspended"] = True
                send_process_notification(
                    event_type="process_restart_failed",
                    process_name=proc_key,
                    message=(
                        f"Process {proc_key} failed to restart {PROCESS_MAX_RESTARTS} times "
                        "and is now suspended until server retry command."
                    ),
                    exit_code=exit_code,
                    max_restarts=PROCESS_MAX_RESTARTS,
                )
                continue

            if not state["suspended"]:
                time.sleep(PROCESS_RESTART_DELAY_SEC)
                _start_named_process(proc_key)

        time.sleep(PROCESS_MONITOR_SLEEP_SEC)

def _shutdown():
    """
    Stop child processes and release shared-memory buffers.
    """
    print("Shutting down...\n")

    for state in PROCESS_STATE.values():
        proc = state.get("proc")
        if proc is not None and proc.is_alive():
            proc.terminate()

    time.sleep(0.25)

    for state in PROCESS_STATE.values():
        proc = state.get("proc")
        if proc is not None:
            proc.close()

    if cmd_buffer is not None:
        cmd_buffer.close()
        cmd_buffer.unlink()
    if radar_buffer is not None:
        radar_buffer.close()
        radar_buffer.unlink()

    print("Done!")


def main():
    global cmd_buffer
    global cmd_data

    #optional arguments to allow for launching in specific modes
    parser = argparse.ArgumentParser(
        description="Start radar with optional settings"
    )
    parser.add_argument(
        "--demo-visualizer", action="store_true",
        help="Start in demo visualizer mode"
    )
    parser.add_argument(
        "--demo-profiler", action="store_true",
        help="Start in demo profiler mode"
    )
    parser.add_argument(
        "--demo-connection", action="store_true",
        help="Start in connection test mode"
    )
    args = parser.parse_args()

    _create_buffers()
    _set_cmd_defaults()
    _initialise_process_table()

    #launch in the demo visualizer mode
    if args.demo_visualizer:
        print("Launching in DEMO VISUALIZER mode\n")
        cmd_data[CMD_INDEX.BOOT_MODE] = BOOT_MODE.DEMO_VISUALIZER
    elif args.demo_profiler:
        print("Launching in DEMO PROFILER mode\n")
        cmd_data[CMD_INDEX.BOOT_MODE] = BOOT_MODE.DEMO_DROPPED_FRAMES
    elif args.demo_connection:
        print("Launcing in CONNECTION TEST mode\n")
        cmd_data[CMD_INDEX.BOOT_MODE] = BOOT_MODE.DEMO_CONNECTION_TEST

    #start processes in expected order: server -> command port -> data port.
    _start_named_process("server")
    _start_named_process("command_port")
    
    #wait until command port is done sending config to start the data port
    while True:
        if cmd_data[CMD_INDEX.CMD_PORT_STATUS] == CMD_PORT_STATUS.ONLINE:
            break
        time.sleep(0.1)
    _start_named_process("data_port")

    _monitor_and_restart_processes()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[EXIT] User stopped.")
    finally:
        _shutdown()
        print("Radar stopped cleanly.")