# ==================================================================================
# Faceplant Forecast, 2025-2026
# This code handles collecting data from the data port and parsing it.
# It is also responsible for running the ML model and determining if a fall has
# occured or not.
# ==================================================================================

from serial_connection.parser_mmw_demo import parser_one_mmw_demo_output_packet as parse_guy
import sys
import os
import serial
import time
import numpy as np
import multiprocessing.shared_memory as sm

#set parent directory so enums can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from enums import PACKET_DATA, DEBUG_LEVEL as DEBUG, BUFF_SIZES, CMD_INDEX, DAT_PORT_STATUS, BOOT_MODE

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global radar_buffer
global radar_data

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
    
def live_visualizer():
    from test_process import live_visualizer as demo_vis

    #debugging on desktop
    #data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming

    #debugging on laptop
    data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming

    demo_vis(data_port)

def check_dropped_frames():
    #debugging on desktop
    #data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming

    #debugging on laptop
    data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming

    stream_frames(data_port, mode=BOOT_MODE.DEMO_DROPPED_FRAMES)

def stream_frames(con, debug=DEBUG.NONE, mode=BOOT_MODE.STANDARD):
    local_frame_buffer = bytearray()

    #create profiling variables
    run_time_sec = 7200
    last_frame = 0
    dropped_frames = 0
    frame_count = 0
    start_time = time.time()

    while True:
        #read any available data
        data = con.read(4096)
        if not data:
            continue

        #append to buffer
        local_frame_buffer.extend(data)

        #parse frame
        while True:
            dropped_frame = False
            parsed_data = parse_guy(local_frame_buffer, len(local_frame_buffer))
            result = parsed_data[PACKET_DATA.RESULT] #TC_PASS or TC_FAIL
            num_bytes = parsed_data[PACKET_DATA.NUM_BYTES]

            if result == 0 and num_bytes > 0:
                #frame parse was successful
                num_det_obj = parsed_data[PACKET_DATA.NUM_DET_OBJ]
                det_x = parsed_data[PACKET_DATA.DET_X]
                det_y = parsed_data[PACKET_DATA.DET_Y]
                det_z = parsed_data[PACKET_DATA.DET_Z]
                det_v = parsed_data[PACKET_DATA.DET_V]
                det_range = parsed_data[PACKET_DATA.RANGE]
                frame_num = parsed_data[PACKET_DATA.FRAME_NUM]

                #check if frame was dropped
                if last_frame == 0 or last_frame > frame_num:
                    last_frame = frame_num
                elif last_frame == frame_num - 1:
                    last_frame += 1
                else:
                    dropped_frames += 1
                    last_frame = frame_num
                    dropped_frame = True

                #print info to console if debug mode is set
                if debug != DEBUG.NONE:
                    print(f"received frame number {frame_num} with {num_det_obj} objects and length {num_bytes} bytes ({dropped_frames} dropped frames)")
                    if debug == DEBUG.VERBOSE:
                        for guy in range(num_det_obj):
                            print(f"    Obj {guy+1}: x={det_x[guy]:.2f}, y={det_y[guy]:.2f}, z={det_z[guy]:.2f}, v={det_v[guy]:.2f}, range={det_range[guy]:.2f}")

                #remove frame from local buffer
                local_frame_buffer = local_frame_buffer[num_bytes:]
            
                #increase frame count
                frame_count += 1
            
            else:
                #not a full frame
                break

        #check if enough time has passed to end profiling
        elapsed_time = time.time() - start_time
        if mode == BOOT_MODE.DEMO_DROPPED_FRAMES:
            if (elapsed_time >= run_time_sec):
                print(f"ELAPSED TIME (SEC): {elapsed_time}\nFRAMES PROCESSED: {frame_count}\nDROPPED FRAMES: {dropped_frames}")
                print(f"DROPPED FRAMES PER MIN: {dropped_frames / (elapsed_time/60)}")
                print(f"DROPPED FRAMES PER HOUR: {dropped_frames / (elapsed_time/3600)}")
                print(f"PERCENTAGE OF FRAMES DROPPED: {(dropped_frames / (dropped_frames + frame_count)) * 100}%")
                break
            elif dropped_frame == True:
                print(f"DROPPED FRAME AT RUN TIME (SEC): {elapsed_time}\n")

        #INSERT ML MODEL CODE HERE (courtesy of Charles Marks [they called him Mr. Machine Learning back in college])
        

        #delay to not consume more resources than necessary
        time.sleep(0.1)

def main():
    global cmd_data
    
    #initiate bootstrapper
    bootstrapper()

    cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.CONNECTING

    #launch in demo mode if needed
    if cmd_data[CMD_INDEX.BOOT_MODE] == BOOT_MODE.DEMO_VISUALIZER:
        live_visualizer()
    elif cmd_data[CMD_INDEX.BOOT_MODE] == BOOT_MODE.DEMO_DROPPED_FRAMES:
        check_dropped_frames()
    elif cmd_data[CMD_INDEX.BOOT_MODE] == BOOT_MODE.DEMO_CONNECTION_TEST:
        print("Not connecting to radar")
    else:
        try:
            #Adjust device names and baud rates (deployment on Raspberry Pi)
            #data_port = serial.Serial('/dev/ttyUSB1', 3125000, timeout=0.1)   # for data streaming

            #debugging on laptop
            data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming

            #debugging on desktop
            #data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming

            cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.RUNNING
        except:
            cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.ERROR

        #stream the frames
        stream_frames(data_port, DEBUG.VERBOSE)

if __name__ == "__main__":
    sys.exit(main())