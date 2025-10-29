# ==================================================================================
# This code handles collecting data from the data port and parsing it
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

from enums import PACKET_DATA, DEBUG_LEVEL as DEBUG, BUFF_SIZES, CMD_INDEX, DAT_PORT_STATUS

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global frame_buffer
global frame_data

def bootstrapper():
    """
    This function handles the startup sequence for the process
    """
    global cmd_buffer
    global cmd_data
    global frame_buffer
    global frame_data

    #create the buffer, give it a name, set create to False, and give the size in bytes
    cmd_buffer = sm.SharedMemory("cmd_buffer", create=False)
    # Create the data, which is the array that is accessed by this script.
    # By setting the buffer, it can be accessed by other scripts as well
    cmd_data = np.ndarray(shape=(BUFF_SIZES.CMD_BUFF,),
                        dtype=np.int8,
                        buffer=cmd_buffer.buf)
    
    #TODO: connect to frame buffer

def stream_frames(con, debug=DEBUG.NONE):
    global frame_data

    local_frame_buffer = bytearray()

    while True:
        #read any available data
        data = con.read(4096)
        if not data:
            continue

        #append to buffer
        local_frame_buffer.extend(data)

        #parse frame
        while True:
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

                #print info to console if debug mode is set
                if debug != DEBUG.NONE:
                    print(f"received frame with {num_det_obj} objects and length {num_bytes} bytes")
                    if debug == DEBUG.VERBOSE:
                        for guy in range(num_det_obj):
                            print(f"    Obj {guy+1}: x={det_x[guy]:.2f}, y={det_y[guy]:.2f}, z={det_z[guy]:.2f}, v={det_v[guy]:.2f}, range={det_range[guy]:.2f}")

                #remove frame from local buffer
                local_frame_buffer = local_frame_buffer[num_bytes:]
            
            else:
                #not a full frame
                break

        #delay to not consume more resources than necessary
        time.sleep(0.25)

def main():
    global cmd_data
    
    #initiate bootstrapper
    bootstrapper()

    cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.CONNECTING

    try:
        #Adjust device names and baud rates (deployment on Raspberry Pi)
        #data_port = serial.Serial('/dev/ttyUSB1', 3125000)   # for data streaming

        #debugging on laptop
        #data_port = serial.Serial('COM3', 3125000)   # for data streaming

        #debugging on desktop
        data_port = serial.Serial('COM4', 3125000, timeout=0.25)   # for data streaming

        cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.RUNNING
    except:
        cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.ERROR

    #stream the frames
    stream_frames(data_port, DEBUG.VERBOSE)

if __name__ == "__main__":
    sys.exit(main())