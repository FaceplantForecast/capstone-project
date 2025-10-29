# ==================================================================================
# This code handles collecting data from the data port and parsing it
# ==================================================================================

from parser_mmw_demo import parser_one_mmw_demo_output_packet as parse_guy
import sys
import os
import serial
import time

#set parent directory so enums can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from enums import PACKET_DATA

def stream_frames(con):
    frame_buffer = bytearray()

    while True:
        #read any available data
        data = con.read(4096)
        if not data:
            continue

        #append to buffer
        frame_buffer.extend(data)

        #parse frame
        while True:
            parsed_data = parse_guy(frame_buffer, len(frame_buffer))
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

                print(f"received frame with {num_det_obj} objects")
                for guy in range(num_det_obj):
                    print(f"    Obj {guy+1}: x={det_x[guy]:.2f}, y={det_y[guy]:.2f}, z={det_z[guy]:.2f}, v={det_v[guy]:.2f}, range={det_range[guy]:.2f}")

                #remove frame from buffer
                frame_buffer = frame_buffer[num_bytes:]
            
            else:
                #not a full frame
                break

        #delay to not consume more resources than necessary
        time.sleep(0.1)

def main():
    #Adjust device names and baud rates (deployment on Raspberry Pi)
    #data_port = serial.Serial('/dev/ttyUSB1', 3125000)   # for CLI commands

    #debugging on laptop
    #data_port = serial.Serial('COM3', 3125000)   # for CLI commands

    #debugging on desktop
    data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for CLI commands

    #stream the frames
    stream_frames(data_port)

if __name__ == "__main__":
    sys.exit(main())