from multiprocessing import shared_memory as sm
import numpy as np
from enums import BUFF_SIZES, CMD_INDEX, PACKET_DATA, DEMO_VIS_DATA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from serial_connection.data_port import parse_guy

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data

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
    cmd_data = np.ndarray(shape=(BUFF_SIZES.CMD_BUFF,),
                        dtype=np.int8,
                        buffer=cmd_buffer.buf)

# -------------------------
# Wrapper: make Nx7 array for visualizer
# -------------------------
def get_next_frame(con):
    """
    Copy of the parsing code from data_port
    """

    local_frame_buffer = bytearray()
    output = [[],[]]

    #read any available data
    data = con.read(4096)
    if not data:
        return

    #append to buffer
    local_frame_buffer.extend(data)

    #parse frame
    while True:
        parsed_data = parse_guy(local_frame_buffer, len(local_frame_buffer))
        result = parsed_data[PACKET_DATA.RESULT] #TC_PASS or TC_FAIL
        num_bytes = parsed_data[PACKET_DATA.NUM_BYTES]

        if result == 0 and num_bytes > 0:
            #frame parse was successful
            det_v = parsed_data[PACKET_DATA.DET_V]
            det_range = parsed_data[PACKET_DATA.RANGE]

            #place data into the output array
            output[DEMO_VIS_DATA.RANGE].extend(det_range)
            output[DEMO_VIS_DATA.DOPPLER_V].extend(det_v)

            #remove frame from local buffer
            local_frame_buffer = local_frame_buffer[num_bytes:]
        
        else:
            #not a full frame
            break
    
    #return data
    return output

# -------------------------
# Live visualizer
# -------------------------
def live_visualizer(con):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim(0, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler V (m/s)")
    ax.set_title(f"Live Radar Doppler-Range Plot")

    last_pts = None  # last non-empty filtered frame
    det_range = None
    det_v = None

    def update(_):
        nonlocal last_pts
        nonlocal det_range
        nonlocal det_v

        pts = get_next_frame(con)
        if pts is not None and len(pts[DEMO_VIS_DATA.RANGE]) > 0:
            det_range = pts[DEMO_VIS_DATA.RANGE]
            det_v     = pts[DEMO_VIS_DATA.DOPPLER_V]
            last_pts = pts

        ax.cla()
        ax.set_xlim(0, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Doppler V (m/s)")

        if last_pts is None:
            ax.set_title("Waiting for data...")
            scat = ax.scatter([], [], s=1)
            return scat,

        ax.set_title(
            f"Live Radar Doppler-Range Plot"
        )

        scat = ax.scatter(det_range, det_v, s=8)
        return scat,

    ani = FuncAnimation(
        fig,
        update,
        interval=100,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()

def main():
    bootstrapper()