from multiprocessing import shared_memory as sm
import numpy as np
from enums import BUFF_SIZES, CMD_INDEX

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

def main():
    bootstrapper()
    print(cmd_data)