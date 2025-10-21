# ========================================================================================================================
# Faceplant Forecast, 2025
# This is the script that must be run to start up all the other code blocks. It must be run whenever the Raspberry Pi
# boots up, and is also necessary if attempting to test integration between systems. This script creates shared memory
# buffers and handles the process of starting the other scripts as daemon processes.
# ========================================================================================================================

import multiprocessing as mp
from multiprocessing import shared_memory as sm
import numpy as np
import serial_connection.command_port as CmdPrt
import test_process as TestProc
import sys

#import enums
from enums import BUFF_SIZES, CMD_INDEX

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data

#processes
global test_proc

def create_buffers():
    """
    This function handles setting up all the buffers
    """
    global cmd_buffer
    global cmd_data

    #================================================================
    #SHARED MEMORY BUFFER EXAMPLE

    #create the buffer, give it a name, set create to True, and give the size in bytes
    cmd_buffer = sm.SharedMemory("cmd_buffer", create=True, size=BUFF_SIZES.CMD_BUFF)
    # Create the data, which is the array that is accessed by this script.
    # By setting the buffer, it can be accessed by other scripts as well
    cmd_data = np.ndarray(shape=(BUFF_SIZES.CMD_BUFF,),
                        dtype=np.int8,
                        buffer=cmd_buffer.buf)
    #populate the array with data. In this case, zeros.
    cmd_data[:] = np.zeros(BUFF_SIZES.CMD_BUFF) #populate the array with zeros
    #================================================================

def set_cmd_defaults():
    """
    This function sets the default values for the command buffer.
    It should only run on startup or on a system reset.
    """
    global cmd_data

    cmd_data[CMD_INDEX.FRAMERATE] = 10 #set default framerate to 10

def start_test_process():
    """
    This function starts the test process. A wrapper function is needed
    to allow calling from another script.
    """

    TestProc.main()

def shutdown():
    """
    This function handles shutting down the system. That means killing
    daemon processes and closing/unlinking shared memory pools
    """
    global cmd_buffer
    global test_proc

    #close processes first
    test_proc.close()

    #close and unlink shared memory buffers
    cmd_buffer.close()
    cmd_buffer.unlink()


def main():
    global cmd_buffer
    global cmd_data
    global test_proc
    create_buffers()
    set_cmd_defaults()

    #create the daemon process and target the wrapper function
    test_proc = mp.Process(target=start_test_process,
                           daemon=True,
                           name="TestProc")
    #start the daemon process
    test_proc.start()

    CmdPrt.main()

    input("press Enter to end...\n")
    shutdown()

if __name__ == "__main__":
    sys.exit(main())