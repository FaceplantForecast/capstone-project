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
import serial_connection.data_port as DatPrt
import test_process as TestProc
import sys, time

#import enums
from enums import BUFF_SIZES, CMD_INDEX, MAIN_STATUS, CMD_PORT_STATUS, AI_STATUS

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global frame_buffer
global frame_data

#processes
global test_proc
global command_proc
global data_proc

def create_buffers():
    """
    This function handles setting up all the buffers
    """
    global cmd_buffer
    global cmd_data
    global frame_buffer
    global frame_data

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

    #create the rest of the shared memory buffers
    #TODO: figure out how to handle the frame buffer

def set_cmd_defaults():
    """
    This function sets the default values for the command buffer.
    It should only run on startup or on a system reset.
    """
    global cmd_data

    cmd_data[CMD_INDEX.FRAMERATE] = 10 #set default framerate to 10
    cmd_data[CMD_INDEX.AI_STATUS] = AI_STATUS.OFFLINE
    cmd_data[CMD_INDEX.MAIN_STATUS] = MAIN_STATUS.RUNNING
    cmd_data[CMD_INDEX.CMD_PORT_STATUS] = CMD_PORT_STATUS.OFFLINE

def start_test_process():
    """
    This function starts the test process. A wrapper function is needed
    to allow calling from another script.
    """

    TestProc.main()

def start_command_process():
    """
    This function starts the data port process. A wrapper function is
    needed to allow calling from another script.
    """

    CmdPrt.main()

def start_data_process():
    """
    This function starts the data port process. A wrapper function is
    needed to allow calling from another script.
    """

    DatPrt.main()

def shutdown():
    """
    This function handles shutting down the system. That means killing
    daemon processes and closing/unlinking shared memory pools
    """
    global cmd_buffer
    global test_proc
    global command_proc
    global data_proc

    #terminate processed first
    #test_proc.terminate()
    command_proc.terminate()
    data_proc.terminate()

    time.sleep(0.25) #delay to give data_proc time to terminate

    #then close processes
    #test_proc.close()
    command_proc.close()
    data_proc.close()

    #close and unlink shared memory buffers
    cmd_buffer.close()
    cmd_buffer.unlink()


def main():
    global cmd_buffer
    global cmd_data
    global test_proc
    global command_proc
    global data_proc

    create_buffers()
    set_cmd_defaults()

    #create the daemon process and target the wrapper function
    """
    test_proc = mp.Process(target=start_test_process,
                           daemon=True,
                           name="TestProc")
    #start the daemon process
    test_proc.start()
    """
    #create and start daemon processes for all components
    command_proc = mp.Process(target=start_command_process,
                              daemon=True,
                              name="CommandProc")
    command_proc.start()

    data_proc = mp.Process(target=start_data_process,
                           daemon=True,
                           name="DataProc")
    data_proc.start()

    input("press Enter to end...\n")

    shutdown()

if __name__ == "__main__":
    sys.exit(main())