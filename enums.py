# ========================================================================
# Faceplant Forecast, 2025-2026
# This is a set of global enums that allow for better readability and more
# efficient changing of the code. This needs to be accessed by everything.
# ========================================================================

from enum import IntEnum

class DEBUG_LEVEL(IntEnum):
    NONE    = 0
    INFO    = 1
    VERBOSE = 2

class BUFF_SIZES(IntEnum):
    CMD_BUFF    = 8 #in bytes
    FRAME_BUFF  = 4096
    RADAR_BUFF  = 1

class CMD_INDEX(IntEnum):
    MAIN_STATUS         = 0
    CMD_PORT_STATUS     = 1
    AI_STATUS           = 2
    DAT_PORT_STATUS     = 3
    SERVER_STATUS       = 4
    FRAMERATE           = 5
    APP_CMD             = 6
    BOOT_MODE           = 7

class MAIN_STATUS(IntEnum):
    RUNNING     = 0
    ERROR       = 1

class CMD_PORT_STATUS(IntEnum):
    ONLINE      = 0
    ERROR       = 1
    OFFLINE     = 2
    CONNECTING  = 3
    SENDING     = 4
    RECEIVING   = 5

class AI_STATUS(IntEnum):
    RUNNING     = 0 
    ERROR       = 1
    OFFLINE     = 2
    NOTICE      = 3
    PAUSED      = 4

class DAT_PORT_STATUS(IntEnum):
    RUNNING     = 0
    ERROR       = 1
    OFFLINE     = 2
    CONNECTING  = 3

class APP_STATUS(IntEnum):
    RUNNING     = 0
    ERROR       = 1
    OFFLINE     = 2
    NOTICE      = 3
    PAUSED      = 4

class PACKET_DATA(IntEnum):
    RESULT              = 0
    HEADER_START_IDX    = 1
    NUM_BYTES           = 2
    NUM_DET_OBJ         = 3
    NUM_TLV             = 4
    SUB_FRAME_NUM       = 5
    DET_X               = 6
    DET_Y               = 7
    DET_Z               = 8
    DET_V               = 9
    RANGE               = 10
    AZIMUTH             = 11
    ELEV_ANGLE          = 12
    SNR                 = 13
    NOISE               = 14
    FRAME_NUM           = 15

class BOOT_MODE(IntEnum):
    STANDARD                = 0
    DEMO_VISUALIZER         = 1
    DEMO_DROPPED_FRAMES     = 2
    DEMO_PARSER             = 3
    DEMO_CONNECTION_TEST    = 4

class DEMO_VIS_DATA(IntEnum):
    RANGE               = 0
    DOPPLER_V           = 1