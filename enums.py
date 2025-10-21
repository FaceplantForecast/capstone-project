# ========================================================================
# This is a set of global enums that allow for better readability and more
# efficient changing of the code. This needs to be accessed by everything.
# ========================================================================

from enum import IntEnum

class BUFF_SIZES(IntEnum):
    CMD_BUFF    = 8 #in bytes

class CMD_INDEX(IntEnum):
    MAIN_STATUS         = 0
    CMD_PORT_STATUS     = 1
    AI_STATUS           = 2
    DESERIALIZER_STATUS = 3
    APP_STATUS          = 4
    FRAMERATE           = 5
    APP_CMD             = 6

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

class DESERIALIZER_STATUS(IntEnum):
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

