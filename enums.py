# ========================================================================
# This is a set of global enums that allow for better readability and more
# efficient changing of the code. This needs to be accessed by everything.
# ========================================================================

from enum import IntEnum

class BUFF_SIZES(IntEnum):
    CMD_BUFF    = 8 #8 bytes

class CMD_INDEX(IntEnum):
    MAIN_STATUS         = 0
    CMD_PORT_STATUS     = 1
    AI_STATUS           = 2
    DESERIALIZER_STATUS = 3
    FRAMERATE           = 4
    APP_CMD             = 5