# ========================================================================================================================
# Faceplant Forecast, 2025
# This is the script that must be run to start up all the other code blocks. It must be run whenever the Raspberry Pi
# boots up, and is also necessary if attempting to test integration between systems. This script creates shared memory
# buffers and handles the process of starting the other scripts as daemon processes.
# ========================================================================================================================

import multiprocessing
import numpy
