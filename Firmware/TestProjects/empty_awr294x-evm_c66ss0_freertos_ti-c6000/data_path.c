/**
 *   @file  data_path.c
 *
 *   @brief
 *      Implements Data path processing functionality.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2020-2021 Texas Instruments, Inc.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <drivers/hwa.h>
#include <ti/demo/awr294x/mmw/dss/mmw_dss.h> 
#include <ti_drivers_config.h>
#include <ti_board_config.h>
#include <ti_drivers_open_close.h>
#include <ti_board_open_close.h>

#include <ti/demo/awr294x/mmw/dss/mmw_dss.h> 


/**************************************************************************
 *************************** Global Definitions ********************************
 **************************************************************************/

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
extern MmwDemo_DSS_MCB    gMmwDssMCB;

/**************************************************************************
 *************************** Static Function Prototype***************************
 **************************************************************************/
static void MmwDemo_hwaOpen(MmwDemo_DataPathObj *obj);
static void MmwDemo_hwaClose(MmwDemo_DataPathObj *obj);



/**
 *  @b Description
 *  @n
 *      Open HWA driver instance
 *
 *  @param[in] obj      Pointer to data path object
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_hwaOpen(MmwDemo_DataPathObj *obj)
{
    int32_t             errCode;

    /* Open the HWA Instance */
    obj->hwaHandle = HWA_open(0, NULL, &errCode);
    
    if (obj->hwaHandle == NULL)
    {
        MmwDemo_debugAssert (0);
        return;
    }
}

/**
 *  @b Description
 *  @n
 *      Close HWA driver instance
 *
 *  @param[in] obj      Pointer to data path object
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_hwaClose(MmwDemo_DataPathObj *obj)
{
    int32_t             errCode;

    /* Close the HWA Instance */
    errCode = HWA_close(obj->hwaHandle);
    if (errCode != 0)
    {
        MmwDemo_debugAssert (0);
        return;
    }
}

/**
 *  @b Description
 *  @n
 *      Close EDMA driver instance
 *
 *  @param[in] obj      Pointer to data path object
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_edmaClose(MmwDemo_DataPathObj *obj)
{
    EDMA_close(obj->edmaHandle);
}

/**
 *  @b Description
 *  @n
 *      This function is called at the init time to open data path driver instances.
 *
 *  @param[in] obj      Pointer to data path object
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_dataPathOpen(MmwDemo_DataPathObj *obj)
{

    int32_t i;

    /*****************************************************************************
     * Open HWA, EDMA drivers instances
     *****************************************************************************/
    Drivers_open();
    Board_driversOpen();

    for (i = 0; i < CONFIG_EDMA_NUM_INSTANCES; i++){
        obj->edmaHandle[i] = gEdmaHandle[i];
    }

    MmwDemo_hwaOpen(obj);

    return;
}

/**
 *  @b Description
 *  @n
 *  This function is called to close data path driver instances.
 *
 *  @param[in] obj      Pointer to data path object
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_dataPathClose(MmwDemo_DataPathObj *obj)
{
    /* DPC close */
    DPM_deinit(obj->objDetDpmHandle);
    
    /* Close HWA driver */
    MmwDemo_hwaClose(obj);


    Board_driversClose();
    Drivers_close();

}
