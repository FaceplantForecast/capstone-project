/*
 *  Copyright (C) 2021 Texas Instruments Incorporated
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

/*
 * Auto generated file
 */

#include "ti_drivers_open_close.h"
#include <kernel/dpl/DebugP.h>

void Drivers_open(void)
{
    Drivers_edmaOpen();
}

void Drivers_close(void)
{
    Drivers_edmaClose();
}

/*
 * EDMA
 */
/* EDMA Driver handles */
EDMA_Handle gEdmaHandle[CONFIG_EDMA_NUM_INSTANCES];

/* EDMA Driver Open Parameters */
EDMA_Params gEdmaParams[CONFIG_EDMA_NUM_INSTANCES] =
{
    {
        .intrEnable = TRUE,
        .errIntrEnable = FALSE,
    },
};

void Drivers_edmaOpen(void)
{
    uint32_t instCnt;
    int32_t  status = SystemP_SUCCESS;

    for(instCnt = 0U; instCnt < CONFIG_EDMA_NUM_INSTANCES; instCnt++)
    {
        gEdmaHandle[instCnt] = NULL;   /* Init to NULL so that we can exit gracefully */
    }

    /* Open all instances */
    for(instCnt = 0U; instCnt < CONFIG_EDMA_NUM_INSTANCES; instCnt++)
    {
        gEdmaHandle[instCnt] = EDMA_open(instCnt, &gEdmaParams[instCnt]);
        if(NULL == gEdmaHandle[instCnt])
        {
            DebugP_logError("EDMA open failed for instance %d !!!\r\n", instCnt);
            status = SystemP_FAILURE;
            break;
        }
    }

    if(SystemP_FAILURE == status)
    {
        Drivers_edmaClose();   /* Exit gracefully */
    }

    return;
}

void Drivers_edmaClose(void)
{
    uint32_t instCnt;

    /* Close all instances that are open */
    for(instCnt = 0U; instCnt < CONFIG_EDMA_NUM_INSTANCES; instCnt++)
    {
        if(gEdmaHandle[instCnt] != NULL)
        {
            EDMA_close(gEdmaHandle[instCnt]);
            gEdmaHandle[instCnt] = NULL;
        }
    }

    return;
}

