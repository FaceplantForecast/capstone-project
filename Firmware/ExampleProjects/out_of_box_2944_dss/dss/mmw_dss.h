/**
 *   @file  mmw_dss.h
 *
 *   @brief
 *      This is the main header file for the Millimeter Wave Demo
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2020 - 2021 Texas Instruments, Inc.
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
#ifndef MMW_DSS_H
#define MMW_DSS_H


#include <drivers/mailbox.h>
#include <drivers/hwa.h>
#include <drivers/edma.h>
#include <kernel/dpl/SemaphoreP.h>
#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/HeapP.h>
#include "FreeRTOS.h"
#include "task.h"

#include <ti/common/mmwave_error.h>

#ifdef MMWDEMO_TDM
#include <ti/demo/awr294x/mmw/mmw_resTDM.h>
#include <ti/datapath/dpc/objectdetection/objdethwa/objectdetection.h>
#else
#include <ti/demo/awr294x/mmw/mmw_resDDM.h>
#include <ti/datapath/dpc/objectdetection/objdethwaDDMA/objectdetection.h>
#endif
#include <ti/demo/awr294x/mmw/include/mmw_output.h>

/* This is used to resolve RL_MAX_SUBFRAMES */
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_dfp_02_04_18_01/ti/control/mmwavelink/mmwavelink.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================== */
/*                           Macros & Typedefs                              */
/* ======================================================================== */

/**
 * @brief
 *  Millimeter Wave Demo Data Path Object
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Millimeter Wave demo's data path
 */
typedef struct MmwDemo_DataPathObj_t
{
    /*! @brief Handle to hardware accelerator driver. */
    HWA_Handle          hwaHandle;

    /*! @brief dpm Handle */
    DPM_Handle          objDetDpmHandle;

    /*! @brief   Handle of the EDMA driver. */
    EDMA_Handle         edmaHandle[EDMA_NUM_CC];

    /*! @brief          Processing Stats */
    MmwDemo_output_message_stats   subFrameStats[RL_MAX_SUBFRAMES];
} MmwDemo_DataPathObj;

/**
 * @brief
 *  Millimeter Wave Demo MCB
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Millimeter Wave demo
 */
typedef struct MmwDemo_DSS_MCB_t
{       
    /*! @brief     DPM Handle */
    TaskHandle_t                 objDetDpmTaskHandle;

    /*! @brief     DPM task object */
    StaticTask_t                 objDetDpmTaskObj;

    /*! @brief     init Task Handle */
    TaskHandle_t                 initTaskHandle;

    /*! @brief     init Task object */
    StaticTask_t                 initTaskObj;

    /*! @brief   Semaphore Object to pend main task */
    SemaphoreP_Object             demoInitTaskCompleteSemHandle;
  
    /*! @brief     Data Path object */
    MmwDemo_DataPathObj         dataPathObj;
} MmwDemo_DSS_MCB;


/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/
extern void MmwDemo_dataPathOpen(MmwDemo_DataPathObj *obj);
extern void MmwDemo_dataPathClose(MmwDemo_DataPathObj *obj);


/* Sensor Management Module Exported API */
extern void _MmwDemo_debugAssert(int32_t expression, const char *file, int32_t line);
#define MmwDemo_debugAssert(expression) {                                      \
                                         DebugP_assert(expression);             \
                                        }
                                        
#ifdef __cplusplus
}
#endif

#endif /* MMW_DSS_H */

