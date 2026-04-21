/*
 *  Copyright (C) 2022 Texas Instruments Incorporated
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
#include <stdio.h>
#include <drivers/soc.h>
#include <kernel/dpl/AddrTranslateP.h>
#include "ti_dpl_config.h"
#include "ti_drivers_config.h"


/* ----------- ClockP ----------- */
#define DSS_RTIA_CLOCK_SRC_MUX_ADDR (0x6000094u)
#define DSS_RTIA_CLOCK_SRC_SYSCLK (0x222u)
#define DSS_RTIA_BASE_ADDR     (0x6F7A000u)

ClockP_Config gClockConfig = {
    .timerBaseAddr = DSS_RTIA_BASE_ADDR, 
    .timerHwiIntNum = 66,
    .timerInputClkHz = 150000000,
    .timerInputPreScaler = 1,
    .usecPerTick = 1000,
    .intrPriority = 15,
};

/* ----------- DebugP ----------- */
void putchar_(char character)
{
    /* Output to shared memory */
    DebugP_shmLogWriterPutChar(character);
}

/* Shared memory log base address, logs of each CPUs are put one after other in the below region.
 *
 * IMPORTANT: Make sure of below,
 * - The section defined below should be placed at the exact same location in memory for all the CPUs
 * - The memory should be marked as non-cached for all the CPUs
 * - The section should be marked as NOLOAD in all the CPUs linker command file
 */
DebugP_ShmLog gDebugShmLog[CSL_CORE_ID_MAX] __attribute__((aligned(128), section(".bss.log_shared_mem")));

/* ----------- CacheP ----------- */
const CacheP_Config       gCacheConfig = {
    .enable = 1,
    .enableForceWrThru = 0,
};

CacheP_Size         gCacheSize = {
    .l1pSize = CacheP_L1Size_32K,
    .l1dSize = CacheP_L1Size_32K,
    .l2Size  = CacheP_L2Size_0K,
};

uint32_t            gCacheMarRegionNum = 1U;

CacheP_MarRegion    gCacheMarRegion[] = {
    {
        .baseAddr = (void *) 0xC5000000U,
        .size = 16U * 1024U * 1024U,
        .value = 0,
    },
};

void Dpl_init(void)
{
    /* Cache and MAR program need to happen early */
    CacheP_init();

    /* initialize Hwi but keep interrupts disabled */
    HwiP_init();

    /* init debug log zones early */
    /* Debug log init */
    DebugP_logZoneEnable(DebugP_LOG_ZONE_ERROR);
    DebugP_logZoneEnable(DebugP_LOG_ZONE_WARN);
    /* Initialize shared memory writer on this CPU */
    DebugP_shmLogWriterInit(&gDebugShmLog[CSL_CORE_ID_C66SS0], CSL_CORE_ID_C66SS0);


    /* set timer clock source */
    SOC_controlModuleUnlockMMR(SOC_DOMAIN_ID_DSS_RCM, 0);
    *(volatile uint32_t*)(DSS_RTIA_CLOCK_SRC_MUX_ADDR) = DSS_RTIA_CLOCK_SRC_SYSCLK;
    SOC_controlModuleLockMMR(SOC_DOMAIN_ID_DSS_RCM, 0);
    /* initialize Clock */
    ClockP_init();



    /* Enable interrupt handling */
    HwiP_enable();
}

void Dpl_deinit(void)
{
}
