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

#include "ti_drivers_config.h"


/*
 * BOOTLOADER
 */

/* Bootloader boot media specific arguments */
Bootloader_MemArgs gBootloader0Args =
{
    .curOffset        = 0,
    .appImageBaseAddr = 0x00000000,
};

/* Configuration option for lockstep or standalone */
void* operatingMode = NULL;

/* Bootloader driver configuration */
Bootloader_Config gBootloaderConfig[CONFIG_BOOTLOADER_NUM_INSTANCES] =
{
    {
        &gBootloaderMemFxns,
        &gBootloader0Args,
        BOOTLOADER_MEDIA_MEM,
        0,
        0,
        NULL,
        .socCoreOpMode= (void *)&operatingMode,
        .isAppimageSigned = TRUE,
        .disableAppImageAuth = FALSE,
        .isAppimageEncrypted = FALSE,
        .initICSSCores = FALSE,
    },
};

uint32_t gBootloaderConfigNum = CONFIG_BOOTLOADER_NUM_INSTANCES;

/*
 * HSM Client
 */

/* memory assigned for each R5x <-> HSM channel */
uint8_t gQueue_HsmToR5[SIPC_NUM_R5_CORES][SIPC_QUEUE_LENGTH*SIPC_MSG_SIZE] __attribute__((aligned(8),section(".bss.sipc_hsm_queue_mem")));
uint8_t gQueue_R5ToHsm[SIPC_NUM_R5_CORES][SIPC_QUEUE_LENGTH*SIPC_MSG_SIZE] __attribute__((aligned(8),section(".bss.sipc_r5f_queue_mem")));
HsmClient_t gHSMClient ;

void HsmClient_config(void)
{
    SIPC_Params sipcParams;
    int32_t status;

    /* initialize parameters to default */
    SIPC_Params_init(&sipcParams);

    sipcParams.ipcQueue_eleSize_inBytes = SIPC_MSG_SIZE;
    sipcParams.ipcQueue_length = SIPC_QUEUE_LENGTH ;
    /* list the cores that will do SIPC communication with this core
    * Make sure to NOT list 'self' core in the list below
    */
    sipcParams.numCores = 1;
    sipcParams.coreIdList[0] = CORE_INDEX_HSM;

    /* specify the priority of SIPC Notify interrupt */
    sipcParams.intrPriority = 7U;


    /* This is HSM -> R5F queue */
    sipcParams.tx_SipcQueues[CORE_INDEX_HSM] = (uintptr_t)gQueue_R5ToHsm[0] ;
    sipcParams.rx_SipcQueues[CORE_INDEX_HSM] = (uintptr_t)gQueue_HsmToR5[0] ;
    sipcParams.secHostCoreId[CORE_INDEX_SEC_MASTER_0] = CORE_ID_R5FSS0_0;

    /* initialize the HsmClient module */
    status = HsmClient_init(&sipcParams);
    DebugP_assert(status==SystemP_SUCCESS);

    /* register a hsm client to detect bootnotify message and keyring import from HSM */
    status = HsmClient_register(&gHSMClient, HSM_BOOT_NOTIFY_CLIENT_ID);
    DebugP_assert(status==SystemP_SUCCESS);
}

void HsmClient_unRegister(void)
{
     /* Unregister bootnotify client */
    HsmClient_unregister(&gHSMClient, HSM_BOOT_NOTIFY_CLIENT_ID);
}

/*
 * UART
 */
#include "drivers/soc.h"

/* UART atrributes */
static UART_Attrs gUartAttrs[CONFIG_UART_NUM_INSTANCES] =
{
    {
        .baseAddr           = CSL_MSS_SCIA_U_BASE,
        .inputClkFreq       = 150000000U,
    },
};
/* UART objects - initialized by the driver */
static UART_Object gUartObjects[CONFIG_UART_NUM_INSTANCES];
/* UART driver configuration */
UART_Config gUartConfig[CONFIG_UART_NUM_INSTANCES] =
{
    {
        &gUartAttrs[CONFIG_UART0],
        &gUartObjects[CONFIG_UART0],
    },
};

uint32_t gUartConfigNum = CONFIG_UART_NUM_INSTANCES;

void Drivers_uartInit(void)
{
    uint32_t i;
    for (i=0; i<CONFIG_UART_NUM_INSTANCES; i++)
    {
        SOC_RcmPeripheralId periphID;
        if(gUartAttrs[i].baseAddr == CSL_MSS_SCIA_U_BASE) {
            periphID = SOC_RcmPeripheralId_MSS_SCIA;
        } else if (gUartAttrs[i].baseAddr == CSL_MSS_SCIB_U_BASE) {
            periphID = SOC_RcmPeripheralId_MSS_SCIB;
        } else if (gUartAttrs[i].baseAddr == CSL_DSS_SCIA_U_BASE) {
            periphID = SOC_RcmPeripheralId_DSS_SCIA;
        } else {
            continue;
        }
        gUartAttrs[i].inputClkFreq = SOC_rcmGetPeripheralClock(periphID);
    }
    UART_init();
}

/*
 * EDMA
 */
/* EDMA atrributes */
static EDMA_Attrs gEdmaAttrs[CONFIG_EDMA_NUM_INSTANCES] =
{
    {

        .baseAddr           = CSL_RSS_TPCC_A_U_BASE,
        .tcBaseAddr[0]    = CSL_RSS_TPTC_A0_U_BASE,
        .numTptc            = 1,
        .compIntrNumber     = CSL_MSS_INTR_RSS_TPCC_A_INTAGG,
        .compIntrNumberDirMap      = 0,
        .isErrIntrAvailable   = 1,
        .errIntrNumber      = CSL_MSS_INTR_RSS_TPCC_A_ERRAGG,
        .errIntrNumberDirMap      = 0,
        .intrAggEnableAddr  = CSL_RSS_CTRL_U_BASE + CSL_RSS_CTRL_RSS_TPCC_A_INTAGG_MASK,
        .intrAggEnableMask  = 0x1FF & (~(2U << 2)),
        .intrAggStatusAddr  = CSL_RSS_CTRL_U_BASE + CSL_RSS_CTRL_RSS_TPCC_A_INTAGG_STATUS,
        .intrAggClearMask   = (2U << 2),
        .errIntrAggEnableAddr  = CSL_RSS_CTRL_U_BASE + CSL_RSS_CTRL_RSS_TPCC_A_ERRAGG_MASK,
        .errIntrAggStatusAddr  = CSL_RSS_CTRL_U_BASE + CSL_RSS_CTRL_RSS_TPCC_A_ERRAGG_STATUS,
        .initPrms           =
        {
            .regionId     = 2,
            .queNum       = 0,
            .initParamSet = FALSE,
            .ownResource    =
            {
                .qdmaCh      = 0x30U,
                .dmaCh[0]    = 0x00000000U,
                .dmaCh[1]    = 0x0000FFFFU,
                .tcc[0]      = 0x00000000U,
                .tcc[1]      = 0x0000FFFFU,
                .paramSet[0] = 0x00000000U,
                .paramSet[1] = 0x00000000U,
                .paramSet[2] = 0xFFFFFFFFU,
                .paramSet[3] = 0x00000000U,
            },
            .reservedDmaCh[0]    = 0x00000000U,
            .reservedDmaCh[1]    = 0x00000001U,
        },
    },
};

/* EDMA objects - initialized by the driver */
static EDMA_Object gEdmaObjects[CONFIG_EDMA_NUM_INSTANCES];
/* EDMA driver configuration */
EDMA_Config gEdmaConfig[CONFIG_EDMA_NUM_INSTANCES] =
{
    {
        &gEdmaAttrs[CONFIG_EDMA0],
        &gEdmaObjects[CONFIG_EDMA0],
    },
};

uint32_t gEdmaConfigNum = CONFIG_EDMA_NUM_INSTANCES;


void Pinmux_init(void);
void PowerClock_init(void);
void PowerClock_deinit(void);

/*
 * Common Functions
 */



void System_init(void)
{
    /* DPL init sets up address transalation unit, on some CPUs this is needed
     * to access SCICLIENT services, hence this needs to happen first
     */
    Dpl_init();

    

    PowerClock_init();
    /* Now we can do pinmux */
    Pinmux_init();
    /* finally we initialize all peripheral drivers */
    HsmClient_config();

    Drivers_uartInit();
    EDMA_init();
}

void System_deinit(void)
{
    HsmClient_unRegister();

    UART_deinit();
    EDMA_deinit();
    PowerClock_deinit();
    Dpl_deinit();
}
