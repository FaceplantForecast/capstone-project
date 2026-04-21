/**
 *   @file  mmwave_link_common.c
 *
 *   @brief
 *      The file implements the common mmWave control module which interfaces
 *      with the mmWave Link API
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2016-2025 Texas Instruments, Inc.
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
/* Uncomment below to enable debugp traces*/
#define DebugP_log_ENABLED 1

#include <stdint.h>
#include <string.h>

/* Includes from MCU Plus SDK */
#include <kernel/dpl/HwiP.h>
#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/ClockP.h>
#include <kernel/dpl/HeapP.h>
#include <drivers/hw_include/hw_types.h>

#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/common/syscommon.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/mmwave/mmwave.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/mmwave/include/mmwave_internal.h>

/* User defined heap memory and handle */
#define BPM_HEAP_MEM_SIZE  (2*1024u)

/* ADV_ESMFAULT Fatal Errors (Applicable for AWR294x, AWR2544, AWR2x44P) */
#define MMW_PROG_FILT_PARITY_ERROR_BIT      (1 << 14U)
#define MMW_ECC_AGG_DED_ERROR_BIT           (1 << 19U)
#define MMW_PROG_FILT_UERR_BIT              (1 << 25U)
#define MMW_ADV_ESM_GROUP1_FATAL_ERRS       (MMW_PROG_FILT_PARITY_ERROR_BIT | \
                                             MMW_ECC_AGG_DED_ERROR_BIT | \
                                             MMW_PROG_FILT_UERR_BIT)


static uint8_t gBpmHeapMem[BPM_HEAP_MEM_SIZE] __attribute__((aligned(HeapP_BYTE_ALIGNMENT)));
static HeapP_Object gBpmHeapObj;

/* Advanced Chirp LUT data */
/* Max size of the LUT is 12KB.*/
/* Maximum of 212 bytes per chunk can be present per message. */
/* This array is created to store the LUT RAM values from the user programmed parameters or config file. */
/* The chirp parameters start address offset should be 4 byte aligned */
#define LOCAL_LUT_SIZE (12*1024U)
rlInt8_t advChirpLUTBuf[LOCAL_LUT_SIZE] = {0};

/**************************************************************************
 ***************************** Local Functions ****************************
 **************************************************************************/

static int32_t MMWave_configureProfileChirp(MMWave_CtrlCfg * ptrControlCfg, int32_t* errCode);
static int32_t MMWave_configureProfile(MMWave_CtrlCfg* ptrControlCfg, int32_t* errCode);
static int32_t MMWave_configureAdvChirp(MMWave_MCB* ptrMMWaveMCB, int32_t* errCode);
static int32_t MMWave_deinitCRC (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode);
static int32_t MMWave_deinitMMWaveLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode);

#if defined (SOC_AWR2544)
static int32_t MMWave_frametrigSelCPTS(void);
#endif

/**************************************************************************
 *************************** Local Definitions ****************************
 **************************************************************************/
/**
 * @brief   This is the bit mask derived from the mmWavelink rlRfInitComplete_t
 * to indicate that the RF Initialization procedure was successful. As per the
 * documentation the bit mask is derived as follows:-
 *
 *  0       [Reserved]
 *  1       APLL tuning
 *  2       SYNTH VCO1 tuning
 *  3       SYNTH VCO2 tuning
 *  4       LODIST calibration
 *  5       RX ADC DC offset calibration
 *  6       HPF cutoff calibration
 *  7       LPF cutoff calibration
 *  8       Peak detector calibration
 *  9       TX Power calibration
 * 10       RX gain calibration
 * 11       TX Phase calibration
 * 12       RX IQMM calibration
 * 31:13    [Reserved]
 */
#define MMWAVE_INIT_CALIB_SUCCESS           0x1FFEU


/**************************************************************************
 ************************ mmWave Link Functions ***************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function which is invoked
 *      to ensure that the spawn function is invoked in a different execution
 *      context
 *
 *  @param[in]  fxn
 *      Pointer to the function to be executed in a different context
 *  @param[in]  pValue
 *      Pointer of data to be passed to the function
 *  @param[in]  flags
 *      Flag to indicate preference
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
rlInt32_t MMWave_spawn
(
    RL_P_OSI_SPAWN_ENTRY    fxn,
    const void*             pValue,
    uint32_t                flags
)
{
    MMWave_SpawnFxnNode*    ptrSpawnFxnNode;
    uintptr_t               key;
    int32_t                 retVal = 0;

    DebugP_logInfo("MMWave_spawn. Adding 0x%x\n",(uintptr_t)fxn);

    /* Critical Section: The spawn free list is a critical resource which is accessed
     * from multiple contexts */
    key = HwiP_disable();
    ptrSpawnFxnNode = (MMWave_SpawnFxnNode*)MMWave_listRemove ((MMWave_ListNode**)&gMMWave_MCB.ptrSpawnFxnFreeList);
    HwiP_restore (key);

    /* Did we get an entry? */
    if (ptrSpawnFxnNode != NULL)
    {
        /* YES: Populate the entry */
        ptrSpawnFxnNode->spawnEntry = fxn;
        ptrSpawnFxnNode->arg        = pValue;

        DebugP_logInfo("MMWave_spawn add list\n");

        /* Critical Section: The spawn active list is a critical resource which is accessed
         * from multiple contexts */
        key = HwiP_disable();
        MMWave_listAdd ((MMWave_ListNode**)&gMMWave_MCB.ptrSpawnFxnActiveList, (MMWave_ListNode*)ptrSpawnFxnNode);
        HwiP_restore (key);

        /* Keep track of the number of spawn messages which have been received */
        gMMWave_MCB.spawnCounter++;
        /* Wake up the mmWave execution thread  */
        SemaphoreP_post(&gMMWave_MCB.linkSemHandle);
    }
    else
    {
        /* Error: No span free node was present. This can happen if all the spawn functions
         * have been taken up and the execute mmWave control API has not been invoked. Increment
         * statistics to report this condition */
        gMMWave_MCB.spawnOverflow++;
        DebugP_logInfo("SPAWN error!!!\n");
        /* Setup the return value to indicate an error. */
        retVal = MINUS_ONE;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to compute
 *      the CRC for the specified data buffer.
 *
 *  @param[in]  ptrData
 *      Pointer to the data buffer for which the CRC is to be computed
 *  @param[in]  dataLen
 *      Length of the data buffer for which the CRC is to be computed
 *  @param[in]  crcType
 *      CRC Type i.e. 16bit, 32bit or 64bit.
 *  @param[out] crc
 *      Computed CRC
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
rlInt32_t MMWave_computeCRC
(
    rlUInt8_t*  ptrData,
    rlUInt32_t  dataLen,
    rlUInt8_t   crcType,
    rlUInt8_t*  crc
)
{
    uint64_t        signature;
    uint32_t        index;
    uint8_t*        ptrSignature;
    MMWave_MCB*     ptrMMWaveMCB;
    int32_t         retVal    = 0;
    uint32_t        crcLength = 0U;
    uint32_t        loopCnt;
    uint32_t        patternCnt;
    CRC_SignatureRegAddr   psaSignRegAddr;
    uint16_t*       ptrData16 = (uint16_t*)ptrData;
    uint32_t*       ptrData32 = (uint32_t*)ptrData;
    CRC_Config     config;

    /* Get the pointer to the control module */
    ptrMMWaveMCB = (MMWave_MCB*)&gMMWave_MCB;

    /* Sanity Check: Validate the parameters; determine the length of the CRC */
    switch (crcType)
    {
        case RL_CRC_TYPE_16BIT_CCITT:
        {
            /* CRC is 2 bytes */
            crcLength = 2U;
            patternCnt = dataLen/2;
            break;
        }
        case RL_CRC_TYPE_32BIT:
        {
            /* CRC is 4 bytes */
            crcLength = 4U;
            patternCnt = dataLen/4;
            break;
        }
        case RL_CRC_TYPE_64BIT_ISO:
        {
            /* CRC is 8 bytes */
            crcLength = 8U;
            patternCnt = dataLen/4;
            break;
        }
        default:
        {
            /* Error: The mmWave link has passed an invalid CRC Type */
            retVal = MINUS_ONE;
            break;
        }
    }

    /* Sanity Check: Did the link pass a valid CRC Type? */
    if (retVal == MINUS_ONE)
    {
        /* NO: Invalid CRC Type has been detected */
        DebugP_assert (0);
        goto exit;
    }

    config.mode         = CRC_OPERATION_MODE_FULLCPU;
    config.type         = CRC_TYPE_16BIT;
    config.dataLen      = CRC_DATALENGTH_16BIT;
    config.bitSwap      = CRC_BITSWAP_MSB;
    config.byteSwap     = CRC_BYTESWAP_ENABLE;
    config.patternCount = patternCnt;
    config.sectorCount  = 1;

    retVal = CRC_configure(ptrMMWaveMCB->initCfg.linkCRCCfg.crcBaseAddr,
                          ptrMMWaveMCB->initCfg.linkCRCCfg.crcChannel,
                          &config);

    if(retVal < 0)
    {
        /* Error: Unable to configure CRC moduel. */
        DebugP_logInfo ("Error: Configure of CRC failed [Error code %d]\n", retVal);
        DebugP_assert (0);
        goto exit;
    }

    /* Get CRC PSA signature register address */
    CRC_getPSASigRegAddr(ptrMMWaveMCB->initCfg.linkCRCCfg.crcBaseAddr,
                       ptrMMWaveMCB->initCfg.linkCRCCfg.crcChannel,
                       &psaSignRegAddr);

    /* Reset the CRC channel*/
    CRC_channelReset(ptrMMWaveMCB->initCfg.linkCRCCfg.crcBaseAddr,
                    ptrMMWaveMCB->initCfg.linkCRCCfg.crcChannel);

    /* compute the CRC by writing the data buffer on which CRC computation is needed */
    for (loopCnt = 0; loopCnt < patternCnt; loopCnt++)
    {
        if(crcType == RL_CRC_TYPE_16BIT_CCITT)
        {
            HW_WR_REG16(psaSignRegAddr.regL, (uint16_t)ptrData16[loopCnt]);
        }
        else
        {
            HW_WR_REG32(psaSignRegAddr.regL, (uint32_t)ptrData32[loopCnt]);
        }
    }

    /* Fetch CRC signature value */
    retVal = CRC_getPSASig(ptrMMWaveMCB->initCfg.linkCRCCfg.crcBaseAddr,
                          ptrMMWaveMCB->initCfg.linkCRCCfg.crcChannel,
                          (CRC_Signature*)&signature);

    /* Get the pointer to the CRC Signature: */
    ptrSignature = (uint8_t*)&signature;

    /* Copy the CRC signature into CRC output array*/
    for(index = 0U; index < crcLength; index++)
    {
        *(crc + index) = *(ptrSignature + index);
    }

    /* CRC signature has been computed correctly */
    retVal = 0;

exit:
    return (rlInt32_t)retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function which
 *      is used to handle asynchronous events which are received from
 *      the BSS.
 *
 *  @param[in]  devIndex
 *      Device Index
 *  @param[in]  sbId
 *      Subblock identifier
 *  @param[in]  sbLen
 *      Length of the subblock
 *  @param[in]  payload
 *      Pointer to the payload buffer
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void MMWave_asyncEventHandler
(
    uint8_t     devIndex,
    uint16_t    sbId,
    uint16_t    sbLen,
    uint8_t*    payload
)
{
    uint16_t            asyncSB;
    uint16_t            msgId;
    rlCpuFault_t        cpufault = { 0 };

    /* Get the sub-block and message identifier. */
    asyncSB = RL_GET_SBID_FROM_UNIQ_SBID(sbId);
    msgId   = RL_GET_MSGID_FROM_SBID(sbId);

    gMMWave_MCB.debugStats.numAsyncEvents++;

    DebugP_logInfo ("mmwavelink async event handler: BSS Event MsgId: %d [Device Index: 0x%x Sub Block Id: 0x%x Sub Block Length: %d]\n",
                msgId, devIndex, sbId, sbLen);


    /* Process the received message: */
    switch (msgId)
    {
        case RL_RF_ASYNC_EVENT_MSG:
        {
            /*******************************************************************
             * Received Asychronous Message
             *******************************************************************/
            switch (asyncSB)
            {
                case RL_RF_AE_INITCALIBSTATUS_SB:
                {
                    rlRfInitComplete_t*  ptrRFInitCompleteMessage;
                    uint32_t            calibrationStatus;

                    /* Get the RF-Init completion message: */
                    ptrRFInitCompleteMessage = (rlRfInitComplete_t*)payload;

                    /* Were we successful?  */
                    calibrationStatus = ptrRFInitCompleteMessage->calibStatus & MMWAVE_INIT_CALIB_SUCCESS;
                    if (calibrationStatus != 0)
                    {
                        /* YES: Mark the link as operational */
                        gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_INITIALIZED;
                    }
                    else
                    {
                        /* NO: Link initialization failed. */
                        gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_INIT_FAILED;
                    }
                    break;
                }
                case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
                {
                    /* Calibration has been completed. */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_CALIBRATION_DONE;
                    break;
                }
                case RL_RF_AE_MON_TIMING_FAIL_REPORT_SB:
                {
                    /* Calibration has failed. */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_CALIBRATION_FAILED;
                    break;
                }
                case RL_RF_AE_CPUFAULT_SB:
                {
                    /* BSS reports CPU Fault: */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_CPU_FAULT;
                    break;
                }
                /* ADV ESM fault is applicable for AWR294x/AWR2544/AWR2x44P devices*/
                case RL_RF_AE_ADV_ESMFAULT_SB:
                {
                    /* BSS reports ADV ESM Fault: */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_ESM_FAULT;

                    /* As per ICD section "7.1 API Error Handling"
                     *   Below listed ADV_ESM faults are fatal:
                     *      - ESM_GROUP2_ERRORS
                     *      - PROG_FILT_PARITY_ERROR
                     *      - PROG_FILT_UERR
                     *      - ECC_AGG_DED_ERROR
                     */
                    if (((rlBssEsmFault_t *)payload)->esmGrp2Err != 0U)
                    {
                        DebugP_log("Error!! RL_RF_AE_ADV_ESMFAULT_SB ESM_GROUP2_ERRORS Fatal error\r\n");
                        DebugP_assert (0);
                    }
                    else if (((rlBssEsmFault_t*)payload)->esmGrp1Err & MMW_ADV_ESM_GROUP1_FATAL_ERRS)
                    {
                        DebugP_log("Error!! RL_RF_AE_ADV_ESMFAULT_SB ESM_GROUP1_ERRORS Fatal error\r\n");
                        DebugP_assert (0);
                    }
                    break;
                }
                case RL_RF_AE_ANALOG_FAULT_SB:
                {
                    /* BSS reports Analog Fault: */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_RF_ANALOG_FAULT;
                    break;
                }
                default:
                {
                    /* Catch condition for any other sub block which the mmWave is NOT interested
                     * in. This is just a fall through. Pass the event to the application registered
                     * event function. */
                    break;
                }
            }
            break;
        }

        /* Async Event from MSS */
        case RL_DEV_ASYNC_EVENT_MSG:
        {
            switch (asyncSB)
            {
                case RL_DEV_AE_MSSPOWERUPDONE_SB:
                {
                    DebugP_logInfo("Received RL_DEV_AE_MSSPOWERUPDONE_SB\n");
                    gMMWave_MCB.frontEndMssStatus[devIndex] = gMMWave_MCB.frontEndMssStatus[devIndex] | MMWAVE_FRONT_END_MSS_POWER_UP_DONE;
                }
                break;
                case RL_DEV_AE_MSS_BOOTERRSTATUS_SB:
                {
                    DebugP_logInfo("Received RL_DEV_AE_MSS_BOOTERRSTATUS_SB\n");
                    gMMWave_MCB.frontEndMssStatus[devIndex] = gMMWave_MCB.frontEndMssStatus[devIndex] | MMWAVE_FRONT_END_MSS_BOOT_ERROR;
                }
                break;
                case RL_DEV_AE_RFPOWERUPDONE_SB:
                {
                    DebugP_logInfo("Received RL_DEV_AE_RFPOWERUPDONE_SB\n");
                    gMMWave_MCB.frontEndMssStatus[devIndex] = gMMWave_MCB.frontEndMssStatus[devIndex] | MMWAVE_FRONT_END_MSS_RF_POWER_UP_DONE;
                }
                break;
                case RL_DEV_AE_MSS_ESMFAULT_SB:
                {
                    DebugP_logInfo("MSS ESM Error \n");
                    gMMWave_MCB.frontEndMssStatus[devIndex] = gMMWave_MCB.frontEndMssStatus[devIndex] | MMWAVE_FRONT_END_MSS_ESM_FAULT;
                }
                break;
                case RL_DEV_AE_MSS_CPUFAULT_SB:
                {
                    DebugP_logInfo("Received MSS CPU Fault\n");
                    gMMWave_MCB.frontEndMssStatus[devIndex] = gMMWave_MCB.frontEndMssStatus[devIndex] | MMWAVE_FRONT_END_MSS_CPU_FAULT;

                    memcpy(&cpufault, payload, sizeof(rlCpuFault_t));

                    if (cpufault.faultLR != 0xFFFFE580)
                    {
                        DebugP_logInfo("Error!! RL_DEV_AE_MSS_CPUFAULT_SB cpufault.faultLR=%d\n",cpufault.faultLR);
                        DebugP_assert (0);
                    }
                }
                break;
                case RL_DEV_AE_MSS_RF_ERROR_STATUS_SB:
                {
                    DebugP_logInfo("MSS RF Error \n");
                }
                break;
                default:
                {
                    DebugP_logInfo("Unhandled Async Event msgId: 0x%x, asyncSB:0x%x  \n\n", msgId, asyncSB);
                    break;
                }
            }
            break;
        }


        case RL_RF_ASYNC_EVENT_1_MSG:
        {
            /*Just fwd message to application*/
            break;
        }

        /* Async Event from MMWL */
        case RL_MMWL_ASYNC_EVENT_MSG:
        {
            switch (asyncSB)
            {
                case RL_MMWL_AE_MISMATCH_REPORT:
                {
                    /* link reports protocol error in the async report from BSS */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_LINK_ASYNC_EVENT_MISMATCH_ERROR;
                    gMMWave_MCB.debugStats.numErrAsyncEvents++;

                    DebugP_logInfo("RL_MMWL_AE_MISMATCH_REPORT\n");

                    int errTemp = *(int32_t*)payload;
                    /* CRC mismatched in the received Async-Event msg */
                    if (errTemp == RL_RET_CODE_CRC_FAILED)
                    {
                        DebugP_logInfo("CRC failed \n");
                    }
                    /* Checksum mismatched in the received msg */
                    else if (errTemp == RL_RET_CODE_CHKSUM_FAILED)
                    {

                       DebugP_logInfo("Checksum failed \n");
                    }
                    /* Polling to HostIRQ timed out,
                    i.e. Device didn't respond to CNYS from the Host */
                    else if (errTemp == RL_RET_CODE_HOSTIRQ_TIMEOUT)
                    {
                        DebugP_logInfo("HostIRQ polling timed out \n");
                    }
                    else
                    {
                        DebugP_logInfo("mmWaveLink error \n");
                    }
                    break;
                }
                case RL_MMWL_AE_INTERNALERR_REPORT:
                {
                    gMMWave_MCB.debugStats.numErrAsyncEvents++;
                    DebugP_logInfo("RL_MMWL_AE_INTERNALERR_REPORT\n");
                    /* link reports internal error during BSS communication */
                    gMMWave_MCB.linkStatus[devIndex] = gMMWave_MCB.linkStatus[devIndex] | MMWAVE_LINK_ASYNC_EVENT_INTERNAL_ERROR;
                    break;
                }
                default:
                {
                    DebugP_logInfo("Unhandled Async Event msgId: 0x%x, asyncSB:0x%x  \n\n", msgId, asyncSB);
                    break;
                }
            }
            break;
        }
        default:
        {
            /* Error: Received an invalid message identifier in the ASYNC Handler */
            DebugP_logInfo ("Error: Invalid message %d is received in the async handler\n", msgId);
            DebugP_assert (0);
            break;
        }
    }

    /* Pass the event to the application registered handler */
    gMMWave_MCB.initCfg.eventFxn (devIndex, msgId, sbId, sbLen, payload);

    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the CRC channel
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_initCRC (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t     retVal = MINUS_ONE;

    /* Do we need to use the CRC Driver? */
    if (ptrMMWaveMCB->initCfg.linkCRCCfg.useCRCDriver == 1U)
    {
        /* Configure CRC Module */
        retVal = CRC_channelReset(ptrMMWaveMCB->initCfg.linkCRCCfg.crcBaseAddr,
                                    ptrMMWaveMCB->initCfg.linkCRCCfg.crcChannel);
        DebugP_assert(retVal == SystemP_SUCCESS);

        retVal = CRC_initialize(ptrMMWaveMCB->initCfg.linkCRCCfg.crcBaseAddr,
                                ptrMMWaveMCB->initCfg.linkCRCCfg.crcChannel, 0, 0);

        if (retVal != SystemP_SUCCESS)
        {
            /* Error: Unable to open the CRC Driver */
            *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINIT, *errCode);
            goto exit;
        }
    }

    /* Control comes here implies that the CRC driver has been setup successfully */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to deinitialize the CRC channel which had
 *      been configured.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MMWave_deinitCRC (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t     retVal = 0;

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to perform an internal synchronization waiting
 *      for the specific flag to be set. The function will loop around until
 *      the condition is met
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[in]  flag
 *      Flag for which we are synchronizing
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void MMWave_internalLinkSync
(
    const MMWave_MCB*   ptrMMWaveMCB,
    uint32_t            flag
)
{
    uint32_t u32DevIdx;

    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        /* Loop around till the condition is met: */
        while((ptrMMWaveMCB->linkStatus[u32DevIdx] & flag) == 0U)
        {
            /* Nothing to do, just wait */
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure BPM
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the MCB
 *  @param[in]  ptrControlCfg
 *      Pointer to the control config
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_configBPM
(
    MMWave_MCB*         ptrMMWaveMCB,
    MMWave_CtrlCfg*     ptrControlCfg,
    int32_t*            errCode
)
{
    int32_t                 retVal;
    uint32_t                numBpmChirps = 0;
    uint32_t                index;
    MMWave_BpmChirpHandle   bpmChirpHandle;
    rlBpmChirpCfg_t**       bpmPtrArray;
    rlBpmCommonCfg_t        bpmCommonCfg;
    uint32_t                arraySize;
    MMWave_BpmChirp*        ptrMMWaveBpmChirp;

    /* Get the number of BPM chirps configured */
    if (MMWave_getNumBpmChirp ((MMWave_Handle)ptrMMWaveMCB, &numBpmChirps, errCode) < 0)
    {
        /* Error: Unable to get the number of BPM chirps. Error code is already setup */
        retVal = MINUS_ONE;
        goto end;
    }

    if(numBpmChirps == 0)
    {
        /* No BPM chirp configured. Nothing to be done.*/
        retVal = 0;
        goto end;
    }

    arraySize = numBpmChirps * sizeof(rlBpmChirpCfg_t*);

    /* Allocate array to store pointers to BPM configurations*/
    bpmPtrArray = (rlBpmChirpCfg_t**) HeapP_alloc(&gBpmHeapObj, arraySize);

    if (bpmPtrArray == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        retVal   = MINUS_ONE;
        goto end;
    }

    /* Initialize the allocated memory for the chirp: */
    memset ((void*)bpmPtrArray, 0, arraySize);

    /* Select source of BPM pattern to be from BPM chirp cfg defined in bpmChirpCfg*/
    memset ((void *)&bpmCommonCfg, 0, sizeof(rlBpmCommonCfg_t));
    bpmCommonCfg.mode.b2SrcSel = 0U;

    /* Set the BPM common config */
    retVal = rlSetBpmCommonConfig(ptrMMWaveMCB->deviceMap, &bpmCommonCfg);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Setting the BPM configuration failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECOMMONBPMCFG, retVal);
        retVal   = MINUS_ONE;
        goto end;
    }

    /* Cycle through all the BPM configurations and populate array. */
    for (index = 1U; index <= numBpmChirps; index++)
    {
        /* Get the Handle associated to the specified index */
        if (MMWave_getBpmChirpHandle ((MMWave_Handle)ptrMMWaveMCB, index, &bpmChirpHandle, errCode) < 0)
        {
            /* Error: Unable to get the handle. Error code is already setup */
            retVal = MINUS_ONE;
            goto end;
        }

        /* Populate the BPM cfg array. Note that index starts from 1 and array starts from zero. */
        ptrMMWaveBpmChirp = (MMWave_BpmChirp*)bpmChirpHandle;
        bpmPtrArray[index-1] = (rlBpmChirpCfg_t*)(&ptrMMWaveBpmChirp->bpmChirp);
    }

    /* Set the BPM chirp configuration in the mmWave link */
    retVal = rlSetMultiBpmChirpConfig(ptrMMWaveMCB->deviceMap,
                                     (rlUInt16_t)numBpmChirps,
                                     bpmPtrArray);

    /* Free the memory used by the config array) */
    HeapP_free(&gBpmHeapObj, (void *)bpmPtrArray);

    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Setting the BPM configuration failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EBPMCFG, retVal);
        retVal   = MINUS_ONE;
        goto end;
    }

end:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure chirp level phase shifter
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the MCB
 *  @param[in]  ptrControlCfg
 *      Pointer to the control config for frontend
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_configPhaseShift
(
    MMWave_MCB*         ptrMMWaveMCB,
    MMWave_CtrlCfg*     ptrControlCfg,
    int32_t*            errCode
)
{
    int32_t                         retVal;
    uint32_t                        numPhaseShiftChirps = 0;
    uint32_t                        index;
    rlRfMiscConf_t                  stRFMiscCnf;
    MMWave_PhaseShiftChirpHandle    phaseShiftChirpHandle;
    rlRfPhaseShiftCfg_t             phaseShiftChirpCfg;
    uint32_t                        u32DevIdx = 0;
    uint8_t                         u8DeviceMap;

    /* Get the number of chirp level phase shifter chirps configured */
    if (MMWave_getNumPhaseShiftChirp ((MMWave_Handle)ptrMMWaveMCB, &numPhaseShiftChirps, errCode) < 0)
    {
        /* Error: Unable to get the number of chirp level phase shifter chirps. Error code is already setup */
        retVal = MINUS_ONE;
        goto end;
    }

    if(numPhaseShiftChirps == 0U)
    {
        /* Per chirp phase shifter is disabled */
        stRFMiscCnf.miscCtl = 0U;
    }
    else
    {
        /* Per chirp phase shifter is enabled */
        stRFMiscCnf.miscCtl = 1U;
    }

    stRFMiscCnf.crdNSlopeMag = 0U;
    stRFMiscCnf.fastResetEndTime = 0U;

   /* Loop across all devices to send phase Shift chirp configurations */
    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        /* Create a local device map to select each device one per one */
        u8DeviceMap = (uint8_t) ((uint32_t) 1U << u32DevIdx);

        retVal = rlRfSetMiscConfig(u8DeviceMap, &stRFMiscCnf);

        if(retVal != RL_RET_CODE_OK)
        {
        /* Error: Setting the Miscellaneous configuration failed */
        retVal = MINUS_ONE;
        goto end;
        }

        if(MMWAVE_RADAR_DEVICES > 1)
        {
            if(u32DevIdx == 0)
            {
                /* Cycle through all the chirp level phase shifter and configure them. */
                for (index = 1U; index <= ptrControlCfg->numOfPhaseShiftChirps[u32DevIdx]; index++)
                {
                    /* Get the Chirp level phase shifter Handle associated at the specified index */
                    if (MMWave_getPhaseShiftChirpHandle ((MMWave_Handle)ptrMMWaveMCB, index, &phaseShiftChirpHandle, errCode) < 0)
                    {
                        /* Error: Unable to get the handle. Error code is already setup */
                        retVal = MINUS_ONE;
                        goto end;
                    }

                    /* Get the chirp level phase shifter configuration: */
                    if (MMWave_getPhaseShiftChirpCfg (phaseShiftChirpHandle, &phaseShiftChirpCfg, errCode) < 0)
                    {
                        /* Error: Unable to get the chirp configuration. Error code is already setup */
                        retVal = MINUS_ONE;
                        goto end;
                    }

                    /* Set the chirp level phase shifter configuration in the mmWave link */
                    retVal = rlRfSetPhaseShiftConfig(u8DeviceMap, 1U, &phaseShiftChirpCfg);

                    if (retVal != RL_RET_CODE_OK)
                    {
                        /* Error: Setting the chirp level phase shifter configuration failed */
                        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPHASESHIFTCFG, retVal);
                        retVal   = MINUS_ONE;
                        goto end;
                    }
                }
            }
            else
            {
                /* Cycle through all the chirp level phase shifter and configure them. */
                for (;index <= numPhaseShiftChirps; index++)
                {
                    /* Get the Chirp level phase shifter Handle associated at the specified index */
                    if (MMWave_getPhaseShiftChirpHandle ((MMWave_Handle)ptrMMWaveMCB, index, &phaseShiftChirpHandle, errCode) < 0)
                    {
                        /* Error: Unable to get the handle. Error code is already setup */
                        retVal = MINUS_ONE;
                        goto end;
                    }

                    /* Get the chirp level phase shifter configuration: */
                    if (MMWave_getPhaseShiftChirpCfg (phaseShiftChirpHandle, &phaseShiftChirpCfg, errCode) < 0)
                    {
                        /* Error: Unable to get the chirp configuration. Error code is already setup */
                        retVal = MINUS_ONE;
                        goto end;
                    }

                    /* Set the chirp level phase shifter configuration in the mmWave link */
                    retVal = rlRfSetPhaseShiftConfig(u8DeviceMap, 1U, &phaseShiftChirpCfg);

                    if (retVal != RL_RET_CODE_OK)
                    {
                        /* Error: Setting the chirp level phase shifter configuration failed */
                        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPHASESHIFTCFG, retVal);
                        retVal   = MINUS_ONE;
                        goto end;
                    }
                }
            }
        }
        else
        {
            /* Cycle through all the chirp level phase shifter and configure them. */
            for (index = 1U; index <= numPhaseShiftChirps; index++)
            {
                /* Get the Chirp level phase shifter Handle associated at the specified index */
                if (MMWave_getPhaseShiftChirpHandle ((MMWave_Handle)ptrMMWaveMCB, index, &phaseShiftChirpHandle, errCode) < 0)
                {
                    /* Error: Unable to get the handle. Error code is already setup */
                    retVal = MINUS_ONE;
                    goto end;
                }

                /* Get the chirp level phase shifter configuration: */
                if (MMWave_getPhaseShiftChirpCfg (phaseShiftChirpHandle, &phaseShiftChirpCfg, errCode) < 0)
                {
                    /* Error: Unable to get the chirp configuration. Error code is already setup */
                    retVal = MINUS_ONE;
                    goto end;
                }

                /* Set the chirp level phase shifter configuration in the mmWave link */
                retVal = rlRfSetPhaseShiftConfig(u8DeviceMap, 1U, &phaseShiftChirpCfg);

                if (retVal != RL_RET_CODE_OK)
                {
                    /* Error: Setting the chirp level phase shifter configuration failed */
                    *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPHASESHIFTCFG, retVal);
                    retVal   = MINUS_ONE;
                    goto end;
                }
            }
        }
    }

end:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure Programmable Filter RAM Coefficients
 *
 *      Filter configuration:
 *
 *      From    To     Gain    Ripple    Actual ripple
 *      0Mhz    4Mhz    1        5dB        3.55dB
 *      6Mhz    10Mhz   0       -60dB      -61.37dB
 *
 *      Sampling Frequency: 20MHz
 *      Desired Taps: 18
 *
 *  @param[in]  progFiltCfg
 *      Pointer to the MMWave_FiltCfg
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */

static void MMWave_configProgFiltCoeff(MMWave_FiltCfg *progFiltCfg)
{
    int16_t coeff[] = {-13,422,1604,2261,403,-2679,-1718,5730,13677,13677,5730,-1718,-2679,403,2261,1604,422,-13};
    uint32_t loopIdx;

    memset((void *)(&progFiltCfg->coeffRam),0,sizeof(progFiltCfg->coeffRam));

    for(loopIdx = 0; loopIdx < sizeof(coeff)/sizeof(coeff[0]); loopIdx++)
    {
        progFiltCfg->coeffRam.coeffArray[loopIdx] = coeff[loopIdx];
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure Programmable Filter RAM Coefficients
 *
 *  @param[in]  filtConf
 *      Pointer to the rlRfProgFiltConf_t
 *  @param[in] profId
 *      Profile ID
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static void MMWave_configProgFilt (rlRfProgFiltConf_t *filtConf, uint32_t profId)
{
    filtConf->coeffStartIdx = 0;
    filtConf->profileId = profId;
    filtConf->progFiltLen = 18;
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure the mmWave link with the specified profile
 *      and corresponding chirp configuration. The following order is preserved in
 *      the function:
 *          - Profile configuration
 *          - Chirp configuration
 *
 *  @param[in]  ptrControlCfg
 *      Pointer to the control config
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MMWave_configureProfileChirp(MMWave_CtrlCfg * ptrControlCfg, int32_t* errCode)
{

    MMWave_ProfileHandle  * ptrProfileHandle;
    int32_t                 retVal;
    int32_t                 index;
    rlProfileCfg_t          profileCfg;
    MMWave_ChirpHandle      chirpHandle;
    rlChirpCfg_t            chirpCfg;
    uint32_t                numChirps;
    uint32_t                chirpIndex;
    uint32_t                u32DevIdx;
    uint8_t                 u8DeviceMap;
    MMWave_FiltCfg          progFiltCfg;

    /* Loop across all devices to send profile and chirp configurations */
    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        /* Create a local device map to select each device one per one */
        u8DeviceMap = (uint8_t) ((uint32_t) 1U << u32DevIdx);

        /* Get the first profile handler of the selected device */
        if (ptrControlCfg->dfeDataOutputMode == MMWave_DFEDataOutputMode_FRAME)
        {
            ptrProfileHandle = &ptrControlCfg->u.frameCfg[u32DevIdx].profileHandle[0];
        }
        else if(ptrControlCfg->dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_FRAME)
        {
            ptrProfileHandle = &ptrControlCfg->u.advancedFrameCfg[u32DevIdx].profileHandle[0];
        }
        else{
            /* control should not come here as this function is called only in above two cases */
            ptrProfileHandle = NULL;
            retVal   = MINUS_ONE;
            goto end;
        }

        /* Prog Filter Coeff Config */
        if(ptrControlCfg->enableProgFilter)
        {
            MMWave_configProgFiltCoeff(&progFiltCfg);
            retVal = rlRfSetProgFiltCoeffRam(u8DeviceMap, &(progFiltCfg.coeffRam));
            if (retVal != RL_RET_CODE_OK)
            {
                *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EPROFILECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
        }

        /* Cycle through all the profile(s) which have been specified. */
        for (index = 0; index < MMWAVE_MAX_PROFILE; index++)
        {
            /* Do we have a valid profile? */
            if (ptrProfileHandle[index] == NULL)
            {
                /* NO: Skip to the next profile */
                continue;
            }
            /* Prog Filter Coeff Config */
            if (ptrControlCfg->enableProgFilter)
            {
                MMWave_configProgFilt(&(progFiltCfg.filtConf[index]), index);
                retVal = rlRfSetProgFiltConfig(u8DeviceMap, &(progFiltCfg.filtConf[index]));
                if (retVal != RL_RET_CODE_OK)
                {
                    *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EPROFILECFG, retVal);
                    retVal   = MINUS_ONE;
                    goto end;
                }
            }

            /* YES: Get the profile configuration */
            if (MMWave_getProfileCfg(ptrProfileHandle[index], &profileCfg, errCode) < 0)
            {
                /* Error: Unable to get the profile configuration. Setup the return value */
                retVal = MINUS_ONE;
                goto end;
            }

            /* Configure the profile using the mmWave Link API */
            retVal = rlSetProfileConfig(u8DeviceMap, 1U, &profileCfg);

            if (retVal != RL_RET_CODE_OK)
            {
                *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EPROFILECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }

            /* Get the number of chirps configured and attached to the profile: */
            if (MMWave_getNumChirps(ptrProfileHandle[index], &numChirps, errCode) < 0)
            {
                /* Error: Unable to get the number of chirps. Error code is already setup */
                retVal = MINUS_ONE;
                goto end;
            }

            /* For the profile; Cycle through all the chirps and configure them. */
            for (chirpIndex = 1U; chirpIndex <= numChirps; chirpIndex++)
            {
                /* Get the Chirp Handle associated at the specified index */
                if (MMWave_getChirpHandle(ptrProfileHandle[index], chirpIndex, &chirpHandle, errCode) < 0)
                {
                    /* Error: Unable to get the chirp handle. Error code is already setup */
                    retVal = MINUS_ONE;
                    goto end;
                }

                /* Get the chirp configuration: */
                if (MMWave_getChirpCfg(chirpHandle, &chirpCfg, errCode) < 0)
                {
                    /* Error: Unable to get the chirp configuration. Error code is already setup */
                    retVal = MINUS_ONE;
                    goto end;
                }

                /* Set the chirp configuration in the mmWave link */
                retVal = rlSetChirpConfig(u8DeviceMap, 1U, &chirpCfg);
                if (retVal != RL_RET_CODE_OK)
                {
                    *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_ECHIRPCFG, retVal);
                    retVal   = MINUS_ONE;
                    goto end;
                }
            }
        }
    }

    /* Control comes here implies that the profile & chirp was configured successfully */
    retVal = 0;

end:
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to configure the mmWave link with the specified profile.
 *  @param[in]  ptrControlCfg
 *      Pointer to the control config
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MMWave_configureProfile
(
    MMWave_CtrlCfg* ptrControlCfg,
    int32_t* errCode
)
{

    MMWave_ProfileHandle*   ptrProfileHandle;
    int32_t                 retVal;
    int32_t                 index;
    rlProfileCfg_t          profileCfg;
    uint32_t                u32DevIdx;
    uint8_t                 u8DeviceMap;

    /* Loop across all devices to send profile and advaced chirp configurations */
    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        /* Create a local device map to select each device one per one */
        u8DeviceMap = (uint8_t) ((uint32_t) 1U << u32DevIdx);

        /* Get the first profile handler of the selected device */
        if (ptrControlCfg->dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME)
        {
            ptrProfileHandle = &ptrControlCfg->u.frameCfg[u32DevIdx].profileHandle[0];
        }
        else if (ptrControlCfg->dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME)
        {
            /* Advanced chirp with advanced frame mode */
            ptrProfileHandle = &ptrControlCfg->u.advancedFrameCfg[u32DevIdx].profileHandle[0];
        }
        else{
            /* control should not come here as this function is called only in above two cases */
            ptrProfileHandle = NULL;
            retVal   = MINUS_ONE;
            goto end;
        }

        /* Cycle through all the profile(s) which have been specified. */
        for (index = 0; index < MMWAVE_MAX_PROFILE; index++)
        {
            /* Do we have a valid profile? */
            if (ptrProfileHandle[index] == NULL)
            {
                /* NO: Skip to the next profile */
                continue;
            }

            /* YES: Get the profile configuration */
            if (MMWave_getProfileCfg(ptrProfileHandle[index], &profileCfg, errCode) < 0)
            {
                /* Error: Unable to get the profile configuration. Setup the return value */
                retVal = MINUS_ONE;
                goto end;
            }

            /* Configure the profile using the mmWave Link API */
            retVal = rlSetProfileConfig(u8DeviceMap, 1U, &profileCfg);

            if (retVal != RL_RET_CODE_OK)
            {
                *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EPROFILECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
        }

    }

    /* Control comes here implies that the profile was configured successfully */
    retVal = 0;

end:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to save Advanced Chirp LUT Data to a file.
 *  @param[in]  advChirpLUTCfgArgs
 *      LUT start and end
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -  <0
 */
int MMWL_saveAdvChirpLUTDataToFile(rlAdvChirpLUTCfg_t advChirpLUTCfgArgs)
{
	int retVal = RL_RET_CODE_OK;
    rlUInt16_t idx=0;
	FILE *AdvChirpLUTDataPtr = fopen("AdvChirpLUTData.txt", "w");

    if (AdvChirpLUTDataPtr == NULL)
        return -1;

    for(idx=advChirpLUTCfgArgs.lutAddressOffset; idx<advChirpLUTCfgArgs.numBytes; idx++)
    {
        fprintf(AdvChirpLUTDataPtr,"%d\n", advChirpLUTBuf[idx]);
    }

	if (AdvChirpLUTDataPtr != NULL)
		fclose(AdvChirpLUTDataPtr);

	return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure the mmWave link with the advanced chirp configuration parameters.
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MMWave_configureAdvChirp
(
    MMWave_MCB* ptrMMWaveMCB,
    int32_t* errCode
)
{
    int32_t                 retVal;
    uint32_t                patternIdx, advChirpParamIndex;
    uint32_t                numAdvChirpCfgParams=0, u32DevIdx;
    rlRfMiscConf_t          stRFMiscCnf;
    MMWave_AdvChirpHandle   advChirpHandle;
    MMWave_AdvChirp*        ptrMMWaveAdvChirp;
    rlAdvChirpCfg_t         advChirpCfg;
    rlInt8_t                lutDataBuf[MMWAVE_EACH_CHIRP_PARAM_MAX_LUT_BUF_SIZE*4U]={0};
    rlInt32_t*              lutDataPtr;
    rlInt32_t               lutData;
    uint8_t                 u8DeviceMap;
    rlFillLUTParams_t       rlFillLUTParams;
    rlUInt16_t              lutOffsetInNBytes = 0, lutStartOffset=LOCAL_LUT_SIZE-1, lutEndOffset=0;
    rlAdvChirpLUTCfg_t      advChirpLUTCfgArgs = { 0 };

    memset((void*)&stRFMiscCnf,0,sizeof(rlRfMiscConf_t));

    /* enable advanced chirp configuration */
    stRFMiscCnf.miscCtl = 3U;
    stRFMiscCnf.crdNSlopeMag = 0U;
    stRFMiscCnf.fastResetEndTime = 0U;

    /* Loop across all devices to send advaced chirp configurations */
    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        /* Create a local device map to select each device one per one */
        u8DeviceMap = (uint8_t) ((uint32_t) 1U << u32DevIdx);

        /* Get the number of advanced chirps parameters configured */
        if (MMWave_getNumAdvChirpParams((MMWave_Handle)ptrMMWaveMCB, &numAdvChirpCfgParams, errCode) < 0)
        {
            /* Error: Unable to get the number of Advanced chirp parameters configured. Error code is already setup */
            retVal = MINUS_ONE;
            goto end;
        }

        if(numAdvChirpCfgParams!= MMWAVE_ADVANCE_CHIRP_NUM_PARAMS)
        {
            /* Error: incorrect number of advanced chirp config parameters */
            retVal = MINUS_ONE;
            goto end;
        }

        retVal = rlRfSetMiscConfig(u8DeviceMap, &stRFMiscCnf);

        /* Cycle through all the chirp parameters and configure them. */
        for (advChirpParamIndex = 0U; advChirpParamIndex < numAdvChirpCfgParams; advChirpParamIndex++)
        {
            /* Get the Advanced Chirp Handle associated at the specified param index */
            if(MMWave_getAdvChirpHandle((MMWave_Handle)ptrMMWaveMCB, advChirpParamIndex, &advChirpHandle, errCode)<0)
            {
                /* Error: Unable to get the chirp handle */
                retVal   = MINUS_ONE;
                goto end;
            }

            ptrMMWaveAdvChirp = (MMWave_AdvChirp*)advChirpHandle;

            /* get the advance chirp configs */
            memcpy((void*)&advChirpCfg, (void*)&ptrMMWaveAdvChirp->advChirpCfg, sizeof(rlAdvChirpCfg_t));

            /* Set the chirp configuration in the mmWave link */
            retVal = rlSetAdvChirpConfig(u8DeviceMap, &advChirpCfg);
            if (retVal != RL_RET_CODE_OK)
            {
                *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EADVCHIRPCFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }

            /* get the lutDataPtr */
            lutDataPtr = (rlInt32_t*)ptrMMWaveAdvChirp->ptrLUTData;

            for(patternIdx = 0; patternIdx < advChirpCfg.numOfPatterns; patternIdx++)
            {
                /* get the lut data */
                lutData = *(lutDataPtr + patternIdx);

                /* Set the lut data in the local buffer lutDataBuf */
                switch(advChirpParamIndex)
                {
                    case RL_LUT_CHIRP_PROFILE_VAR:
                    case RL_LUT_CHIRP_FREQ_SLOPE_VAR:
                    case RL_LUT_CHIRP_TX_EN_VAR:
                    case RL_LUT_CHIRP_BPM_VAL_VAR:
                    case RL_LUT_TX0_PHASE_SHIFT_VAR:
                    case RL_LUT_TX1_PHASE_SHIFT_VAR:
                    case RL_LUT_TX2_PHASE_SHIFT_VAR:
                    case RL_LUT_TX3_PHASE_SHIFT_VAR:
                    {
                        lutDataBuf[patternIdx] = (rlInt8_t)lutData;
                        break;
                    }
                    case RL_LUT_CHIRP_FREQ_START_VAR:
                    {
                        /* 4 Bytes of size */
                        if (advChirpCfg.lutChirpParamSize == 0U)
                        {
                            /* copy 4 bytes for each data value */
                            memcpy(lutDataBuf + 4*patternIdx, (rlInt8_t*)&lutData, 4U);
                        }
                        /* 2 Bytes of size */
                        else if (advChirpCfg.lutChirpParamSize == 1U)
                        {
                            /* copy 2 bytes of data value */
                            memcpy(lutDataBuf + 2*patternIdx, (rlInt8_t*)&lutData, 2U);
                        }
                        /* 1 Bytes of size */
                        else if (advChirpCfg.lutChirpParamSize == 2U)
                        {
                            /* copy 1 byte of data value */
                            lutDataBuf[patternIdx] = (rlInt8_t)lutData;
                        }
                        /* Invalid chirp param size */
                        else
                        {
                            retVal = MINUS_ONE;
                        }
                        break;

                    }
                    case RL_LUT_CHIRP_IDLE_TIME_VAR:
                    case RL_LUT_CHIRP_ADC_START_TIME_VAR:
                    {
                        /* 2 Bytes of size */
                        if (advChirpCfg.lutChirpParamSize == 0U)
                        {
                            /* copy 2 bytes for each data value */
                            memcpy(lutDataBuf + 2*patternIdx, (rlInt8_t*)&lutData, 2U);
                        }
                        /* 1 Bytes of size */
                        else if (advChirpCfg.lutChirpParamSize == 1U)
                        {
                            /* copy 1 byte of data value */
                            lutDataBuf[patternIdx] = (rlInt8_t)lutData;
                        }
                        /* Invalid chirp param size */
                        else
                        {
                            retVal = MINUS_ONE;
                        }
                        break;
                    }
                }/* end of switch case */
            }/* end of numOfPatterns lut data loop */

            /* fill up the Chirp LUT buffer which is used later for rlSetAdvChirpLUTConfig API */
            rlFillLUTParams.chirpParamIndex = advChirpCfg.chirpParamIdx;
            rlFillLUTParams.chirpParamSize = advChirpCfg.lutChirpParamSize;
            rlFillLUTParams.inputSize = advChirpCfg.numOfPatterns;
            rlFillLUTParams.lutGlobalOffset = advChirpCfg.lutPatternAddressOffset;
            /* check to avoid overwriting of parameters; params can be in any order but lutoffset should be sequential to avoid chances of error*/
            if(lutOffsetInNBytes > rlFillLUTParams.lutGlobalOffset)
            {
                *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EFillLUTBUF, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
            retVal = rlDevSetFillLUTBuff(&rlFillLUTParams, &lutDataBuf[0], &advChirpLUTBuf[rlFillLUTParams.lutGlobalOffset], &lutOffsetInNBytes);
            if (retVal != RL_RET_CODE_OK)
            {
                *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EFillLUTBUF, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }

            /* Since parameters can come in any order; ensure LUT data is saved starting from minimum address to maximum*/
            if(lutStartOffset > rlFillLUTParams.lutGlobalOffset)
            {
                lutStartOffset = rlFillLUTParams.lutGlobalOffset;
            }
            if(lutEndOffset < lutOffsetInNBytes)
            {
                lutEndOffset = lutOffsetInNBytes;
            }
        }/* end of chirp parameter loop*/

        /* Send the locally programmed LUT data to the device */
        advChirpLUTCfgArgs.lutAddressOffset = lutStartOffset;
        advChirpLUTCfgArgs.numBytes = ((lutEndOffset+3U)/4U)*4U; /* last address round to 4 bytes*/

        retVal = rlSetMultiAdvChirpLUTConfig(u8DeviceMap, &advChirpLUTCfgArgs, &advChirpLUTBuf[0]);
        if(retVal != RL_RET_CODE_OK)
        {
            goto end;
        }

        retVal = MMWL_saveAdvChirpLUTDataToFile(advChirpLUTCfgArgs);
        if(retVal != RL_RET_CODE_OK)
        {
            goto end;
        }
    }

    /* Control comes here implies that the advance chirp parameters and LUT buffer was configured successfully */
    retVal = 0;

end:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure the mmWave link with the supplied
 *      configuration
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[in]  ptrControlCfg
 *      Pointer to the control configuration
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_configLink
(
    MMWave_MCB*         ptrMMWaveMCB,
    MMWave_CtrlCfg*     ptrControlCfg,
    int32_t*            errCode
)
{
    int32_t retVal;

    /* create heap for BPM. */
    HeapP_construct(&gBpmHeapObj, gBpmHeapMem, BPM_HEAP_MEM_SIZE);

#if defined (SOC_AWR2544)
    if(ptrControlCfg->u.frameCfg[0].frameCfg.triggerSelect == 0x3U)
    {
        /* Tigger select is configured as CPTS */
        MMWave_frametrigSelCPTS();

        /* configure frameCfg triggerSelect as HW TRIGGER */
        ptrControlCfg->u.frameCfg[0].frameCfg.triggerSelect = 0x2U;
    }
#endif

    /* Determine the DFE Output mode? */
    switch (ptrControlCfg->dfeDataOutputMode)
    {
        case MMWave_DFEDataOutputMode_FRAME:
        {
            /**************************************************************************
             * Frame Mode:
             * Order of operations as specified by the mmWave Link are
             *  - Profile configuration
             *  - Chirp configuration
             *  - Frame configuration
             **************************************************************************/
            retVal = MMWave_configureProfileChirp(ptrControlCfg, errCode);

            if(retVal < 0)
            {
                goto end;
            }

            /* BPM configuration is for all devices. */
            retVal = MMWave_configBPM(ptrMMWaveMCB, ptrControlCfg, errCode);

            if(retVal < 0)
            {
                retVal = -1;
                goto end;
            }

            /* Phase Shifter configuration is for all devices. */
            retVal = MMWave_configPhaseShift(ptrMMWaveMCB, ptrControlCfg, errCode);

            if(retVal < 0)
            {
                retVal = -1;
                goto end;
            }


            /* Set the frame configuration: */
            DebugP_logInfo ("rlSetFrameConfig...\n");
            retVal = rlSetFrameConfig(ptrMMWaveMCB->deviceMap, &ptrControlCfg->u.frameCfg[0].frameCfg);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Setting the frame configuration failed */
                DebugP_logInfo("rlSetFrameConfig error %d\n",retVal);

                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EFRAMECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }

            break;
        }
        case MMWave_DFEDataOutputMode_CONTINUOUS:
        {
            /**************************************************************************
             * Continuous Mode:
             **************************************************************************/
            /* Master configuration is applied to all devices */
            retVal = rlSetContModeConfig (ptrMMWaveMCB->deviceMap, &ptrControlCfg->u.continuousModeCfg[0].cfg);
            if (retVal != RL_RET_CODE_OK)
            {
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECONTMODECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
            break;
        }
        case MMWave_DFEDataOutputMode_ADVANCED_FRAME:
        {

            /**************************************************************************
             * Advanced Frame Configuration:
             * Order of operations as specified by the mmWave Link are
             *  - Profile configuration
             *  - Chirp configuration
             *  - Advanced Frame configuration
             **************************************************************************/
            retVal = MMWave_configureProfileChirp(ptrControlCfg, errCode);

            if (retVal < 0)
            {
                goto end;
            }

            /* BPM configuration is for all devices. */
            retVal = MMWave_configBPM(ptrMMWaveMCB, ptrControlCfg, errCode);

            if(retVal < 0)
            {
                retVal = -1;
                goto end;
            }

            /* Phase Shifter configuration is for all devices. */
            retVal = MMWave_configPhaseShift(ptrMMWaveMCB, ptrControlCfg, errCode);

            if(retVal < 0)
            {
                retVal = -1;
                goto end;
            }


            /* Set the advanced frame configuration: */
            retVal = rlSetAdvFrameConfig(ptrMMWaveMCB->deviceMap, &ptrControlCfg->u.advancedFrameCfg[0].frameCfg);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Setting the frame configuration failed */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EFRAMECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
            break;
        }
        case MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME:
        {
            /**************************************************************************
             * Advanced Chirp with normal frame Configuration:
             * Order of operations as specified by the mmWave Link are
             *  - Profile configuration
             *  - Advanced Chirp configuration
             *  - LUT Configuration
             *  - Frame configuration
             **************************************************************************/
            retVal = MMWave_configureProfile(ptrControlCfg, errCode);
            if(retVal < 0)
            {
                goto end;
            }

            retVal= MMWave_configureAdvChirp(ptrMMWaveMCB, errCode);
            if(retVal<0)
            {
                goto end;
            }


            /* Set the frame configuration: */
            DebugP_logInfo ("rlSetFrameConfig...\n");
            retVal = rlSetFrameConfig(ptrMMWaveMCB->deviceMap, &ptrControlCfg->u.frameCfg[0].frameCfg);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Setting the frame configuration failed */
                DebugP_logInfo("rlSetFrameConfig error %d\n",retVal);

                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EFRAMECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
            break;
        }
        case MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME:
        {
            /**************************************************************************
             * Advanced Chirp with advanced frame Configuration:
             * Order of operations as specified by the mmWave Link are
             *  - Profile configuration
             *  - Advanced Chirp configuration
             *  - LUT Configuration
             *  - Advanced Frame configuration
             **************************************************************************/
            retVal = MMWave_configureProfile(ptrControlCfg, errCode);
            if(retVal < 0)
            {
                goto end;
            }

            retVal= MMWave_configureAdvChirp(ptrMMWaveMCB, errCode);
            if(retVal<0)
            {
                goto end;
            }


            /* Set the advanced frame configuration: */
            retVal = rlSetAdvFrameConfig(ptrMMWaveMCB->deviceMap, &ptrControlCfg->u.advancedFrameCfg[0].frameCfg);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Setting the frame configuration failed */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EFRAMECFG, retVal);
                retVal   = MINUS_ONE;
                goto end;
            }
            break;
        }
        default:
        {
            /* Error: This should never occur and the user seems to have ignored a warning. */
            *errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
            retVal   = MINUS_ONE;
            goto end;
        }
    }

    /* Set the return value to be success. */
    retVal = 0;

end:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to start the mmWave link. This function can only be
 *      invoked once the configuration has been completed successfully.
 *        - Configures run-time calibration based on user input.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_startLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    rlRunTimeCalibConf_t    runTimeCalib;
    rlContModeEn_t          contModeEnable;

    int32_t  retVal;
    uint32_t u32DevIdx = 0;
    uint32_t  MasterDevIdx = 0;
    uint32_t  u8DevMap = 0;

    /* Sensor is being started; reset the link status to get rid of the calibration flags. */
    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        ptrMMWaveMCB->linkStatus[u32DevIdx] &= ~ (uint32_t) MMWAVE_RF_CALIBRATION_DONE;
        ptrMMWaveMCB->linkStatus[u32DevIdx] &= ~ (uint32_t) MMWAVE_RF_CALIBRATION_FAILED;
    }

    /* Initialize the runtime calibration configuration: */
    memset ((void *)&runTimeCalib, 0, sizeof(rlRunTimeCalibConf_t));

    /* Enable calibration: */
    runTimeCalib.reportEn           = ptrMMWaveMCB->calibrationCfg.u.chirpCalibrationCfg.reportEn;
    runTimeCalib.oneTimeCalibEnMask = CSL_FMKR (4U,  4U,  1U) | /* LODIST calibration   */
                                      CSL_FMKR (8U,  8U,  1U) | /* PD calibration */
                                      CSL_FMKR (9U,  9U,  1U) | /* TX Power calibration */
                                      CSL_FMKR (10U, 10U, 1U) | /* RX gain calibration  */
                                      CSL_FMKR (11U, 11U, 1U);  /* TX Phase Shifter Calibration */

    /* Are we operating in Chirp or Continuous mode? */
    if ((ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_FRAME) ||
        (ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_FRAME)||
        (ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME)||
        (ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        /******************************************************************************
         * FRAME Mode: Is calibration enabled?
         ******************************************************************************/
        if (ptrMMWaveMCB->calibrationCfg.u.chirpCalibrationCfg.enableCalibration == true)
        {
            /* Do we need to configure periodic calibration? */
            if (ptrMMWaveMCB->calibrationCfg.u.chirpCalibrationCfg.enablePeriodicity == true)
            {
                /* YES: Enable all calibrations with the configured periodicity */
                runTimeCalib.periodicCalibEnMask = runTimeCalib.oneTimeCalibEnMask;
                runTimeCalib.calibPeriodicity    = ptrMMWaveMCB->calibrationCfg.u.chirpCalibrationCfg.periodicTimeInFrames;
            }

            /* Configure the calibrations: */
            retVal = rlRfRunTimeCalibConfig(ptrMMWaveMCB->deviceMap, &runTimeCalib);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Runtime calibration failed */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALTRIG, retVal);

                /* Setup the return value to indicate an error. */
                retVal   = MINUS_ONE;
                goto exit;
            }

            /* Check if application requested for checking run-time calibration report. */
            if(ptrMMWaveMCB->calibrationCfg.u.chirpCalibrationCfg.reportEn == 1)
            {
                /***************************************************************************
                 * SYNCHRONIZATION: We need to loop around till the BSS has completed the
                 * ONE SHOT calibration [Done or Failed]. The application would have
                 * received an asynchronous event through the callback function about the
                 * failure.
                 ***************************************************************************/
                MMWave_internalLinkSync(ptrMMWaveMCB, MMWAVE_RF_CALIBRATION_DONE | MMWAVE_RF_CALIBRATION_FAILED);

                /* Did the calibration fail? */
                for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
                {
                    if (ptrMMWaveMCB->linkStatus[u32DevIdx] & MMWAVE_RF_CALIBRATION_FAILED)
                    {
                        /* YES: Calibration failed. Abort and don't start the sensor. */
                        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALFAIL, 0);

                        /* Setup the return value to indicate an error. */
                        retVal   = MINUS_ONE;
                        goto exit;
                    }
                }
            }
        }

        /* Start all Slave devices */
        u8DevMap = (uint8_t) ((uint8_t) 1U << (uint8_t) MasterDevIdx);
        u8DevMap = ptrMMWaveMCB->deviceMap & ~u8DevMap;

        if(u8DevMap)
        {
            retVal = rlSensorStart(u8DevMap);

            if(retVal != RL_RET_CODE_OK)
            {
                /* Error: Starting the sensor failed */
                * errCode = MMWave_encodeError(MMWave_ErrorLevel_ERROR, MMWAVE_ESENSOR, retVal);

                retVal = MINUS_ONE;
                goto exit;
            }
            ClockP_usleep(1 * 1000);
        }

#ifdef SOC_AWR2544
        /* Reset the RSS Bookkeeping registers required for application header value */
        CSL_rss_ctrlRegs *ptrRssCtrlregs = (CSL_rss_ctrlRegs *)CSL_RSS_CTRL_U_BASE;

        /* Reset the RSS_BOOKKEEPING_SEQ_NUM */
        CSL_FINS(ptrRssCtrlregs->RSS_BOOKKEEPING_CTRL, RSS_CTRL_RSS_BOOKKEEPING_CTRL_RSS_BOOKKEEPING_CTRL_SEQ_NUM_RST, 1);

        /* Reset the RSS_BOOKKEEPING_FRM_CNT */
        CSL_FINS(ptrRssCtrlregs->RSS_BOOKKEEPING_CTRL, RSS_CTRL_RSS_BOOKKEEPING_CTRL_RSS_BOOKKEEPING_CTRL_FRM_CNT_RST, 1);

        /* Reset the RSS_BOOKKEEPING_CHRP_CNT */
        CSL_FINS(ptrRssCtrlregs->RSS_BOOKKEEPING_CTRL, RSS_CTRL_RSS_BOOKKEEPING_CTRL_RSS_BOOKKEEPING_CTRL_CHRP_CNT_RST, 1);

        /* Write 0: RSS_BOOKKEEPING_FRM_CNT is incremented on every FRAME_START */
        CSL_FINS(ptrRssCtrlregs->RSS_BOOKKEEPING_CTRL, RSS_CTRL_RSS_BOOKKEEPING_CTRL_RSS_BOOKKEEPING_CTRL_FRM_CNT_TRIG_SRC, 0);

        /* Write 0: RSS_BOOKKEEPING_CHRP_CNT is incremented on every CHIRP_START */
        CSL_FINS(ptrRssCtrlregs->RSS_BOOKKEEPING_CTRL, RSS_CTRL_RSS_BOOKKEEPING_CTRL_RSS_BOOKKEEPING_CTRL_CHRP_CNT_TRIG_SRC, 0);

        /* Write 0: RSS_BOOKKEEPING_CHRP_CNT is reset on every FRAME_START */
        CSL_FINS(ptrRssCtrlregs->RSS_BOOKKEEPING_CTRL, RSS_CTRL_RSS_BOOKKEEPING_CTRL_RSS_BOOKKEEPING_CTRL_CHRP_CNT_RST_SRC, 0);

        /* Set the Magic Number in RSS_APP_GP */
        CSL_FINS(ptrRssCtrlregs->RSS_APP_GP, RSS_CTRL_RSS_APP_GP_RSS_APP_GP_REG, 0x01234567);
#endif

        /* Start Master device */
        u8DevMap = (uint8_t) ((uint8_t) 1U << (uint8_t) MasterDevIdx);

        retVal = rlSensorStart(u8DevMap);

        if (retVal != RL_RET_CODE_OK)
        {
            /* Error: Starting the sensor failed */
            * errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ESENSOR, retVal);

            retVal = MINUS_ONE;
            goto exit;
        }
    }
    else
    {
        /******************************************************************************
         * Continuous Mode: Is calibration enabled?
         ******************************************************************************/
        if (ptrMMWaveMCB->calibrationCfg.u.contCalibrationCfg.enableOneShotCalibration == true)
        {
            runTimeCalib.reportEn           = ptrMMWaveMCB->calibrationCfg.u.contCalibrationCfg.reportEn;

            /* Configure the calibrations: */
            retVal = rlRfRunTimeCalibConfig(ptrMMWaveMCB->deviceMap, &runTimeCalib);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Runtime calibration failed */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALTRIG, retVal);

                /* Setup the return value to indicate an error. */
                retVal   = MINUS_ONE;
                goto exit;
            }

            /* Check if application requested for checking run-time calibration report. */
            if(ptrMMWaveMCB->calibrationCfg.u.contCalibrationCfg.reportEn == 1)
            {
                /***************************************************************************
                 * SYNCHRONIZATION: We need to loop around till the BSS has completed the
                 * ONE SHOT calibration [Done or Failed]. The application would have
                 * received an asynchronous event through the callback function about the
                 * failure.
                 ***************************************************************************/
                MMWave_internalLinkSync(ptrMMWaveMCB, MMWAVE_RF_CALIBRATION_DONE | MMWAVE_RF_CALIBRATION_FAILED);

                /* Did the calibration fail? */
                for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
                {
                    if (ptrMMWaveMCB->linkStatus[u32DevIdx] & MMWAVE_RF_CALIBRATION_FAILED)
                    {
                        /* YES: Calibration failed. Abort and don't start enable the continuous mode. */
                        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALFAIL, 0);

                        /* Setup the return value to indicate an error. */
                        retVal   = MINUS_ONE;
                        goto exit;
                    }
                }
            }
        }

        /* Start the sensor in continuous mode: */
        memset ((void*)&contModeEnable, 0, sizeof(rlContModeEn_t));

        /* Populate the continuous mode configuration: */
        contModeEnable.contModeEn = 1U;
        retVal = rlEnableContMode (ptrMMWaveMCB->deviceMap, &contModeEnable);
        if (retVal != RL_RET_CODE_OK)
        {
            /* Error: Unable to enable the continuous mode */
            *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECONTMODE, retVal);

            /* Setup the return value to indicate an error. */
            retVal   = MINUS_ONE;
            goto exit;
        }
    }

    /* Control comes here indicates that the sensor has been started successfully */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to stop the mmWave link.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_stopLink (const MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t         retVal;
    rlContModeEn_t  contModeEnable;
    uint32_t        MasterDevIdx = 0;
    uint32_t        DevMap = 0;

    /* Are we operating in Chirp or Continuous mode? */
    if ((ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_FRAME) ||
        (ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_FRAME)||
        (ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME)||
        (ptrMMWaveMCB->calibrationCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        /******************************************************************************
         * CHIRP or ADVANCED: Stop the sensor
         ******************************************************************************/
        /* Stop Master device */
        DevMap = (uint8_t) ((uint8_t) 1U << (uint8_t) MasterDevIdx);

        retVal = rlSensorStop(DevMap);

        if (retVal != RL_RET_CODE_OK)
        {
            /* Even if there is an error, try to stop Slave devices as well */
            if (retVal == (int32_t)RL_RET_CODE_FRAME_ALREADY_ENDED)
            {
                /* Special Case: This is treated as a warning */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_WARNING, MMWAVE_ESENSOR, retVal);
                retVal    = 0;
            }
            else
            {
                /* All other are treated as FATAL error */
                * errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ESENSOR, retVal);
                retVal    = MINUS_ONE;
            }
        }

        /* Stop all Slave devices */
        DevMap = ptrMMWaveMCB->deviceMap & ~DevMap;

        if(DevMap)
        {
            retVal = rlSensorStop(DevMap);

            if (retVal != RL_RET_CODE_OK)
            {
                /* Even if there is an error, try to stop Slave devices as well */
                if (retVal == (int32_t) RL_RET_CODE_FRAME_ALREADY_ENDED)
                {
                    /* Special Case: This is treated as a warning */
                    * errCode = MMWave_encodeError (MMWave_ErrorLevel_WARNING, MMWAVE_ESENSOR, retVal);
                    retVal    = 0;
                }
                else
                {
                    /* All other are treated as FATAL error */
                    * errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ESENSOR, retVal);
                    retVal    = MINUS_ONE;
                }
            }
        }
    }
    else
    {
        /******************************************************************************
         * Continuous Mode: Disable continuous mode
         ******************************************************************************/
        memset ((void*)&contModeEnable, 0, sizeof(rlContModeEn_t));

        /* Populate the continuous mode configuration: */
        contModeEnable.contModeEn = 0U;
        retVal = rlEnableContMode (ptrMMWaveMCB->deviceMap, &contModeEnable);
        if (retVal != RL_RET_CODE_OK)
        {
            /* Error: Unable to disable the continuous mode */
            *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECONTMODE, retVal);
            retVal   = MINUS_ONE;
            goto exit;
        }
        else
        {
            /* Successfully stopped the sensor: */
            retVal = 0;
        }
    }

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to execute the mmWave link.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_executeLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    MMWave_SpawnFxnNode*    ptrSpawnFxnNode;
    uintptr_t               key;

    /* Semaphore has been posted; process any active jobs in the spawn active list.  */
    while (1)
    {
        /* Critical Section: The spawn active list is a critical resource which is accessed
         * from multiple contexts */

         DebugP_logInfo("MMWave_executeLink: will get  node from active list\n");

        key = HwiP_disable();
        ptrSpawnFxnNode = (MMWave_SpawnFxnNode*)MMWave_listRemove((MMWave_ListNode**)&ptrMMWaveMCB->ptrSpawnFxnActiveList);
        HwiP_restore (key);

        /* Is there an active entry to be processed */
        if (ptrSpawnFxnNode == NULL)
        {
            /* No other spawn nodes to be processed; we are done break out of the loop*/
            DebugP_logInfo("MMWave_executeLink:No other spawn nodes to be processed\n");
            break;
        }
        else
        {
            /* Execute the spawn function */
            DebugP_logInfo("MMWave_executeLink: Execute the spawn function 0x%x\n",(uintptr_t)(ptrSpawnFxnNode->spawnEntry));
            ptrSpawnFxnNode->spawnEntry (ptrSpawnFxnNode->arg);
            DebugP_logInfo("MMWave_executeLink: Finished executing the spawn function 0x%x\n",(uintptr_t)(ptrSpawnFxnNode->spawnEntry));

            /* Critical Section: The spawn free list is a critical resource which is accessed
             * from multiple contexts */
            key = HwiP_disable();
            MMWave_listAdd ((MMWave_ListNode**)&ptrMMWaveMCB->ptrSpawnFxnFreeList, (MMWave_ListNode*)ptrSpawnFxnNode);
            HwiP_restore (key);
        }
    }
    return 0;
}


/**
 *  @b Description
 *  @n
 *      The function is used to close the mmWave Link module.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_closeLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    uint32_t u32DevIdx;

    /* Link is not operational: */
    for(u32DevIdx = 0U; u32DevIdx < MMWAVE_RADAR_DEVICES; u32DevIdx++)
    {
        ptrMMWaveMCB->linkStatus[u32DevIdx] = 0U;
    }
    return 0;
}



/**
 *  @b Description
 *  @n
 *      The function is used to get the version of the various components
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_deviceGetVersion(MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t retVal = MINUS_ONE;

    retVal = rlDeviceGetVersion(ptrMMWaveMCB->deviceMap, &ptrMMWaveMCB->version);

    DebugP_logInfo ("RF H/W Version    : %02d.%02d\n",
               ptrMMWaveMCB->version.rf.hwMajor, ptrMMWaveMCB->version.rf.hwMinor);
    DebugP_logInfo ("RF F/W Version    : %02d.%02d.%02d.%02d\n",
               ptrMMWaveMCB->version.rf.fwMajor, ptrMMWaveMCB->version.rf.fwMinor,
               ptrMMWaveMCB->version.rf.fwBuild, ptrMMWaveMCB->version.rf.fwDebug);
    DebugP_logInfo ("RF F/W Version continuing    : %02d.%02d.%02d\n",
                 ptrMMWaveMCB->version.rf.fwYear, ptrMMWaveMCB->version.rf.fwMonth, ptrMMWaveMCB->version.rf.fwDay);

    DebugP_logInfo ("mmWaveLink Version: %02d.%02d.%02d.%02d\n",
                  ptrMMWaveMCB->version.mmWaveLink.major, ptrMMWaveMCB->version.mmWaveLink.minor,
                  ptrMMWaveMCB->version.mmWaveLink.build, ptrMMWaveMCB->version.mmWaveLink.debug);
    DebugP_logInfo ("mmWaveLink Version continuing: %02d.%02d.%02d\n",
                 ptrMMWaveMCB->version.mmWaveLink.day,ptrMMWaveMCB->version.mmWaveLink.month,
                 ptrMMWaveMCB->version.mmWaveLink.year);

    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Unable to get the device version */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EVERSION, retVal);
        DebugP_logInfo("Error MMWave_deviceGetVersion\n");
        retVal   = MINUS_ONE;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to deinitialize the mmWave Link
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MMWave_deinitMMWaveLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t     retVal;

    /* Power off the Device: */
    retVal = (int32_t)rlDevicePowerOff();
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Unable to power off the BSS */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EDEINIT, retVal);
        retVal   = MINUS_ONE;
    }
    return retVal;
}

#if defined (SOC_AWR2544)
/**
 *  @b Description
 *  @n
 *      The function performs a sequence of register writes
 *      to configure frame trigger source as CPTS_GENF0
 *
 *  RSS_PROC_CTRL                               MSS_CTRL.MSS_DMM_EVENT6_REG
 * FRCCFG2.FRCCFG2_FRCSYNCINSEL    mss_dmm_event_sel [25]    mss_dmm_event_sel[26]    selected event
 *          1                      1                         don't care               rti_interrupts_to_rss[0]
 *          1                      0                         1                        frc_triggers[4] (cpts_comp)
 *          1                      0                         0                        frc_triggers[6](cpts_genf1)
 *          2                      1                         don’t care	              rti_interrupts_to_rss[1]
 *          2                      0                         1                        frc_triggers[5] (cpts_genf0)
 *          2                      0                         0                        frc_triggers[7](cpts_genf2)
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 */
static int32_t MMWave_frametrigSelCPTS(void)
{
    int32_t retVal = SystemP_SUCCESS;
    CSL_rss_proc_ctrlRegs *ptrRssProcCtrlRegs = (CSL_rss_proc_ctrlRegs *) CSL_RSS_PROC_CTRL_U_BASE;
    CSL_mss_ctrlRegs *ptrMssCtrlRegs = (CSL_mss_ctrlRegs *) CSL_MSS_CTRL_U_BASE;

    /* Select external syncin instead of software syncin - Multibit write */
    CSL_FINS(ptrRssProcCtrlRegs->FRCCFG, RSS_PROC_CTRL_FRCCFG_FRCCFG_FRCSYNCINSRCSEL, 7U);

    /* Select RTID comparison interrupt0 as the source */
    CSL_FINS(ptrRssProcCtrlRegs->FRCCFG2, RSS_PROC_CTRL_FRCCFG2_FRCCFG2_FRCSYNCINSEL, 2U);

    /* selects the CPTS_GENF0 as the trigger to start the frame */
    CSL_FINS(ptrMssCtrlRegs->MSS_DMM_EVENT6_REG, MSS_CTRL_MSS_DMM_EVENT6_REG_MSS_DMM_EVENT6_REG_EVENT_SEL25, 0U);
    CSL_FINS(ptrMssCtrlRegs->MSS_DMM_EVENT6_REG, MSS_CTRL_MSS_DMM_EVENT6_REG_MSS_DMM_EVENT6_REG_EVENT_SEL26, 1U);


    return retVal;
}
#endif

/**
 *  @b Description
 *  @n
 *      The function is used to deinitialize the mmWave link.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_deinitLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t                 retVal = 0;
    MMWave_SpawnFxnNode*    ptrSpawnFxnNode;
    uintptr_t               key;

    /* Deinitialize the mmWave Link: */
    retVal = MMWave_deinitMMWaveLink (ptrMMWaveMCB, errCode);
    if (retVal < 0)
    {
        /* Error: Unable to deinitialize the mmWave link; error code is already setup */
        goto exit;
    }

    /* Deinitialize the CRC channel: */
    retVal = MMWave_deinitCRC (ptrMMWaveMCB, errCode);
    if (retVal < 0)
    {
        /* Error: Unable to deinitialize the CRC channel; error code is already setup */
        goto exit;
    }

    /* Cycle through and cleanup the active spawn lists: There might be some entries in
     * the Active list which still need to be handled but because we are shutting down
     * the module we simply flush out the entries. */
    key = HwiP_disable();
    ptrSpawnFxnNode = (MMWave_SpawnFxnNode*)MMWave_listRemove ((MMWave_ListNode**)&gMMWave_MCB.ptrSpawnFxnActiveList);
    while (ptrSpawnFxnNode != NULL)
    {
        /* Add this back to the free list: */
        MMWave_listAdd ((MMWave_ListNode**)&ptrMMWaveMCB->ptrSpawnFxnFreeList, (MMWave_ListNode*)ptrSpawnFxnNode);

        /* Get the next entry from the active list: */
        ptrSpawnFxnNode = (MMWave_SpawnFxnNode*)MMWave_listRemove ((MMWave_ListNode**)&gMMWave_MCB.ptrSpawnFxnActiveList);
    }
    HwiP_restore (key);

    /* Control comes here implies that the deinitialization of the module was successful. */
    retVal = 0;

exit:
    return retVal;
}
