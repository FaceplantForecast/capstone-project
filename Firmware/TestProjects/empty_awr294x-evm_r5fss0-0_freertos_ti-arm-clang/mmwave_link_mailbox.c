/**
 *   @file  mmwave_link_mailbox.c
 *
 *   @brief
 *      The file implements the mmWave control module which interfaces
 *      with the mmWave Link API and communicates with the BSS
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
#include <stdint.h>
#include <string.h>

#include <drivers/hw_include/hw_types.h>
#include <drivers/hw_include/cslr_soc.h>
#include <kernel/dpl/ClockP.h>

#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/common/syscommon.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/mmwave/mmwave.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/mmwave/include/mmwave_internal.h>

/**************************************************************************
 ***************************** Local Functions ****************************
 **************************************************************************/
static void MMWave_mboxCallbackFxn (uint32_t remoteCoreId, void *args);
static rlComIfHdl_t MMWave_mboxOpen(rlUInt8_t deviceIndex, uint32_t flags);
static int32_t MMWave_mboxRead(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len);
static int32_t MMWave_mboxWrite(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len);
static int32_t MMWave_mboxClose(rlComIfHdl_t fd);
static rlInt32_t MMWave_enableDevice(rlUInt8_t deviceIndex);
static rlInt32_t MMWave_disableDevice(rlUInt8_t deviceIndex);
static void MMWave_maskHostIRQ(rlComIfHdl_t fd);
static void MMWave_unmaskHostIRQ(rlComIfHdl_t fd);
static rlInt32_t MMWave_waitIRQStatus(rlComIfHdl_t fd, rlUInt8_t Level);
static rlInt32_t MMWave_registerInterruptHandler(rlUInt8_t deviceIndex, RL_P_EVENT_HANDLER pHandler, void* pValue);
static int32_t MMWave_initMMWaveLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode);

uint32_t mailbox_dummy_handle = 0;
/**************************************************************************
 ************************ mmWave Link Functions ***************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      This is the registered callback function which is invoked when the
 *      Mailbox receives data.
 *
 *  @param[in]  remoteCoreId
 *      Handle to the Mailbox on which data was received
 *  @param[in]  args
 *      arguments to mailbox callback function
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_mboxCallbackFxn (uint32_t remoteCoreId, void *args)
{
    /* Indicate to the Radar Link that a message has been received. */
    gMMWave_MCB.mmwavelinkInterruptFunc(0, NULL);
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to open the communication
 *      interface channel
 *
 *  @param[in]  deviceIndex
 *      Device index for which the communication channel to be opened
 *  @param[in]  flags
 *      Flags which are passed to configure the interface
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - Communicate Interface Channel Handle
 *  @retval
 *      Error   - NULL
 */
static rlComIfHdl_t MMWave_mboxOpen(rlUInt8_t deviceIndex, uint32_t flags)
{

   /*Developer notes: DSS mailbox memory needs to be initialized.
    * This needs to be taken care in SBL.
    * MSS mailbox memory is initialized bu IPC driver. implies not required
    * to initialized.
    */
#ifdef SUBSYS_DSS
    /* Clear init Done status. */
    HW_WR_REG32((CSL_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_MAILBOX_MEMINIT_DONE), 1U);

    /* Initialize the memory for the mailbox */
    HW_WR_REG32((CSL_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_MAILBOX_MEMINIT_START), 1U);

    while ((HW_RD_REG32(CSL_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_MAILBOX_MEMINIT_DONE) & 0x01U) != 0x01)
    {
        /* Wait till init is complete. */
    }

    /* Clear init Done status. */
    HW_WR_REG32((CSL_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_MAILBOX_MEMINIT_DONE), 1U);
#endif

    Mailbox_setReadCallback(MMWave_mboxCallbackFxn, NULL);

    return (rlComIfHdl_t)&mailbox_dummy_handle;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to close the communication
 *      interface channel
 *
 *  @param[in]  fd
 *      Opaque Handle to the communication channel which is to be closed
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static int32_t MMWave_mboxClose(rlComIfHdl_t fd)
{
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to read data from the
 *      communication interface channel
 *
 *  @param[in]  fd
 *      Opaque Handle to the communication channel
 *  @param[out]  pBuff
 *      Pointer to the data buffer which is populated with the received message
 *  @param[in]  len
 *      Size of the data buffer
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - Number of bytes of received data
 *  @retval
 *      Error   - <0
 */
static int32_t MMWave_mboxRead(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len)
{
    int32_t status;

    status = Mailbox_read(CSL_CORE_ID_RSS_R4, pBuff, len, SystemP_WAIT_FOREVER);

    if (status != SystemP_SUCCESS)
    {
        DebugP_logInfo("Mailbox Read Error. \n");
    }
    else
    {
        status = len;
    }

    return status;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to write data to the
 *      communication interface channel
 *
 *  @param[in]  fd
 *      Opaque Handle to the communication channel
 *  @param[in]  pBuff
 *      Pointer to the data buffer which is to be written out
 *  @param[in]  len
 *      Size of the data buffer
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - Number of bytes of data which is sent
 *  @retval
 *      Error   - <0
 */
static int32_t MMWave_mboxWrite(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len)
{
    int32_t status;

    status = Mailbox_write(CSL_CORE_ID_RSS_R4, pBuff, len, SystemP_WAIT_FOREVER);

    if (status != SystemP_SUCCESS)
    {
        DebugP_logInfo("Mailbox Write Error. \n");
    }
    else
    {
        status = len;
    }

    return status;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to power on and enable
 *      the XWR1XX device
 *
 *  @param[in]  deviceIndex
 *      Device index which is to be enabled
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MMWave_enableDevice(rlUInt8_t deviceIndex)
{
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to power off and disable
 *      the XWR1XX device
 *
 *  @param[in]  deviceIndex
 *      Device index which is to be disabled
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MMWave_disableDevice(rlUInt8_t deviceIndex)
{
    /* Note: halting of the BSS is not currently supported */
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to mask the
 *      interrupts. Since the mmWave control module is using the Mailbox
 *      communication interface the driver is handling all the interrupt
 *      management. This function is a dummy stub.
 *
 *  @param[in]  fd
 *      Handle to the communication interface
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_maskHostIRQ(rlComIfHdl_t fd)
{
    return;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to unmask the
 *      interrupts to indicate that the message has been successfully handled
 *
 *  @param[in]  fd
 *      Handle to the communication interface
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_unmaskHostIRQ(rlComIfHdl_t fd)
{
    /* The Mailbox driver indicates to the remote endpoint that the message
     * have been processed. The mmWave is now capable of receiving another message */
    Mailbox_readDone(CSL_CORE_ID_RSS_R4);
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to poll the
 *      host interrupt status. In the case of the Mailbox the driver is
 *      responsible for the interrupt management. This function is a
 *      dummy stub
 *
 *  @param[in]  fd
 *      Handle to the communication interface
 *  @param[in]  Level
 *      Wait for IRQ Level(high/low)
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Always returns 0 indicating that the IRQ Line is low
 */
static rlInt32_t MMWave_waitIRQStatus(rlComIfHdl_t fd, rlUInt8_t Level)
{
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave link registered callback function to register the
 *      interrupt handler. In the case of the Mailbox the driver is responsible
 *      for the interrupt management. This function is a dummy stub
 *
 *  @param[in]  deviceIndex
 *      Device for which the interrupt is to be registered
 *  @param[in]  pHandler
 *      ISR Handler
 *  @param[in]  pValue
 *      Argument to the ISR
 *
 *  \ingroup  MMWAVE_INTERNAL_FUNCTION
 *
 *  @retval
 *      Always returns 0
 */
static rlInt32_t MMWave_registerInterruptHandler
(
    rlUInt8_t           deviceIndex,
    RL_P_EVENT_HANDLER  pHandler,
    void*               pValue
)
{
    gMMWave_MCB.mmwavelinkInterruptFunc = pHandler;
    return 0;
}



/**
 *  @b Description
 *  @n
 *      The function is used to initialize the mmWave Link
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
static int32_t MMWave_initMMWaveLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    rlClientCbs_t   RlApp_ClientCtx;
    int32_t         retVal = MINUS_ONE;

    /* Initialize the link status: */
    ptrMMWaveMCB->linkStatus[0] = 0U;

    /* Reset the client context: */
    memset ((void *)&RlApp_ClientCtx, 0, sizeof(rlClientCbs_t));

    /* Populate the mmWave link context configuration: */
    RlApp_ClientCtx.ackTimeout = MMWAVE_ACK_TIMEOUT;

    /* Are we allowed to use the CRC Driver? */
    if (ptrMMWaveMCB->initCfg.linkCRCCfg.useCRCDriver == 1U)
    {
        /* YES: Use the 16bit CRC */
        RlApp_ClientCtx.crcType = RL_CRC_TYPE_16BIT_CCITT;
    }
    else
    {
        /* NO: No CRC is used. */
        RlApp_ClientCtx.crcType = RL_CRC_TYPE_NO_CRC;
    }

    /* Setup the platform on which the mmWave Link executes */
    #ifdef SUBSYS_MSS
    RlApp_ClientCtx.platform = RL_PLATFORM_MSS;
    #endif

    #ifdef SUBSYS_DSS
    RlApp_ClientCtx.platform = RL_PLATFORM_DSS;
    #endif
    RlApp_ClientCtx.arDevType = RL_AR_DEVICETYPE_22XX;

    /* Initialize the Communication Interface API: */
    RlApp_ClientCtx.comIfCb.rlComIfOpen  = &MMWave_mboxOpen;
    RlApp_ClientCtx.comIfCb.rlComIfClose = &MMWave_mboxClose;
    RlApp_ClientCtx.comIfCb.rlComIfRead  = &MMWave_mboxRead;
    RlApp_ClientCtx.comIfCb.rlComIfWrite = &MMWave_mboxWrite;

    /* Initialize OSI Mutex Interface */
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexCreate = &MMWave_osalMutexCreate;
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexLock   = &MMWave_osalMutexLock;
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexUnLock = &MMWave_osalMutexUnlock;
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexDelete = &MMWave_osalMutexDelete;

    /* Initialize OSI Semaphore Interface */
    RlApp_ClientCtx.osiCb.sem.rlOsiSemCreate = &MMWave_osalSemCreate;
    RlApp_ClientCtx.osiCb.sem.rlOsiSemWait   = &MMWave_osalSemWait;
    RlApp_ClientCtx.osiCb.sem.rlOsiSemSignal = &MMWave_osalSemSignal;
    RlApp_ClientCtx.osiCb.sem.rlOsiSemDelete = &MMWave_osalSemDelete;

    /* Initialize OSI Queue Interface */
    RlApp_ClientCtx.osiCb.queue.rlOsiSpawn = &MMWave_spawn;

    /* Initialize OSI Timer Interface */
    RlApp_ClientCtx.timerCb.rlDelay = NULL;

    /* Initialize the CRC Interface */
    RlApp_ClientCtx.crcCb.rlComputeCRC = &MMWave_computeCRC;

    /* Initialize Device Control Interface */
    RlApp_ClientCtx.devCtrlCb.rlDeviceDisable            = &MMWave_disableDevice;
    RlApp_ClientCtx.devCtrlCb.rlDeviceEnable             = &MMWave_enableDevice;
    RlApp_ClientCtx.devCtrlCb.rlDeviceMaskHostIrq        = &MMWave_maskHostIRQ;
    RlApp_ClientCtx.devCtrlCb.rlDeviceUnMaskHostIrq      = &MMWave_unmaskHostIRQ;
    RlApp_ClientCtx.devCtrlCb.rlDeviceWaitIrqStatus      = &MMWave_waitIRQStatus;
    RlApp_ClientCtx.devCtrlCb.rlRegisterInterruptHandler = &MMWave_registerInterruptHandler;

    /* Initialize the Asynchronous Event Handler: */
    RlApp_ClientCtx.eventCb.rlAsyncEvent = &MMWave_asyncEventHandler;

    /* Power on the Device: */
    retVal = (int32_t)rlDevicePowerOn(1U, RlApp_ClientCtx);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Unable to initialize and power on the BSS. Encode the error code to account
         * for the subsystem error code. */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINIT, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /* Check for BSS powerup */
    /* Check if BSS powerup is done */
    /* Bit 18 - The bit will be set once the BSS boots up before entering the idle task loop */
    if((CSL_REG32_RD(CSL_RSS_PROC_CTRL_U_BASE + CSL_RSS_PROC_CTRL_RSS_CR4_BOOT_INFO_REG0) & 0x00040000) == 0x00040000)
    {
        DebugP_log ("BSS is powered up...\r\n");
    }
    else
    {
        DebugP_log ("BSS is not powered up...\r\n");
        DebugP_assert(0);
    }

    /* Get the version information: */
    retVal = MMWave_deviceGetVersion(ptrMMWaveMCB, errCode);

    if(retVal != RL_RET_CODE_OK)
    {
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EVERSION, retVal);
        retVal = MINUS_ONE;
        goto exit;
    }

    /* Link has been setup successfully. */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to configure the asynchronous events. On Dual core
 *      either the DSS or the MSS could be configured to be the recepient of
 *      asynchronous events.
 *      It also configures if the start/stop asynchronous events are 
 *      enabled or disabled.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the mmWave control block
 *  @param[out]  errCode
 *      Error code populated on error
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_deviceCfgAsyncEvent(const MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t         retVal;
    rlRfDevCfg_t    asyncEventCfg;
    
    /* Sanity Check: Validate the arguments */
    if ((ptrMMWaveMCB->openCfg.defaultAsyncEventHandler != MMWave_DefaultAsyncEventHandler_MSS) &&
        (ptrMMWaveMCB->openCfg.defaultAsyncEventHandler != MMWave_DefaultAsyncEventHandler_DSS))
    {
        /* Error: Invalid argument. */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        retVal   = MINUS_ONE;
        goto exit;
    }
    
    /* Initialize the configuration: */
    memset ((void*)&asyncEventCfg, 0, sizeof(rlRfDevCfg_t));
    
    /* Which domain are we executing on? */
    if (ptrMMWaveMCB->openCfg.defaultAsyncEventHandler == MMWave_DefaultAsyncEventHandler_MSS)
    {
        /* MSS: */
        asyncEventCfg.aeDirection = 0U;
    }
    else
    {
        /* DSS: */
        asyncEventCfg.aeDirection = 0xAU;
    }
    
    /*Disable Frame Start async event? */
    if(ptrMMWaveMCB->openCfg.disableFrameStartAsyncEvent)
    {
        asyncEventCfg.aeControl |= 0x1U;
    }
    
    /*Disable Frame Stop async event? */
    if(ptrMMWaveMCB->openCfg.disableFrameStopAsyncEvent)
    {
        asyncEventCfg.aeControl |= 0x2U;
    }

    /* Set the configuration in the link: */
    retVal = rlRfSetDeviceCfg(RL_DEVICE_MAP_CASCADED_1, (rlRfDevCfg_t*)&asyncEventCfg);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Set the Async Event Direction Failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EASYNCEVENT, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /* Control comes here implies either of the following:-
     * - Asynchronous event direction was configured successfully
     * - We are not the default event handler.
     * Either case we report success */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to open the mmWave Link module.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the control MCB
 *  @param[in] ptrCalibrationData
 *      Optional pointer to the calibration data which needs to be
 *      restored.
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
int32_t MMWave_openLink 
(
    MMWave_MCB*                 ptrMMWaveMCB, 
    MMWave_CalibrationData*     ptrCalibrationData, 
    int32_t*                    errCode
)
{
    int32_t                     retVal = MINUS_ONE;
    rlRfCalMonFreqLimitConf_t   freqLimit;
    rlRfInitCalConf_t           rfInitCalib;
    rlRfCalMonTimeUntConf_t     timeCfg;

    /* Initialize the configurations: */
    memset ((void *)&freqLimit,   0, sizeof(rlRfCalMonFreqLimitConf_t));
    memset ((void *)&rfInitCalib, 0, sizeof(rlRfInitCalConf_t));
    memset ((void *)&timeCfg,     0, sizeof(rlRfCalMonTimeUntConf_t));

    /* Link is not operational: */
    ptrMMWaveMCB->linkStatus[0] = 0U;

    /****************************************************************************************
     * Setup the RF Calibration Time unit:
     ****************************************************************************************/
    if(ptrMMWaveMCB->openCfg.calibMonTimeUnit != 0)
    {        
        timeCfg.numOfCascadeDev = 1;
        timeCfg.calibMonTimeUnit = ptrMMWaveMCB->openCfg.calibMonTimeUnit;
        retVal = rlRfSetCalMonTimeUnitConfig(RL_DEVICE_MAP_INTERNAL_BSS, &timeCfg);
        if (retVal != RL_RET_CODE_OK)
        {
            /* Error: Unable to set the calibration time unit */
            *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALPERIOD, retVal);
            retVal   = MINUS_ONE;
            goto exit;
        }
    }
    else
    {
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVALIDCALMONUNIT, 0);        
        retVal   = MINUS_ONE;
        goto exit;
    }
    

    /****************************************************************************************
     * Setup the RF Calibration Frequency limit:
     ****************************************************************************************/
    freqLimit.freqLimitLow  = ptrMMWaveMCB->openCfg.freqLimitLow;
    freqLimit.freqLimitHigh = ptrMMWaveMCB->openCfg.freqLimitHigh;
    retVal = rlRfSetCalMonFreqLimitConfig(RL_DEVICE_MAP_INTERNAL_BSS, &freqLimit);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Frequency Limit Calibration  */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALCFG, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /* Are we supporting custom calibration or not? */
    if (ptrMMWaveMCB->openCfg.useCustomCalibration == true)
    {
        /*********************************************************************
         * Custom Calibration: Use the application provided calibration mask
         *********************************************************************/
        rfInitCalib.calibEnMask = ptrMMWaveMCB->openCfg.customCalibrationEnableMask;
    }
    else
    {
        /*********************************************************************
         * Default: Enable all calibrations
         *********************************************************************/
        rfInitCalib.calibEnMask = CSL_FMKR (4U, 4U, 1U)     |   /* LODIST calibration            */
                                  CSL_FMKR (5U, 5U, 1U)     |   /* RX ADC DC offset calibration  */
                                  CSL_FMKR (6U, 6U, 1U)     |   /* HPF cutoff calibration        */
                                  CSL_FMKR (7U, 7U, 1U)     |   /* LPF cutoff calibration        */
                                  CSL_FMKR (8U, 8U, 1U)     |   /* Peak detector calibration     */
                                  CSL_FMKR (9U, 9U, 1U)     |   /* TX Power calibration          */
                                  CSL_FMKR (10U, 10U, 1U)   |   /* RX gain calibration           */
                                  CSL_FMKR (11U, 11U, 1U)   |   /* TX phase calibration          */
                                  CSL_FMKR (12U, 12U, 1U);      /* RX IQMM calibration           */
    }
    retVal = rlRfInitCalibConfig(RL_DEVICE_MAP_INTERNAL_BSS, &rfInitCalib);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Calibration init failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALINIT, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /****************************************************************************************
     * Set the channel configuration:
     ****************************************************************************************/
    retVal = rlSetChannelConfig (RL_DEVICE_MAP_INTERNAL_BSS, &ptrMMWaveMCB->openCfg.chCfg);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Set the channel configuration failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECHCFG, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /****************************************************************************************
     * Set the ADC Output configuration:
     ****************************************************************************************/
    retVal = rlSetAdcOutConfig (RL_DEVICE_MAP_INTERNAL_BSS, &ptrMMWaveMCB->openCfg.adcOutCfg);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Set the ADC configuration failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EADCCFG, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /****************************************************************************************
     * Set the low power mode configuration:
     ****************************************************************************************/
    retVal = rlSetLowPowerModeConfig(RL_DEVICE_MAP_INTERNAL_BSS, &ptrMMWaveMCB->openCfg.lowPowerMode);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Set the Low power configuration failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPOWERCFG, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /****************************************************************************************
     * Asynchronous event configuration:
     ****************************************************************************************/
    retVal = MMWave_deviceCfgAsyncEvent (ptrMMWaveMCB, errCode);
    if (retVal < 0)
    {
        /* Error: Asynchronous Event configuration failed; error code is already setup */
        goto exit;
    }

    /****************************************************************************************
     * Do we need to restore calibration data?
     ****************************************************************************************/
    if (ptrCalibrationData != NULL)
    {
        if (ptrCalibrationData->ptrCalibData != NULL)
        {
            /* YES: Restore the calibration data */
            retVal = rlRfCalibDataRestore(RL_DEVICE_MAP_INTERNAL_BSS, ptrCalibrationData->ptrCalibData);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Calibration data restore failed */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALFAIL, retVal);
                retVal   = MINUS_ONE;
                goto exit;
            }

            /* As per the Link documentation; radarSS will consume the data and will send an
             * asynchronous message to indicate the result of the calibration based on the
             * calibration data which has been restored. So we need to wait on the status of
             * the operation. */
            MMWave_internalLinkSync(ptrMMWaveMCB, MMWAVE_RF_INITIALIZED | MMWAVE_RF_INIT_FAILED);

            /* Was the calibration restored succesfully? */
            if (ptrMMWaveMCB->linkStatus[0] & MMWAVE_RF_INIT_FAILED)
            {
                /* NO: Calibration was not restored. The asynchronous message has been
                 * passed to the application too. Setup the error code */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALFAIL, 0);
                retVal = MINUS_ONE;
                goto exit;
            }

            /* Calibration was restored successfully so we need to reset the link operational
             * status again. */
            ptrMMWaveMCB->linkStatus[0] = 0U;
        }

        /* Re-store Phase shift calibration. */
        if (ptrCalibrationData->ptrPhaseShiftCalibData != NULL)
        {
            /* User should have set txIndex equal to PhShiftcalibChunk array index when calling
                rlRfPhShiftCalibDataStore() so that data for all Tx are obtained regardless of
                channels enabled in chCfg. However, phase shift data will be non-zero only
                for Tx channels that are enabled in chCfg */
            if ((ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[0].txIndex != 0) ||
                (ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[1].txIndex != 1) ||
                (ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[2].txIndex != 2) ||
                (ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[3].txIndex != 3))
            {
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPHASESHIFTCALDATARESTOREFAIL, 0);
                retVal   = MINUS_ONE;
                goto exit;
            }
            /* User should not set calibApply to 1 for any Tx. rlRfPhShiftCalibDataRestore() will take care of
                it internally */
            if ((ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[0].calibApply != 0) ||
                (ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[1].calibApply != 0) ||
                (ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[2].calibApply != 0) ||
                (ptrCalibrationData->ptrPhaseShiftCalibData->PhShiftcalibChunk[3].calibApply != 0))
            {
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPHASESHIFTCALDATARESTOREFAIL, 0);
                retVal   = MINUS_ONE;
                goto exit;
            }
            /* Basic validation passed: Restore the phase shift calibration data */
            retVal = rlRfPhShiftCalibDataRestore(ptrMMWaveMCB->deviceMap, ptrCalibrationData->ptrPhaseShiftCalibData);
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Phase shift Calibration data restore failed */
                *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EPHASESHIFTCALDATARESTOREFAIL, retVal);
                retVal   = MINUS_ONE;
                goto exit;
            }
        }
    }

    /****************************************************************************************
     * Initialize the RF:
     ****************************************************************************************/
    retVal = rlRfInit(RL_DEVICE_MAP_INTERNAL_BSS);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: RF Initialization failed */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ERFINIT, retVal);
        retVal   = MINUS_ONE;
        goto exit;
    }

    /***************************************************************************
     * SYNCHRONIZATION: We need to loop around till the BSS has performed the
     * RF Initialization; either a success or failure.
     ***************************************************************************/
    MMWave_internalLinkSync(ptrMMWaveMCB, MMWAVE_RF_INITIALIZED | MMWAVE_RF_INIT_FAILED);

    /* Was the RF Initialized succesfully? */
    if (ptrMMWaveMCB->linkStatus[0] & MMWAVE_RF_INIT_FAILED)
    {
        /* NO: The RF initialization failed. The asynchronous event was passed to the application
         * The application can decode the exact reason for the failure. We simply set the error
         * code and report failure */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ECALFAIL, 0);
        retVal = MINUS_ONE;
        goto exit;
    }

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the mmWave link.
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
int32_t MMWave_initLink (MMWave_MCB* ptrMMWaveMCB, int32_t* errCode)
{
    int32_t                 retVal = 0;
    MMWave_SpawnFxnNode*    ptrSpawnFxnNode;
    uint32_t                index;

    /* Initialize and setup the spawn lists */
    for (index = 0U; index < MMWAVE_MAX_NUM_SPAWN_LIST; index++)
    {
        /* Get the pointer to the spawn node */
        ptrSpawnFxnNode = &ptrMMWaveMCB->spawnTable[index];

        /* Initialize the spawn node */
        memset ((void*)ptrSpawnFxnNode, 0, sizeof(MMWave_SpawnFxnNode));

        /* Add the node to the free list: */
        MMWave_listAdd ((MMWave_ListNode**)&ptrMMWaveMCB->ptrSpawnFxnFreeList, (MMWave_ListNode*)ptrSpawnFxnNode);
    }

    /* Reset the active list: */
    ptrMMWaveMCB->ptrSpawnFxnActiveList = NULL;

    /* Setup and Initialize the CRC Driver: */
    if (MMWave_initCRC (ptrMMWaveMCB, errCode) == 0)
    {
        /* YES: Setup the mmWave Link */
        if (MMWave_initMMWaveLink (ptrMMWaveMCB, errCode) == 0)
        {
            /* Successfully initialized the mmWave Link: */
            retVal = 0;
        }
        else
        {
            /* Error: Unable to setup the mmWave Link */
            retVal = MINUS_ONE;
        }
    }
    else
    {
        /* Error: Unable to setup the CRC Driver */
        retVal = MINUS_ONE;
    }
    return retVal;
}

