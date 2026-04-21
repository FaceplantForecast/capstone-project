/*
 *   @file  objectdetection.c
 *
 *   @brief
 *      Object Detection DPC implementation.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2017 - 2025 Texas Instruments, Inc.
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

/* MCU+SDK include files */
#include <kernel/dpl/HeapP.h>
#include <kernel/dpl/ClockP.h>
#include <kernel/dpl/CycleCounterP.h>
#include <kernel/dpl/CacheP.h>
#ifdef SOC_AWR294X
#include <drivers/hw_include/cslr_soc.h>
#endif
/* mmWave SDK Include Files: */
#include <ti/common/syscommon.h>
#include <ti/utils/mathutils/mathutils.h>
/* HWA_SOC Include files */
#include <drivers/soc.h>
#include <ti/control/dpm/dpm.h>



/* DSP Mathlib include files */
#include <ti/mathlib/src/cossp/c66/cossp.h>
#include <ti/dsplib/src/DSPF_sp_dotp_cplx/DSPF_sp_dotp_cplx.h>

/* C66x mathlib */
/* Suppress the mathlib.h warnings
 *  #48-D: incompatible redefinition of macro "TRUE"
 *  #48-D: incompatible redefinition of macro "FALSE"
 */
#pragma diag_push
#pragma diag_suppress 48
#include <ti/mathlib/mathlib.h>
#pragma diag_pop


 /** @addtogroup DPC_OBJDET_IOCTL__INTERNAL_DEFINITIONS
  @{ */

/*! This is supplied at command line when application builds this file. This file
 * is owned by the application and contains all resource partitioning, an
 * application may include more than one DPC and also use resources outside of DPCs.
 * The resource definitions used by this object detection DPC are prefixed by DPC_OBJDET_ */
#include APP_RESOURCE_FILE

/* Obj Det instance etc */
#include <ti/datapath/dpc/objectdetection/objdethwaDDMA/include/objectdetectioninternal.h>
#include <ti/datapath/dpc/objectdetection/objdethwaDDMA/objectdetection.h>

/* Power Optimization configurations */
#if defined(SOC_AWR2X44P)
#define DPC_OBJDET_HWA_CG_ENABLE                  (0x2U)
#define DPC_OBJDET_HWA_CLOCK_GATE                 (0x7U)
#define DPC_OBJDET_HWA_CLOCK_UNGATE               (0x0U)
#define DPC_OBJDET_DSP_PG_ENABLE                  (0x1U)
#define DPC_OBJDET_DSP_CLK_SRC_DSP_PLL_MUX        (0x222U)
#define DPC_OBJDET_DSP_UC_ENABLE                  (0x2U)
#define DPC_OBJDET_DSP_POWERED_UP                 (0x30U)
#define DPC_OBJDET_DSP_POWERED_DOWN               (0x0U)
#define DPC_OBJDET_DSP_PD_STATUS_MASK             (0x30U)
#endif

/******************************************************************************/
/* Local definitions */

#define DOUBLEWORD_ALIGNED    (8U)

#define QVALUE_NOISE          (11U)
#define QVALUE_SIGNAL         (11U)


/*! Radar cube data buffer alignment in bytes. */
#if defined(SUBSYS_MSS) || defined (SUBSYS_M4)
#define DPC_OBJDET_RADAR_CUBE_DATABUF_BYTE_ALIGNMENT      DPU_RANGEPROCHWA_RADARCUBE_BYTE_ALIGNMENT_R5F
#else
#define DPC_OBJDET_RADAR_CUBE_DATABUF_BYTE_ALIGNMENT      DPU_RANGEPROCHWA_RADARCUBE_BYTE_ALIGNMENT_DSP
#endif


/*! Detection matrix alignment is declared by CFAR dpu, we size to
 *  the max of this and CPU alignment for accessing detection matrix
 *  it is exported out of DPC in processing result so assume CPU may access
 *  it for post-DPC processing. Note currently the CFAR alignment is the same as
 *  CPU alignment so this max is redundant but it is more to illustrate the
 *  generality of alignments should be done.
 */
#define DPC_OBJDET_DET_MATRIX_DATABUF_BYTE_ALIGNMENT       (CSL_MAX(sizeof(uint16_t), \
                                                                DPU_DOPPLER_DET_MATRIX_BYTE_ALIGNMENT))

#define DPC_OBJDET_HWA_MAX_WINDOW_RAM_SIZE_IN_SAMPLES    (CSL_DSS_HWA_WINDOW_RAM_U_SIZE >> 3)

#define DPC_USE_SYMMETRIC_WINDOW_RANGE_DPU
#define DPC_USE_SYMMETRIC_WINDOW_DOPPLER_DPU
#define DPC_DPU_RANGEPROC_FFT_WINDOW_TYPE                  MATHUTILS_WIN_HANNING
#define DPC_DPU_RANGEPROC_INTERFMITIG_WINDOW_TYPE          MATHUTILS_WIN_HANNING
#define DPC_DPU_DOPPLERPROC_FFT_WINDOW_TYPE                MATHUTILS_WIN_HANNING

/*! Number of interference mitigation window samples. Used 16 as the size
    instead of 14 because mathUtils generates the first and the last samples
    as 0, which are not useful. */
#define DPC_OBJDET_RANGEPROC_NUM_INTFMITIG_WIN_SIZE_TOTAL       (16U)

/*! Interference mitigation window type */
#define DPC_OBJDET_RANGEPROC_INTERFMITIG_WINDOW_TYPE            MATHUTILS_WIN_HANNING

/*! Q Format of interference mitigation window */
#define DPC_OBJDET_QFORMAT_RANGEPROC_INTERFMITIG_WINDOW         (5U)

#define DPC_OBJDET_QFORMAT_RANGE_FFT 17
#define DPC_OBJDET_QFORMAT_DOPPLER_FFT 17

/* Number of Azim FFT Bins */
#define OBJECTDETECTION_NUM_AZIM_FFT_BINS (32U)

#ifdef SOC_AWR294X
#define OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ    360000
#endif

// #define OBJECTDETHWA_PRINT_DPC_TIMING_INFO
#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
#define OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE 10U
/* As and when frames come in, these variables will store the actual timestamps.
   As soon as the last frame processing is completed, they will be replaced by
   time values in milliseconds, with respect to the start of the first frame of
   the OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE set. */
typedef struct timingInfo
{
    uint32_t frameStartTimes[OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    uint32_t rangeEndTimes[OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    uint32_t dopEndTimes[OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    uint32_t aoaStartTimes[OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    uint32_t aoaEndTimes[OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    uint32_t resEndTimes[OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    uint32_t rangeEndCnt, dopEndCnt, aoaStartCnt, aoaEndCnt, frameCnt, resEndCnt;
}timingInfo;

timingInfo gTimingInfo;
#endif

ObjDetObj gObjDetObj __attribute__((aligned(HeapP_BYTE_ALIGNMENT))) 
#if SUBSYS_M4
__attribute__((section(".dpcGlobals")))
#endif
;

/* Buffer to store DC and Interf Stats of unique band chirps of all RX.
 * Increase this size for more number of bands or RX. 
 * This buffer is used in below rangeProcChain modes:
 * 1. DPU_RANGEPROCHWA_PREVIOUS_FRAME_DC_MODE
 *    Size is 1(DC Estimation) * RL_MAX_SUBFRAMES (numSubframes) * 4 (numRx) * sizeof(uint32_t) (numBytesPerSample)
 * 2. DPU_RANGEPROCHWA_PREVIOUS_NTH_CHIRP_ESTIMATES_MODE
 *    Size is (1(DC Estimation) + 2(Interference Statistics)) * RL_MAX_SUBFRAMES (numSubframes) * 6 (numBandsTotal) * 4 (numRx) * sizeof(uint32_t) (numBytesPerSample) 
 * But, for DPU_RANGEPROCHWA_DEFAULT_MODE mode, this buffer is not required. 
 * Also, the section .preProcBuf used to store this buffer should be kept in M4_RAM,
 * otherwise performance overhead might be observed. */
uint8_t preProcBuffer[3U * RL_MAX_SUBFRAMES * 6U * 4U * (uint32_t)sizeof(uint32_t)] __attribute__((section(".preProcBuf")));

/* Buffer to generate window for range and doppler fft. It is required for
  subframe re-configuration as window buffer should not overlap with
finalDetObjList. */
uint8_t gWinBuf[4096] __attribute__((section(".dpc_l2Heap")));

#define DOPPLER_MAXDOP_SUBBAND_BUFFER_SIZE 256U /* Allocated for 768 chirps, 6 subbands, 2 ping-pong */
uint8_t dopMaxSubBandScratchBuf[DOPPLER_MAXDOP_SUBBAND_BUFFER_SIZE];

/**************************************************************************
 ************************** Local Functions Declarations ******************
 **************************************************************************/
static int32_t DPC_ObjDet_rangeConfig(DPU_RangeProcHWA_Handle dpuHandle,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
#if 0
                   DPC_ObjectDetection_DynCfg    *dynCfg,
#endif
                   EDMA_Handle                   edmaHandle,
                   DPIF_RadarCube                *radarCube,
                   MemPoolObj                    *CoreLocalRamObj,
                   MemPoolObj                    *WinBufRamObj,
                   uint32_t                      *windowOffset,
                   uint32_t                      *CoreLocalRamScratchUsage,
                   DPU_RangeProcHWA_Config       *cfgSave,
                   ObjDetObj                     *ptrObjDetObj) 
#ifdef SUBSYS_M4                   
                   __attribute__((section(".customCode")))
#endif
;

static int32_t DPC_ObjDet_dopplerConfig(SubFrameObj *obj,
                   DPU_DopplerProcHWA_Handle dpuHandle,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
                   uint8_t log2NumDopplerBins,
                   float *                       antennaCalibParamsPtr,
                   EDMA_Handle                   edmaHandle,
                   uint32_t                      radarCubeDecompressedSizeInBytes,
                   DPIF_RadarCube                *radarCube,
                   DPIF_DetMatrix                *detMatrix,
                   MemPoolObj                    *CoreLocalRamObj,
                   MemPoolObj                    *L3ramObj,
                   MemPoolObj                    *WinBufRamObj,
                   void *                        CoreLocalScratchStartPoolAddr,
                   volatile void *               CoreLocalScratchStartPoolAddrNextDPU,
                   volatile void *               l3RamStartPoolAddrNextDPU,
                   uint32_t                      *windowOffset,
                   uint32_t                      *CoreLocalRamScratchUsage,
                   DPU_DopplerProcHWA_Config     *cfgSave,
                   ObjDetObj                     *objDetObj) 
#ifdef SUBSYS_M4                     
                   __attribute__((section(".customCode")))
#endif
;

static int32_t DPC_ObjDet_rangeCfarConfig(DPU_RangeCFARProcHWA_Handle dpuHandle,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
                   EDMA_Handle                   edmaHandle,
                   DPIF_DetMatrix                *detMatrix,
                   MemPoolObj                    *CoreLocalRamObj,
                   MemPoolObj                    *L3ramObj,
                   void                          *CoreLocalScratchStartPoolAddrNextDPU,
                   void                          *l3RamStartPoolAddrNextDPU,
                   DPU_RangeCfarProcHWA_Config   *cfgSave,
                   ObjDetObj                     *ptrObjDetObj) 
#ifdef SUBSYS_M4                      
                   __attribute__((section(".customCode")))
#endif
;

static int32_t DPC_ObjDet_reconfigSubFrame(ObjDetObj *objDetObj, uint8_t subFrameIndx);

static void DPC_ObjDet_EDMAChannelConfigAssist(EDMA_Handle handle, uint32_t chNum, uint32_t shadowParam, uint32_t eventQueue, DPEDMA_ChanCfg *chanCfg);

static void DPC_ObjectDetection_ConfigureADCBuf(uint16_t rxChannelEn, uint32_t chanDataSize);

static void checkFFTClipStatus(ObjDetObj *objDetObj, uint32_t* clipCount);

#ifdef INCLUDE_DPM
static DPM_DPCHandle DPC_ObjectDetection_init(
    DPM_Handle dpmHandle,
    DPM_InitCfg *ptrInitCfg,
    int32_t *errCode);

static int32_t DPC_ObjectDetection_execute(
    DPM_DPCHandle handle,
    DPM_Buffer *ptrResult);

static int32_t DPC_ObjectDetection_ioctl(
    DPM_DPCHandle handle,
    uint32_t cmd,
    void *arg,
    uint32_t argLen);

static int32_t DPC_ObjectDetection_start(DPM_DPCHandle handle);
static int32_t DPC_ObjectDetection_stop(DPM_DPCHandle handle);
static int32_t DPC_ObjectDetection_deinit(DPM_DPCHandle handle);
static void DPC_ObjectDetection_frameStart(DPM_DPCHandle handle);
/**
@}
*/

/**************************************************************************
 ************************* Global Declarations ****************************
 **************************************************************************/

/** @addtogroup DPC_OBJDET__GLOBAL
 @{ */

/**
 * @brief   Global used to register Object Detection DPC in DPM
 */
DPM_ProcChainCfg gDPC_ObjectDetectionCfg =
    {
        DPC_ObjectDetection_init,      /* Initialization Function:         */
        DPC_ObjectDetection_start,     /* Start Function:                  */
        DPC_ObjectDetection_execute,   /* Execute Function:                */
        DPC_ObjectDetection_ioctl,     /* Configuration Function:          */
        DPC_ObjectDetection_stop,      /* Stop Function:                   */
        DPC_ObjectDetection_deinit,    /* Deinitialization Function:       */
        NULL,                          /* Inject Data Function:            */
        NULL,                          /* Chirp Available Function:        */
        DPC_ObjectDetection_frameStart /* Frame Start Function:            */
};

/**
@}
*/
#endif
/**************************************************************************
 ************************** Local Functions *******************************
 **************************************************************************/
/**
 *  @b Description
 *  @n
 *      Utility function for reseting memory pool.
 *
 *  @param[in]  pool Handle to pool object.
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      none.
 */
static void DPC_ObjDet_MemPoolReset(MemPoolObj *pool)
{
    pool->currAddr = (uintptr_t)pool->cfg.addr;
    pool->maxCurrAddr = pool->currAddr;
}

/**
 *  @b Description
 *  @n
 *      Utility function for setting memory pool to desired address in the pool.
 *      Helps to rewind for example.
 *
 *  @param[in]  pool Handle to pool object.
 *  @param[in]  addr Address to assign to the pool's current address.
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      None
 */
static void DPC_ObjDet_MemPoolSet(MemPoolObj *pool, void *addr)
{
    pool->currAddr = (uintptr_t)addr;
    pool->maxCurrAddr = CSL_MAX(pool->currAddr, pool->maxCurrAddr);
}

/**
 *  @b Description
 *  @n
 *      Utility function for getting memory pool current address.
 *
 *  @param[in]  pool Handle to pool object.
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      pointer to current address of the pool (from which next allocation will
 *      allocate to the desired alignment).
 */
static void *DPC_ObjDet_MemPoolGet(MemPoolObj *pool)
{
    return((void *)pool->currAddr);
}

#if 0 /* may be useful in future */
/**
 *  @b Description
 *  @n
 *      Utility function for getting current memory pool usage.
 *
 *  @param[in]  pool Handle to pool object.
 *
 *  @retval
 *      Amount of pool used in bytes.
 */
static uint32_t DPC_ObjDet_MemPoolGetCurrentUsage(MemPoolObj *pool)
{
    return((uint32_t)(pool->currAddr - (uintptr_t)pool->cfg.addr));
}
#endif

/**
 *  @b Description
 *  @n
 *      Utility function for getting maximum memory pool usage.
 *
 *  @param[in]  pool Handle to pool object.
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Amount of pool used in bytes.
 */
static uint32_t DPC_ObjDet_MemPoolGetMaxUsage(MemPoolObj *pool)
{
    return((uint32_t)(pool->maxCurrAddr - (uintptr_t)pool->cfg.addr));
}

/**
 *  @b Description
 *  @n
 *      Utility function for allocating from a static memory pool.
 *
 *  @param[in]  pool Handle to pool object.
 *  @param[in]  size Size in bytes to be allocated.
 *  @param[in]  align Alignment in bytes
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      pointer to beginning of allocated block. NULL indicates could not
 *      allocate.
 */
static void *DPC_ObjDet_MemPoolAlloc(MemPoolObj *pool,
                              uint32_t size,
                              uint8_t align)
{
    void *retAddr = NULL;
    uintptr_t addr;

    addr = CSL_MEM_ALIGN(pool->currAddr, align);
    if ((addr + size) <= ((uintptr_t)pool->cfg.addr + pool->cfg.size))
    {
        retAddr = (void *)addr;
        pool->currAddr = addr + size;
        pool->maxCurrAddr = CSL_MAX(pool->currAddr, pool->maxCurrAddr);
    }

    return (retAddr);
}

#ifdef INCLUDE_DPM

/**
 *  @b Description
 *  @n
 *      Sends Assert
 *
 *  @retval
 *      Not Applicable.
 */
void _DPC_Objdet_Assert(DPM_Handle handle, int32_t expression,
                        const char *file, int32_t line)
{
    DPM_DPCAssert fault;

    if (!expression)
    {
        fault.lineNum = (uint32_t)line;
        fault.arg0 = 0U;
        fault.arg1 = 0U;
        strncpy(fault.fileName, file, (DPM_MAX_FILE_NAME_LEN - 1));

        /* Report the fault to the DPM entities */
        DPM_ioctl(handle,
                  DPM_CMD_DPC_ASSERT,
                  (void *)&fault,
                  sizeof(DPM_DPCAssert));
    }
}
#endif

/**
 *  @b Description
 *  @n
 *      DPC frame start function registered with DPM. This is invoked on reception
 *      of the frame start ISR from the RF front-end. This API is also invoked
 *      when application issues @ref DPC_OBJDET_IOCTL__TRIGGER_FRAME to simulate
 *      a frame trigger (e.g for unit testing purpose).
 *
 *  @param[in]  handle DPM's DPC handle
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void DPC_ObjectDetection_frameStart(DPM_DPCHandle handle)
{
    ObjDetObj *objDetObj = (ObjDetObj *)handle;

    objDetObj->stats.frameStartTimeStamp = CycleCounterP_getCount32();
#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    gTimingInfo.frameStartTimes[gTimingInfo.frameCnt % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE] = CycleCounterP_getCount32();
    gTimingInfo.frameCnt++;
#endif

#ifdef SOC_AWR2X44P
    CSL_dss_rcmRegs *ptrDssRcmRegs = (CSL_dss_rcmRegs *)CSL_CM4_DSS_RCM_U_BASE;

    if(objDetObj->subFrameObj[0].staticCfg.powerOptCfg.hwaStateAfterFrameProc == DPC_OBJDET_HWA_CG_ENABLE)
    {
        
        ptrDssRcmRegs->DSS_HWA_CLK_GATE = DPC_OBJDET_HWA_CLOCK_UNGATE;
    }
#endif

    DebugP_logInfo("ObjDet DPC: Frame Start, frameIndx = %d, subFrameIndx = %d\n",
                   objDetObj->stats.frameStartIntCounter, objDetObj->subFrameIndx);

#ifdef INCLUDE_DPM
    /* Perform DPM_notifyExecute only if the previous frame's/subFrame's result has been exported,
       i.e., the result exported ioctl has been received from the MSS. Otherwise,
       DPM_notifyExecute will be performed in the result exported ioctl */
    if (objDetObj->numTimesResultExported == objDetObj->stats.subframeStartIntCounter)
    {
        DebugP_assert(DPM_notifyExecute(objDetObj->dpmHandle, handle) == 0);
    }
    else
    {
        /* Do Nothing */
    }
#else
    /* Start the DPC Execution for this frame. */
    SemaphoreP_post (&objDetObj->dpcExecSemHandle);
#endif

    /* Increment interrupt counter for debugging, sync, and reporting purpose */
    if (objDetObj->subFrameIndx == 0U)
    {
        objDetObj->stats.frameStartIntCounter++;
    }

    objDetObj->stats.subframeStartIntCounter++;
}

/**
 *  @b Description
 *  @n
 *      Computes the length of window to generate for range DPU.
 *
 *  @param[in]  cfg Range DPU configuration
 *
 *  @retval   Length of window to generate
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static uint32_t DPC_ObjDet_GetRangeWinGenLen(DPU_RangeProcHWA_Config *cfg)
{
    uint16_t numAdcSamples;
    uint32_t winGenLen;

    numAdcSamples = cfg->staticCfg.ADCBufData.dataProperty.numAdcSamples;

#ifdef DPC_USE_SYMMETRIC_WINDOW_RANGE_DPU
    winGenLen = ((uint32_t)numAdcSamples + 1U) / 2U;
#else
    winGenLen = numAdcSamples;
#endif
    return (winGenLen);
}

/**
 *  @b Description
 *  @n
 *      Generate the range DPU window using mathutils API.
 *
 *  @param[in]  cfg Range DPU configuration, output window is generated in window
 *                  pointer in the staticCfg of this.
 *
 *  @retval   None
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static void DPC_ObjDet_GenRangeWindow(DPU_RangeProcHWA_Config *cfg)
{

    /* Symmetric window */
    uint32_t interfMitigWindow[DPC_OBJDET_RANGEPROC_NUM_INTFMITIG_WIN_SIZE_TOTAL >> 1];
    uint8_t idx;

    mathUtils_genWindow((uint32_t *)interfMitigWindow,
                        DPC_OBJDET_RANGEPROC_NUM_INTFMITIG_WIN_SIZE_TOTAL,
                        DPC_OBJDET_RANGEPROC_NUM_INTFMITIG_WIN_SIZE_TOTAL >> 1,
                        DPC_DPU_RANGEPROC_INTERFMITIG_WINDOW_TYPE,
                        DPC_OBJDET_QFORMAT_RANGEPROC_INTERFMITIG_WINDOW);

    /* Only 5 win samples are supported by the HWA */
    for (idx = 0; idx < DPU_RANGEPROCHWADDMA_NUM_INTFMITIG_WIN_HWACOMMONCFG_SIZE; idx++)
    {
        cfg->hwRes.hwaCfg.hwaInterfMitigWindow[DPU_RANGEPROCHWADDMA_NUM_INTFMITIG_WIN_HWACOMMONCFG_SIZE - 1U - idx] =
            (uint8_t)interfMitigWindow[(DPC_OBJDET_RANGEPROC_NUM_INTFMITIG_WIN_SIZE_TOTAL >> 1U) - 2U - idx];
    }

    /* Range FFT window */
    mathUtils_genWindow((uint32_t *)cfg->staticCfg.window,
                        cfg->staticCfg.ADCBufData.dataProperty.numAdcSamples,
                        DPC_ObjDet_GetRangeWinGenLen(cfg),
                        DPC_DPU_RANGEPROC_FFT_WINDOW_TYPE,
                        DPC_OBJDET_QFORMAT_RANGE_FFT);
}

/**
 *  @b Description
 *  @n
 *      Computes the length of window to generate for doppler DPU.
 *
 *  @param[in]  cfg Doppler DPU configuration
 *
 *  @retval   Length of window to generate
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static uint32_t DPC_ObjDet_GetDopplerWinGenLen(DPU_DopplerProcHWA_Config *cfg)
{
    uint16_t numDopplerChirps;
    uint32_t winGenLen;

    numDopplerChirps = cfg->staticCfg.numChirps;

#ifdef DPC_USE_SYMMETRIC_WINDOW_DOPPLER_DPU
    winGenLen = ((uint32_t)numDopplerChirps + 1U) / 2U;
#else
    winGenLen = numDopplerChirps;
#endif
    return (winGenLen);
}

/**
 *  @b Description
 *  @n
 *      Generate the doppler DPU window using mathutils API.
 *
 *  @param[in]  cfg Doppler DPU configuration, output window is generated in window
 *                  pointer embedded in this configuration.
 *
 *  @retval   winType window type, see mathutils.h
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static uint32_t DPC_ObjDet_GenDopplerWindow(DPU_DopplerProcHWA_Config *cfg)
{
    uint32_t winType;

    /* For too small window, force rectangular window to avoid loss of information
     * due to small window values (e.g. hanning has first and last coefficients 0) */
    if (cfg->staticCfg.numChirps <= 4U)
    {
        winType = MATHUTILS_WIN_RECT;
    }
    else
    {
        winType = DPC_DPU_DOPPLERPROC_FFT_WINDOW_TYPE;
    }

    mathUtils_genWindow((uint32_t *)cfg->hwRes.hwaCfg.window,
                        cfg->staticCfg.numChirps,
                        DPC_ObjDet_GetDopplerWinGenLen(cfg),
                        winType,
                        DPC_OBJDET_QFORMAT_DOPPLER_FFT);

    return (winType);
}

/**
 *  @b Description
 *  @n
 *      Allocates Shawdow paramset
 */
static void allocateEDMAShadowChannel(EDMA_Handle edmaHandle, uint32_t *param)
{
    int32_t             testStatus = SystemP_SUCCESS;
    EDMA_Config        *config;
    EDMA_Object        *object;

    config = (EDMA_Config *) edmaHandle;
    object = config->object;

    if(*param < SOC_EDMA_NUM_PARAMSETS)
    {
        if((object->allocResource.paramSet[*param/32U] & ((uint32_t)1U << *param%32U)) != ((uint32_t)1U << *param%32U))
        {
            testStatus = EDMA_allocParam(edmaHandle, param);
            DebugP_assert(testStatus == SystemP_SUCCESS);
        }
    }
    else
    {
        DebugP_assert(false);
    }

    return;
}

/**
 *  @b Description
 *  @n
 *     Function calls EDMA param, channel, tcc allocation.
 *     DDMA Datapath assumes paramsetNumber = channelNumber = TCC
 *
 *  @param[in]  handle   EDMA handle
 *  @param[in]  chNum    DMA channel number
 *  @param[in]  shadowParam    DMA shadow paramId
 *  @param[in]  eventQueue    Event queue num
 *  @param[out]  chanCfg    Stores channel configuration
 *  @retval   None
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static void DPC_ObjDet_EDMAChannelConfigAssist(EDMA_Handle handle, uint32_t chNum, uint32_t shadowParam, uint32_t eventQueue, DPEDMA_ChanCfg *chanCfg)
{

    DebugP_assert(chanCfg != NULL);

    DPEDMA_allocateEDMAChannel(handle, &chNum, &chNum, &chNum);

    chanCfg->channel = chNum;
    chanCfg->tcc = chNum;
    chanCfg->paramId = chNum;

    chanCfg->shadowPramId = shadowParam;

    allocateEDMAShadowChannel(handle, &shadowParam);

    chanCfg->eventQueue = eventQueue;

    return;

}


/**
 *  @b Description
 *  @n
 *     EDMA configuration that sens intersected objects between
 *    Doppler DPU and Range CFAR stage to L2 scratch buffer from L3
 *    memory.
 *
 *  @param[in]  edmaHandle   EDMA handle
 *  @param[in]  hwRes    Dop DPU hw resources
 *  @param[in]  edmaDetObjs    Channel Configuration
 *  @retval   Error Code
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjectDetection_configEdmaDetObjsOut
(
    EDMA_Handle                   edmaHandle,
    DPU_DopplerProcHWA_HW_Resources *hwRes,
    DPEDMA_ChanCfg *edmaDetObjs
)
{
    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;

    syncABCfg.aCount = (uint16_t)sizeof(DetObjParams);
    syncABCfg.bCount = 1U;
    syncABCfg.cCount = 1U;
    syncABCfg.srcAddress = (uint32_t)hwRes->detObjList;
    syncABCfg.destAddress = (uint32_t)hwRes->finalDetObjList;
    syncABCfg.srcBIdx = (int32_t)sizeof(DetObjParams); // doesn't matter
    syncABCfg.dstBIdx = (int32_t)sizeof(DetObjParams); // doesn't matter
    syncABCfg.srcCIdx = (int16_t)sizeof(DetObjParams); // doesn't matter
    syncABCfg.dstCIdx = (int16_t)sizeof(DetObjParams); // doesn't matter

    chainingCfg.chainingChannel  = (uint8_t)edmaDetObjs->channel;
    chainingCfg.isIntermediateChainingEnabled = false;
    chainingCfg.isFinalChainingEnabled        = false;

     retVal =  DPEDMA_configSyncAB(edmaHandle,
                        edmaDetObjs,
                        &chainingCfg,
                        &syncABCfg,
                        false,//isEventTriggered
                        false, //isIntermediateTransferCompletionEnabled
                        true,//isTransferCompletionEnabled
                        NULL, //transferCompletionCallbackFxn
                        NULL,
                        NULL);//transferCompletionCallbackFxnArg

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }
exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *     Function checks if the same object is present
 *     in both the rangeCFAR detected object list and the
 *     dopplerProc detected object list (also called doppler
 *     list)
 *
 *  @param[in]  rangeIdx  range index of the object (from the doppler list)
 *  @param[in]  dopIdx    doppler index of the object (from the doppler list)
 *  @param[in]  rangeCfarList list of rangeCFAR detected objects.
 *  @param[in]  numObjToSearch number of Objects to search.
 *  @retval   boolean indicating presence (true) or absence (false).
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static inline uint32_t isObjInRangeAndDopplerList(const uint32_t rangeIdx,
                                          const uint32_t dopIdx,
                                            RangeCfarListObj *rangeCfarList,
                                            uint32_t numObjToSearch)
{
    uint32_t idx;
    for (idx = 0; idx < numObjToSearch; idx++)
    {
        if (rangeCfarList[idx].rangeIdx == rangeIdx)
        {
            if (rangeCfarList[idx].dopIdx == dopIdx)
            {
                return 1;
            }
        }
    }

    return 0;
}

#ifdef SUBSYS_DSS


/**
 *  @b Description
 *  @n
 *     Function performs quadratic interpolation around a peak
 *
 *  @param[in]  y A Three sample array ([y0,y1,y2]) where
 *              (y1 > y2) and (y1 > y0)
 *
 *  @retval   location of the interpolated peak, relative to y1.
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static inline float DPC_ObjDet_quadInterpAroundPeak(const uint32_t * restrict y)
{

    float ym1, y0, yp1;
    float thetapk; //, yOut;

    ym1 = (float) y[0]; /* y(peak-1) */
    y0  = (float) y[1]; /* y(peak) */
    yp1 = (float) y[2]; /* y(peak+1) */

    thetapk = divsp((yp1 - ym1), (2 * (2 * y0 - yp1 - ym1)));
    /* yOut = y0 + (((yp1 - ym1) / 4) * thetapk); */

    return thetapk;

}

/**
 *  This routine calculates the dot product of 2 single-precision complex
 *  float vectors. The even numbered locations hold the real parts of the
 *  complex numbers while the odd numbered locations contain the imaginary
 *  portions. It is an exact copy of the DSPF_sp_dotp_cmplx function from
 *  the DSPLIB.
 *
 *         @param x   Pointer to array holding the first floating-point vector
 *         @param y   Pointer to array holding the second floating-point vector
 *         @param nx  Number of values in the x and y vectors
 *         @param re  Pointer to the location storing the real part of the result
 *         @param im  Pointer to the location storing the imaginary part of the result
 *
 * @par Assumptions:
 *   Loop counter must be multiple of 4 and > 0. <BR>
 *   The x and y arrays must be double-word aligned. <BR>
 *
 *
 */
static inline void dotpCmplxf(const float * restrict x, const float * restrict y, int nx,
                       float * restrict re, float * restrict im)
{
    int i;
    __float2_t x0_im_re, y0_im_re, result0 = 0;
    __float2_t x1_im_re, y1_im_re, result1 = 0;
    __float2_t x2_im_re, y2_im_re, result2 = 0;
    __float2_t x3_im_re, y3_im_re, result3 = 0;
    __float2_t result;

    _nassert(nx % 4 == 0);
    _nassert(nx > 0);
    _nassert((int)x % 8 == 0);
    _nassert((int)y % 8 == 0);

    for(i = 0; i < 2 * nx; i += 8)
    {
        /* load 4 sets of input data */
        x0_im_re = _amem8_f2((void*)&x[i]);
        y0_im_re = _amem8_f2((void*)&y[i]);

        x1_im_re = _amem8_f2((void*)&x[i+2]);
        y1_im_re = _amem8_f2((void*)&y[i+2]);

        x2_im_re = _amem8_f2((void*)&x[i+4]);
        y2_im_re = _amem8_f2((void*)&y[i+4]);

        x3_im_re = _amem8_f2((void*)&x[i+6]);
        y3_im_re = _amem8_f2((void*)&y[i+6]);

        /* calculate 4 running sums */
        result0 = _daddsp(_complex_mpysp(x0_im_re, y0_im_re), result0);
        result1 = _daddsp(_complex_mpysp(x1_im_re, y1_im_re), result1);
        result2 = _daddsp(_complex_mpysp(x2_im_re, y2_im_re), result2);
        result3 = _daddsp(_complex_mpysp(x3_im_re, y3_im_re), result3);
    }

    result = _daddsp(_daddsp(result0,result1),_daddsp(result2,result3));
    *re =  _hif2(result);
    *im =  _lof2(result);
}

/*! @brief  Complex data type, natural for C66x complex
 * multiplication instructions. */
typedef struct cmplxfImRe_t_
{
    float imag; /*!< @brief imaginary part */
    float real; /*!< @brief real part */
} cmplxfImRe_t;
/*! @brief  Complex union type, natural for C66x intrinsic
 * instructions. */
typedef union cmplxfUnion_t_
{
	cmplxfImRe_t cmplx;
	float dat[2];
	double ddat;
}cmplxfUnion_t;

/*! @brief  Unsigned round (for floats). */
#define ROUND_UNSIGNED(x) ((x) + 0.5f)
#define AOA_DFT_LEN (128)
/* A simple sin-cos LUT used for the DFT computations in
 * DPC_ObjDet_estimateXYZ */
cmplxfImRe_t dftSinCosTable[AOA_DFT_LEN] __attribute__((aligned(8))) = {
#include "cossintable.c"
};

/**
 *  @b Description
 *  @n
 *     Function estimates XYZ coordinates of objects in the object list
 *
 *  @param[in]  subFrmObj   subframe object
 *  @param[in]  objDetObj   DPC object detection object
 *  @param[in]  detObjList  Detected object list
 *  @param[out] objOut      List with x, y, z coordinates populated for each object
 *  @param[in]  numObjOut   Number of detected objects
 *  @param[out] finalNumObjOut  Number of validated objects
 *
 *  @retval   None
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
int32_t DPC_ObjDet_estimateXYZ(SubFrameObj * restrict subFrmObj,
                               ObjDetObj * restrict objDetObj,
                               const DetObjParams * restrict detObjList,
                               DPIF_PointCloudCartesian * restrict objOut,
                               uint32_t numObjOut,
                               uint32_t * restrict finalNumObjOut)
{

    uint16_t azimFFTSize = subFrmObj->dpuCfg.dopplerCfg.staticCfg.numAzimFFTBins;
    const float invAzimFFTSize = divsp(1.0f,(float) azimFFTSize);
    uint32_t objIdx, sampIdx, idx;
    uint32_t maxAzimMaskWidth = 8*sizeof(objDetObj->commonCfg.zeroInsrtMaskCfg.zeroInsrtMaskAzim);
    uint32_t maxElevMaskWidth = 8*sizeof(objDetObj->commonCfg.zeroInsrtMaskCfg.zeroInsrtMaskElev);
    int32_t currLoc;
    float noisedB, signaldB, snrdB;
    float peakIdxOffset, peakIdxFlt;
    int32_t peakLoc;
    float    azimSinPhase;
    cmplxfUnion_t DFTValAzim, DFTValElev, elevOutput;
    float  elevSinPhase, elevCosPhase;
    float rangeStep, range, dopplerStep, x, ySquared, z;
    float wz,peakLocFlt, peakIdxFlt_DFT;
    int32_t dopIdx;
    uint32_t numDopplerBins = subFrmObj->staticCfg.numDopplerBins;
    int16_t ValidObjIdx;

    /* Alignment is to be done because we use the antenna calib params for
     * multiplication, using optimized DSP routines, which require a 8 byte alignment */
    cmplxfImRe_t samplesCalib[MAX_NUM_VIRT_ANT] __attribute__((aligned(8)));
    cmplx32ImRe_t rearrangedAzimSamples[MAX_NUM_AZIM_VIRT_ANT] __attribute__((aligned(8)));
    cmplx32ImRe_t rearrangedElevSamples[MAX_NUM_ELEV_VIRT_ANT] __attribute__((aligned(8)));

    int32_t retVal = 0;
    cmplxfImRe_t dftFactorsAzim[MAX_NUM_AZIM_VIRT_ANT] __attribute__((aligned(8)));
    cmplxfImRe_t dftFactorsElev[MAX_NUM_ELEV_VIRT_ANT] __attribute__((aligned(8)));
    double * azimSamplesCalib = (double *)&samplesCalib[0];
    double * elevSamplesCalib = (double *)&samplesCalib[MAX_NUM_AZIM_VIRT_ANT];
    int64_t  * restrict azimSamples;
    int64_t  * restrict elevSamples;
    double samplesFlt2;
    double *antennaCalibParams = (double *)&objDetObj->commonCfg.antennaCalibParams[0];

    rangeStep = subFrmObj->staticCfg.rangeStep;
    dopplerStep = subFrmObj->staticCfg.dopplerStep;

    /* This variable will index the final object list */
    ValidObjIdx = 0;

    for (objIdx = 0; objIdx < numObjOut; objIdx++)
    {
        /* 1. Interpolate around peak to get fractional estimate of Azimuth index */
        peakIdxOffset = DPC_ObjDet_quadInterpAroundPeak(detObjList[objIdx].azimPeakSamples);

        /* Correct peak index with the fractional index*/
        peakIdxFlt = (float)detObjList[objIdx].azimIdx + peakIdxOffset;
        peakIdxFlt_DFT = peakIdxFlt * (invAzimFFTSize * AOA_DFT_LEN);
        peakLoc = ROUND_UNSIGNED(peakIdxFlt_DFT);


        /* 2a. Calculate DFT Factors corresponding to wx for Row 1.
            *  i.e. calculate \f$\e^{j wx}\f$
            */
        idx = 0;
        for (sampIdx = 0; sampIdx < maxAzimMaskWidth; sampIdx ++)
        {
            if((objDetObj->commonCfg.zeroInsrtMaskCfg.zeroInsrtMaskAzim >> sampIdx) & 0x1U)
            {
                currLoc = (peakLoc*sampIdx)%AOA_DFT_LEN;
                dftFactorsAzim[idx++] = dftSinCosTable[currLoc];
            }

            /* Break the loop after computing all dft factors in azimuth dimension */
            if(idx == MAX_NUM_AZIM_VIRT_ANT)
                break;
        }

        /* 2b. Calculate DFT Factors corresponding to wx for Row 0.
            *  i.e. calculate \f$\e^{j wx}\f$
            */
        idx = 0;
        for (sampIdx = 0; sampIdx < maxElevMaskWidth; sampIdx ++)
        {
            if((objDetObj->commonCfg.zeroInsrtMaskCfg.zeroInsrtMaskElev >> sampIdx) & 0x1U)
            {
                currLoc = (peakLoc*sampIdx)%AOA_DFT_LEN;
                dftFactorsElev[idx++] = dftSinCosTable[currLoc];
            }

            /* Break the loop after computing all dft factors in elevation dimension */
            if(idx == MAX_NUM_ELEV_VIRT_ANT)
                break;
        }

        /* 2c. Rearrange the antenna samples according to the virtual antenna mapping. */
        for (sampIdx = 0; sampIdx < MAX_NUM_AZIM_VIRT_ANT; sampIdx ++)
        {
            rearrangedAzimSamples[sampIdx] = detObjList[objIdx].azimSamples[objDetObj->commonCfg.antennaGeometryCfg[sampIdx]];
        }

        for (sampIdx = 0; sampIdx < MAX_NUM_ELEV_VIRT_ANT; sampIdx ++)
        {
            rearrangedElevSamples[sampIdx] = detObjList[objIdx].elevSamples[objDetObj->commonCfg.antennaGeometryCfg[MAX_NUM_AZIM_VIRT_ANT+sampIdx]];
        }

        azimSamples = (int64_t*) rearrangedAzimSamples;
        elevSamples = (int64_t*) rearrangedElevSamples;

        /* 3. Azimuth Antenna Calibration:
            * Multiply azimuth samples (azimSamples) of Doppler FFT with antenna calib params (antennaCalibParams)
            */

        for (sampIdx = 0; sampIdx < MAX_NUM_AZIM_VIRT_ANT; sampIdx++)
        {
            samplesFlt2 = _dintsp(azimSamples[sampIdx]);
            azimSamplesCalib[sampIdx] =  _complex_mpysp(samplesFlt2,antennaCalibParams[sampIdx]);
        }

        /* 4. Elevation Antenna calibration
            * Multiply elev samples with antenna calib params  */
        for (sampIdx = MAX_NUM_AZIM_VIRT_ANT; sampIdx < MAX_NUM_VIRT_ANT ; sampIdx++ )
        {
            samplesFlt2 = _dintsp(elevSamples[sampIdx - MAX_NUM_AZIM_VIRT_ANT]);
            elevSamplesCalib[sampIdx-MAX_NUM_AZIM_VIRT_ANT] = _complex_mpysp(samplesFlt2, antennaCalibParams[sampIdx]);
        }

        /* 5. Single Bin DFT on the azimuth antennas to estimate phase at peak.
            *
            \f[
            X_{azim} (\omega_x) = \sum_{k=0}^{N_{azim} - 1} azimSample(k)  e^{-j k \omega_x}
            \f]
            * Multiply DFT factors with azimuth of Doppler FFT samples corrected for antenna calibration.
            */
        dotpCmplxf((float *)&azimSamplesCalib[0], (float *)&dftFactorsAzim[0], MAX_NUM_AZIM_VIRT_ANT, &DFTValAzim.cmplx.real, &DFTValAzim.cmplx.imag);


        /* 6.  Single Bin DFT on the elevation antennas to estimate phase at peak.
            *
            \f[
            X_{elev} (\omega_x) = \sum_{k=0}^{N_{elev} - 1} elevSample(k)  e^{-j (k+2) \omega_x}
            \f]
            * The elevation antennas (essentially the 4 virtual antennas corresponding to the
            * elevation offset Tx antenna) are 4 in number and offset by 3 positions from the
            * azimuth virtual array. Hence when the DFT is computed, begin from the 3rd DFT parameter.
            *
            * Both elevSamplesCalib and cosValSinVal[4] are double-word aligned. */
        dotpCmplxf((float *)&elevSamplesCalib[0], (float *)&dftFactorsElev[0], MAX_NUM_ELEV_VIRT_ANT, &DFTValElev.cmplx.real, &DFTValElev.cmplx.imag);

        /* 7. Estimate phase difference between the peak location at azimuth antennas and elevation antennas at peak.
            *  - 1. compute the conjugate product to get the phase difference (i.e. AzimVal * conj(ElevVal)) */
        elevOutput.ddat =  _complex_conjugate_mpysp (DFTValElev.ddat, DFTValAzim.ddat);


        /* - 2. Compute the angle of the product to estimate the phase change in elevation.
            \f[
            \omega_z = angle (\ X_{elev} (\omega_x)' \times X_{azim} (\omega_x) )\
            \f]
        */
        if (fabsf(elevOutput.cmplx.imag) < (0.15f*fabsf(elevOutput.cmplx.real)))
        {
            // small angle approximation.
            wz = divsp(elevOutput.cmplx.imag, elevOutput.cmplx.real);
        }
        else
        {
            wz = atan2sp(elevOutput.cmplx.imag, elevOutput.cmplx.real);
            if (wz > PI_)
            {
                    wz -= 2.0f*PI_;
            }
        }

        /* 8. Obtain range using the range resolution and the range Index  */
        range = rangeStep * (float)detObjList[objIdx].rangeIdx;

        /* 9. Obtain z, x coordinates.
            \f[
            \Phi = asin(\frac{\omega_z}{2 \pi d_z})
            \f]

        \f[
            z = range \times sin(\phi) = range * \frac{\omega_z}{2 \pi d_z}
        \f]

        */
        elevSinPhase = wz * (1.0f / (2.0f * PI_ * objDetObj->commonCfg.antennaSpacing.zSpacingByLambda));
        if ((elevSinPhase > subFrmObj->aoaFovSinVal.minElevationSinVal) && (elevSinPhase < subFrmObj->aoaFovSinVal.maxElevationSinVal))
        {
            z = range * elevSinPhase;

            /*
            \f[
                x = range  cos(\phi)  sin(\theta) =  range  /frac{\omega_x}{2 \pi d_x}
            \f]

            */
            peakLocFlt = peakLoc * (1.0f/ AOA_DFT_LEN);
            if (peakLocFlt > 0.5f)
            {
                peakLocFlt -= 1.0f;
            }

            x = range * peakLocFlt * (1.0f / objDetObj->commonCfg.antennaSpacing.xSpacingByLambda);

            /* Obtain 'square of y' coordinate
                \f[
                y^2 = range^2 -x^2 - z^2
            \f]
            */
            ySquared = (range * range) - (z * z) - (x * x);

            /* It is possible that ySquared is less than zero (i.e. a degenerate case). In such a case ignore the object.
                * If the case is not degenerate, proceed to check if the object is in the field of view (FoV).
                * If so , store the newly validated object in the final object list.*/
            if (ySquared > 0)
            {
                /* Estimate azimuth phase.
                    \f[
                    sin(\Theta) = \frac{x}{range \times cos(\Phi)}
                    \f]
                */
                elevCosPhase = sqrtsp(1 - (elevSinPhase * elevSinPhase));
                azimSinPhase = divsp(x, (range * elevCosPhase));

                /* Check if object is in azimuth FoV, If object is in FoV, proceed to store the coordinates in the final object list */
                if ((azimSinPhase > subFrmObj->aoaFovSinVal.minAzimuthSinVal) && (azimSinPhase < subFrmObj->aoaFovSinVal.maxAzimuthSinVal))
                {

                    /* Store x, y, z values */
                    objOut[ValidObjIdx].z = z;
                    objOut[ValidObjIdx].x = x;
                    objOut[ValidObjIdx].y = sqrtsp(ySquared);

                    /* Obtain and store Velocity */
                    if (detObjList[objIdx].dopIdxActual > numDopplerBins / 2)
                    {
                        dopIdx = detObjList[objIdx].dopIdxActual - numDopplerBins;
                    }
                    else
                    {
                        dopIdx = detObjList[objIdx].dopIdxActual;
                    }
                    objOut[ValidObjIdx].velocity = dopIdx * dopplerStep;

                    /* Calcute the side info of final detected object */
                    /* output is 20*log10(2)*value/2^(QVALUE) */
                    noisedB = 6.0 * ((float)detObjList[objIdx].dopCfarNoise) * (1.0f/(1 << QVALUE_NOISE));
                    signaldB = 6.0 * ((float)detObjList[objIdx].azimPeakSamples[1]) * (1.0f/(1<<QVALUE_SIGNAL));
                    snrdB = signaldB - noisedB;

                    subFrmObj->detObjOutSideInfo[ValidObjIdx].snr = (int)(10*snrdB);
                    subFrmObj->detObjOutSideInfo[ValidObjIdx].noise = (int)(10*noisedB);

                    /* Increment output list index */
                    ValidObjIdx++;
                }
            }
        } /* End of elevation FoV check cond */
    }
    *finalNumObjOut = ValidObjIdx;

    goto exit;
exit:
    return retVal;
}
#endif

/**
 *  @b Description
 *  @n
 *      Function to configure the ADC Buffer register bits
 *      as per the the RX channel offset.
 *
 *  @param[in]  channel   RX Channel (0/1/2/3)
 *  @param[in]  offset    offset for the received data
 *
 */
static void channelOffsetConfig(uint8_t channel, uint16_t offset)
{
#if defined(SOC_AWR2X44P)
    CSL_rss_ctrlRegs *ptrrssCtrlRegs = (CSL_rss_ctrlRegs *)CSL_CM4_RSS_CTRL_U_BASE;
#else
    CSL_rss_ctrlRegs *ptrrssCtrlRegs = (CSL_rss_ctrlRegs *)CSL_RSS_CTRL_U_BASE;
#endif

    switch (channel)
    {
    case 0U:

        /* Setup the offset */
        CSL_REG32_FINS_RAW(&ptrrssCtrlRegs->ADCBUFCFG2,
                           CSL_RSS_CTRL_ADCBUFCFG2_ADCBUFCFG2_ADCBUFADDRX0_MASK,
                           CSL_RSS_CTRL_ADCBUFCFG2_ADCBUFCFG2_ADCBUFADDRX0_SHIFT,
                           ((uint32_t)offset >> 4U));
        break;
    case 1U:

        /* Setup the offset */
        CSL_REG32_FINS_RAW(&ptrrssCtrlRegs->ADCBUFCFG2,
                           CSL_RSS_CTRL_ADCBUFCFG2_ADCBUFCFG2_ADCBUFADDRX1_MASK,
                           CSL_RSS_CTRL_ADCBUFCFG2_ADCBUFCFG2_ADCBUFADDRX1_SHIFT,
                           ((uint32_t)offset >> 4U));
        break;
    case 2U:

        /* Setup the offset */
        CSL_REG32_FINS_RAW(&ptrrssCtrlRegs->ADCBUFCFG3,
                           CSL_RSS_CTRL_ADCBUFCFG3_ADCBUFCFG3_ADCBUFADDRX2_MASK,
                           CSL_RSS_CTRL_ADCBUFCFG3_ADCBUFCFG3_ADCBUFADDRX2_SHIFT,
                           ((uint32_t)offset >> 4U));
        break;
    case 3U:

        /* Setup the offset */
        CSL_REG32_FINS_RAW(&ptrrssCtrlRegs->ADCBUFCFG3,
                           CSL_RSS_CTRL_ADCBUFCFG3_ADCBUFCFG3_ADCBUFADDRX3_MASK,
                           CSL_RSS_CTRL_ADCBUFCFG3_ADCBUFCFG3_ADCBUFADDRX3_SHIFT,
                           ((uint32_t)offset >> 4U));
        break;

    default:
        /* Not  supported channels, code should not end up here */
        DebugP_assert(false);
        break;
    }
}

/**
 *  @b Description
 *  @n
 *      Function to calculate the RX channel offeset based
 *      on the data size of the channel.
 *
 *  @param[in]  rxChannelEn   RX Channel Bitmap b[3:0]=>RX4:RX1
 *  @param[in]  chanDataSize  Number of bytes per RX channel
 *
 */
static void DPC_ObjectDetection_ConfigureADCBuf(
    uint16_t rxChannelEn,
    uint32_t chanDataSize)
{
    uint8_t channel;
    uint16_t offset = 0;

    /* channel offset reconfigure */

    for (channel = 0; channel < SYS_COMMON_NUM_RX_CHANNEL; channel++)
    {
        if ((rxChannelEn & ((uint16_t)0x1U << channel)) != 0U)
        {
            channelOffsetConfig(channel, offset);
            /* Calculate offset for the next channel */
            offset += (uint16_t)chanDataSize;
        }
    }
}


/**
 *  @b Description
 *  @n
 *      Computes the rx phase compensation from the detection matrix
 *      during calibration measurement procedure of these parameters.
 *
 *  @param[in]  staticCfg Pointer to static configuration
 *  @param[in]  targetDistance Target distance in meters
 *  @param[in]  searchWinSize Search window size in meters
 *  @param[in]  detMatrix Pointer to detection matrix
 *  @param[in]  detObjList Pointer to detected object list
 *  @param[in]  numObjOut Number of detected objects
 *  @param[out] compRxChanCfg computed output range bias and rx phase comp vector
 *
 *  @retval   None
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */


static void DPC_ObjDet_RxChPhaseMeasure
(
    DPC_ObjectDetection_StaticCfg       *staticCfg,
    float                   targetDistance,
    float                   searchWinSize,
    uint16_t                *detMatrix,
    DetObjParams            *detObjList,
    uint32_t                numObjOut,
    Measure_compRxChannelBiasCfg *compRxChanCfg
)
{
    float antMagSq[SYS_COMMON_NUM_RX_CHANNEL * SYS_COMMON_NUM_TX_ANTENNAS];
    float antMagSqMin;
    float scal;
    float truePosition;
    int32_t truePositionIndex;
    int32_t halfWinSize;
    uint8_t antennaIdx, elevIdx;
    uint32_t objIdx, rangeidx;
    int32_t iMaxPos =-1, objIdxMax =-1;
    int32_t temp;
    float temp_f;

    uint16_t numDopFFTSubBins = staticCfg->numDopplerBins / staticCfg->numBandsTotal;
    uint16_t maxVal = 0;

    truePosition = targetDistance / staticCfg->rangeStep;
    temp_f = truePosition + 0.5F;
    truePositionIndex = (int32_t) (temp_f);

    temp_f = 0.5F * searchWinSize / staticCfg->rangeStep + 0.5F;
    halfWinSize = (int32_t) (temp_f);

    /**** Strongest target position index ****/
    for(objIdx=0; objIdx< numObjOut; objIdx++){
        /** for all detected objects, if object is in the target distance range,
         * find object with maximum SNR in that range */
        rangeidx = detObjList[objIdx].rangeIdx;

        if(( (int32_t)rangeidx >= (truePositionIndex - halfWinSize)) && ((int32_t)rangeidx <= (truePositionIndex + halfWinSize))){
            /* doppler bin =0 ; considering target at zero velocity*/
            if (detMatrix[rangeidx*numDopFFTSubBins] > maxVal)
            {
                maxVal = detMatrix[rangeidx * numDopFFTSubBins];
                iMaxPos = (int32_t)rangeidx;
                objIdxMax = (int32_t)objIdx;
            }
        }
    }

    /*should not be first range bin (0) or -1 (error) */
    if(iMaxPos>0){
     /*** Calculate antenna normalization coefficients ***/
        for (antennaIdx = 0; antennaIdx < staticCfg->numVirtualAntennas; antennaIdx++)
        {
            if(antennaIdx < staticCfg->numVirtualAntAzim){
                antMagSq[antennaIdx] = (float) detObjList[objIdxMax].azimSamples[antennaIdx].real * (float) detObjList[objIdxMax].azimSamples[antennaIdx].real +
                            (float) detObjList[objIdxMax].azimSamples[antennaIdx].imag * (float) detObjList[objIdxMax].azimSamples[antennaIdx].imag;
            }
            else{
                elevIdx = antennaIdx - staticCfg->numVirtualAntAzim;
                antMagSq[antennaIdx] = (float) detObjList[objIdxMax].elevSamples[elevIdx].real * (float) detObjList[objIdxMax].elevSamples[elevIdx].real +
                            (float) detObjList[objIdxMax].elevSamples[elevIdx].imag * (float) detObjList[objIdxMax].elevSamples[elevIdx].imag;
            }
        }

        if(staticCfg->numVirtualAntennas > 0U)
        {
            antMagSqMin = antMagSq[0];
            for (antennaIdx = 1; antennaIdx < staticCfg->numVirtualAntennas; antennaIdx++)
            {
                if (antMagSq[antennaIdx] < antMagSqMin)
                {
                    antMagSqMin = antMagSq[antennaIdx];
                }
            }

            for (antennaIdx = 0; antennaIdx < staticCfg->numVirtualAntennas; antennaIdx++)
            {
                scal = 16384.0F/ antMagSq[antennaIdx] * (float)sqrt((float)antMagSqMin);

                if(antennaIdx < staticCfg->numVirtualAntAzim){
                    temp_f = MATHUTILS_ROUND_FLOAT(scal * (float)detObjList[objIdxMax].azimSamples[antennaIdx].real);
                    temp = (int32_t) (temp_f);
                    MATHUTILS_SATURATE16(temp);
                    compRxChanCfg->rxChPhaseComp[antennaIdx].real = (int16_t) (temp);

                    temp_f = MATHUTILS_ROUND_FLOAT(-scal * (float)detObjList[objIdxMax].azimSamples[antennaIdx].imag);
                    temp = (int32_t) (temp_f);
                    MATHUTILS_SATURATE16(temp);
                    compRxChanCfg->rxChPhaseComp[antennaIdx].imag = (int16_t) (temp);
                }

                else{
                    elevIdx = antennaIdx - staticCfg->numVirtualAntAzim;
                    temp_f = MATHUTILS_ROUND_FLOAT(scal * (float)detObjList[objIdxMax].elevSamples[elevIdx].real);
                    temp = (int32_t) (temp_f);
                    MATHUTILS_SATURATE16(temp);
                    compRxChanCfg->rxChPhaseComp[antennaIdx].real = (int16_t) (temp);

                    temp_f = MATHUTILS_ROUND_FLOAT(-scal * (float)detObjList[objIdxMax].elevSamples[elevIdx].imag);
                    temp = (int32_t) (temp_f);
                    MATHUTILS_SATURATE16(temp);
                    compRxChanCfg->rxChPhaseComp[antennaIdx].imag = (int16_t) (temp);
                }
            }
            compRxChanCfg->targetRange = (float)detObjList[objIdxMax].rangeIdx * staticCfg->rangeStep;
            compRxChanCfg->peakVal = maxVal;
        }
    }
    else{
        /* target object not found */
        for (antennaIdx = 0; antennaIdx < staticCfg->numVirtualAntennas; antennaIdx++)
        {
                compRxChanCfg->rxChPhaseComp[antennaIdx].real = 16384;
                compRxChanCfg->rxChPhaseComp[antennaIdx].imag = 0;
        }
        compRxChanCfg->targetRange = -1.0F;
        compRxChanCfg->peakVal = 0;
    }
}

/**
 *  @b Description
 *  @n
 *     Check the FFT clip status.
 *
 *  @param[in]  objDetObj Pointer to DPC object
 *  @param[out] clipCount FFT Clip Count
 *
 *  @retval   None
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */

static void checkFFTClipStatus(ObjDetObj *objDetObj, uint32_t* clipCount)
{
    uint16_t clipStatusResult = 0U;

    /* Check the FFT clip Register after range fft */
    (void)HWA_readClipStatus(objDetObj->hwaHandle, &clipStatusResult, HWA_CLIPREG_TYPE_FFT);
    /* Is FFT clipped? Yes: increment the clip count */
    if(clipStatusResult!=0U)
    {
        (*clipCount)++;
    }
    /* Clear the FFT clip register */
    (void)HWA_clearClipStatus(objDetObj->hwaHandle, HWA_CLIPREG_TYPE_FFT);
}

/**
 *  @b Description
 *  @n
 *      Creates the final detected object list with the objects that
 *     are present in both doppler nad range cfar detection list.
 *
 *  @param[in]  objDetObj Pointer to DPC object
 *  @param[in]  subFrmObj Pointer to subframe object
 *  @param[in]  dopNumObjOut Number of detected objects by Doppler DPU
 *  @param[in]  detObjList Pointer to detected object list by Doppler DPU
 *  @param[out] finalNumDetObjs Final number of detected objects after intersection
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjDet_intersectDopAndRangeCFAR(
    ObjDetObj *objDetObj,
    SubFrameObj *subFrmObj,
    uint32_t dopNumObjOut,
    DetObjParams *detObjList,
    uint32_t *finalNumDetObjs
)
{
    int32_t retVal=0;
    uint16_t valSubBinObj, cfarListStartIdx;
    uint32_t isValidObj, objIdx, dopIdx, finalNumObjs = 0;
    uint16_t * rangeCfarObjPerDopList;
    uint32_t baseAddr = EDMA_getBaseAddr(objDetObj->edmaHandle[0]);
    uint32_t edmaSrcAddr, edmaDstAddr, edmaTrigReg, edmaIntrStatusReg, edmaClrIntrStatusReg, channelMask;
    edmaSrcAddr = baseAddr + EDMA_TPCC_OPT(objDetObj->edmaDetObjs.channel) + 0x4U;
    edmaDstAddr = edmaSrcAddr + 0x8U;
    edmaTrigReg = baseAddr + EDMA_TPCC_ESR_RN(0U);
    edmaIntrStatusReg =  baseAddr + EDMA_TPCC_IPR_RN(0U);
    edmaClrIntrStatusReg = baseAddr + EDMA_TPCC_ICR_RN(0U);
    channelMask = (uint32_t)1U << objDetObj->edmaDetObjs.channel;

    /* As the detected object list is written on radar cube, find the intersection of range-cfar
       and doppler proc stage here, and copy the result to Local Scratch Buf before triggering range proc.
       As range processing will result in overwritten of detObjList by the radar cube.
       This step is required as Elev Estimation has to happen in parallel with Range Proc. */
    if(subFrmObj->staticCfg.rangeCfarCfg.cfg.isEnabled)
    {
        rangeCfarObjPerDopList = (uint16_t *)subFrmObj->dpuCfg.rangeCfarCfg.res.rangeCfarNumObjPerDopplerBinBuf;
        for (objIdx = 0; objIdx < dopNumObjOut; objIdx++)
        {
            dopIdx = detObjList[objIdx].dopIdx;

            if (dopIdx > 0U)
            {
                /* Obtain the number of objects for a sub bin in the Range CFAR list by
                * subtracting two consecutive elements from the cummulative distribution */
                valSubBinObj = rangeCfarObjPerDopList[dopIdx] - rangeCfarObjPerDopList[dopIdx - 1U];
                cfarListStartIdx = rangeCfarObjPerDopList[dopIdx - 1U];
            }
            else
            {
                /* For the 0th sub bin, do not perform a subtraction */
                valSubBinObj = rangeCfarObjPerDopList[dopIdx];
                cfarListStartIdx = 0;
            }

            /* If the sub bin has valid objects in the range CFAR list, check whether the
            * object currently being looked at (from the Doppler CFAR list), is also available
            * in the range CFAR list. cfarListStartIdx tells us where to start searching for in
            * the range CFAR list, and valSubBinObj tells us how many elements to search in.
            * This saves computation time. */
            if (valSubBinObj)
            {
                isValidObj = isObjInRangeAndDopplerList((uint32_t)detObjList[objIdx].rangeIdx,
                                                        detObjList[objIdx].dopIdx,
                                                        (RangeCfarListObj *)&subFrmObj->dpuCfg.rangeCfarCfg.res.rangeCfarList[cfarListStartIdx],
                                                        valSubBinObj);

                if(isValidObj)
                {
                    /* Check the completion of previous transfer before triggering the next. */
                    if(finalNumObjs > 0U)
                    {
                        while(((*(volatile uint32_t*)((uint32_t)edmaIntrStatusReg)) & (channelMask)) != (channelMask))
                        {
                            /* wait */
                        }
                        *(volatile uint32_t*)((uint32_t)edmaClrIntrStatusReg) = channelMask;
                    }


                    /* EDMA this obj to L2 for further processing */

                    /* update src address */
                    *(volatile uint32_t*)((uint32_t)edmaSrcAddr) = (uint32_t)SOC_virtToPhy((void*)&subFrmObj->dpuCfg.dopplerCfg.hwRes.detObjList[objIdx]);

                    /* update dst address */
                    *(volatile uint32_t*)((uint32_t)edmaDstAddr) = (uint32_t)SOC_virtToPhy((void*)&subFrmObj->dpuCfg.dopplerCfg.hwRes.finalDetObjList[finalNumObjs]);

                    /* trigger */
                    *(volatile uint32_t*)((uint32_t)edmaTrigReg) = channelMask;

                    finalNumObjs++;
                    if(finalNumObjs >= subFrmObj->dpuCfg.dopplerCfg.hwRes.finalMaxNumDetObjs)
                    {
                        break;
                    }
                }
            }
        }

        /* Monitor the completion of last transfer here. */
        if(finalNumObjs > 0U)
        {
            while(((*(volatile uint32_t*)((uint32_t)edmaIntrStatusReg)) & (channelMask)) != (channelMask))
            {
                /* wait */
            }
            *(volatile uint32_t*)((uint32_t)edmaClrIntrStatusReg) = channelMask;
        }
    }
    else
    {
        /* Copy the entire detected object list to L2.
           We can copy only a limited number of objects (set to finalMaxNumDetObjs).
           This constraint is due to limited available L2 buffer memory. */
        finalNumObjs = (dopNumObjOut > subFrmObj->dpuCfg.dopplerCfg.hwRes.finalMaxNumDetObjs)? subFrmObj->dpuCfg.dopplerCfg.hwRes.finalMaxNumDetObjs : dopNumObjOut;

        /* Check if only A-dim transfer is sufficient to do this. If not, increase the number of B-dim transfers. */
        uint32_t acnt = finalNumObjs * sizeof(DetObjParams);
        uint32_t bcnt = 1;
        while( acnt > 65535U )
        {
            acnt /= 2U;
            bcnt *= 2U;
        }

        uint32_t bcnt_acnt = ((uint32_t)bcnt << 16U) | (uint32_t)acnt;
        EDMA_dmaSetPaRAMEntry(baseAddr, objDetObj->edmaDetObjs.channel, EDMACC_PARAM_ENTRY_ACNT_BCNT, bcnt_acnt);
        uint32_t dstBidx_srcBidx = ((uint32_t)acnt << 16U) | (uint32_t)acnt;
        EDMA_dmaSetPaRAMEntry(baseAddr, objDetObj->edmaDetObjs.channel, EDMACC_PARAM_ENTRY_SRC_DST_BIDX, dstBidx_srcBidx);

        *(volatile uint32_t*)((uint32_t)edmaTrigReg) = channelMask;
        while(((*(volatile uint32_t*)((uint32_t)edmaIntrStatusReg)) & (channelMask)) != (channelMask))
        {
            /* wait */
        }
        *(volatile uint32_t*)((uint32_t)edmaClrIntrStatusReg) = channelMask;
    }

    *finalNumDetObjs = finalNumObjs;

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      DPC's (DPM registered) execute function which is invoked by the application
 *      in the DPM's execute context when the DPC issues DPM_notifyExecute API from
 *      its registered @ref DPC_ObjectDetection_frameStart API that is invoked every
 *      frame interrupt.
 *
 *  @param[in]  handle       DPM's DPC handle
 *  @param[out]  ptrResult   Pointer to the result
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */

int32_t DPC_ObjectDetection_execute(
    DPM_DPCHandle   handle,
    DPM_Buffer *ptrResult)
{
    ObjDetObj   *objDetObj;
    SubFrameObj *subFrmObj;
    DPU_RangeProcHWA_OutParams outRangeProc;
    DPU_DopplerProcHWA_OutParams outDopplerProc;
    DPU_RangeCFARProcHWA_OutParams outRangeCfarProc;
    DetObjParams * detObjList;
    DPIF_PointCloudCartesian * objOut;
    uint32_t saveRestoreDataSize;
    int32_t retVal;
    DPC_ObjectDetection_ExecuteResult *result;
    DPC_ObjectDetection_ProcessCallBackCfg *processCallBack;
    int32_t i;

    objDetObj = (ObjDetObj *) handle;
    DebugP_assert (objDetObj != NULL);
    DebugP_assert (ptrResult != NULL);

#ifndef INCLUDE_DPM
    (void)SemaphoreP_pend(&objDetObj->dpcExecSemHandle, SystemP_WAIT_FOREVER);
#endif

    processCallBack = &objDetObj->processCallBackCfg;

    if (processCallBack->processFrameBeginCallBackFxn != NULL)
    {
        (*processCallBack->processFrameBeginCallBackFxn)(objDetObj->subFrameIndx);
    }

    result = &objDetObj->executeResult;

    subFrmObj = &objDetObj->subFrameObj[objDetObj->subFrameIndx];

    /* Cache invalidation is required to mitigate incoherency
     * issues associated with EDMA transfer from/to L3. */
    CacheP_wbInvAll(CacheP_TYPE_ALL);

    retVal = DPU_RangeProcHWA_process(subFrmObj->dpuRangeObj,  &subFrmObj->dpuCfg.rangeCfg, &outRangeProc);
    if (retVal != 0)
    {
        goto exit;
    }
    DebugP_assert(outRangeProc.endOfChirp == true);

    checkFFTClipStatus(objDetObj, &result->FFTClipCount[0]);

    if (processCallBack->processInterFrameBeginCallBackFxn != NULL)
    {
        (*processCallBack->processInterFrameBeginCallBackFxn)(objDetObj->subFrameIndx);
    }

    objDetObj->stats.interFrameStartTimeStamp = CycleCounterP_getCount32();

#ifdef INCLUDE_DPM
    DPC_Objdet_Assert(objDetObj->dpmHandle, (objDetObj->interSubFrameProcToken == 0));
#else
    DebugP_assert(objDetObj->interSubFrameProcToken == 0);
#endif
    objDetObj->interSubFrameProcToken++;

#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    gTimingInfo.rangeEndTimes[gTimingInfo.rangeEndCnt % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE] = CycleCounterP_getCount32();
    gTimingInfo.rangeEndCnt++;
#endif

#ifdef SOC_AWR2X44P
    CSL_dss_rcmRegs *ptrDssRcmRegs = (CSL_dss_rcmRegs *)CSL_CM4_DSS_RCM_U_BASE;

    if (subFrmObj->staticCfg.powerOptCfg.dspStateAfterFrameProc == DPC_OBJDET_DSP_PG_ENABLE)
    {
        if ((ptrDssRcmRegs->DSP_PD_STATUS & DPC_OBJDET_DSP_PD_STATUS_MASK) == DPC_OBJDET_DSP_POWERED_DOWN)
        {
            /* Trigger Wakeup */
            ptrDssRcmRegs->DSP_PD_TRIGGER_WAKUP |= 0x1U;
            while((ptrDssRcmRegs->DSP_PD_STATUS & DPC_OBJDET_DSP_PD_STATUS_MASK) != DPC_OBJDET_DSP_POWERED_UP)
            {
                /* Wait for DSP power up */
            }
        }
    }
    else if (subFrmObj->staticCfg.powerOptCfg.dspStateAfterFrameProc == DPC_OBJDET_DSP_UC_ENABLE)
    {
        /* Switch DSP core clock back to normal rate */
        ptrDssRcmRegs->DSS_DSP_CLK_SRC_SEL = DPC_OBJDET_DSP_CLK_SRC_DSP_PLL_MUX;
    }
#endif
    retVal = DPU_DopplerProcHWA_process(subFrmObj->dpuDopplerObj, &subFrmObj->dpuCfg.dopplerCfg, &outDopplerProc);
    if (retVal != 0)
    {
        goto exit;
    }
#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    gTimingInfo.dopEndTimes[gTimingInfo.dopEndCnt % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE] = CycleCounterP_getCount32();
    gTimingInfo.dopEndCnt++;
#endif

    checkFFTClipStatus(objDetObj, &result->FFTClipCount[1]);

    if (subFrmObj->staticCfg.rangeCfarCfg.cfg.isEnabled)
    {
        retVal = DPU_RangeCFARProcHWA_process(subFrmObj->dpuRangeCfarObj, &subFrmObj->dpuCfg.rangeCfarCfg, &outRangeCfarProc);
        if (retVal != 0)
        {
            goto exit;
        }
    }

    detObjList = subFrmObj->dpuCfg.dopplerCfg.hwRes.detObjList;
    objOut     = subFrmObj->dpuCfg.dopplerCfg.hwRes.objOut;

    /* Procedure for Rx channels gain/phase offset measurement */
    if(objDetObj->commonCfg.measureRxChannelBiasCfg.enabled)
    {
        DPC_ObjDet_RxChPhaseMeasure(&subFrmObj->staticCfg,
            objDetObj->commonCfg.measureRxChannelBiasCfg.targetDistance,
            objDetObj->commonCfg.measureRxChannelBiasCfg.searchWinSize,
            subFrmObj->dpuCfg.dopplerCfg.hwRes.detMatrix.data,
            detObjList,
            outDopplerProc.numObjOut,
            &objDetObj->compRxChanCfgMeasureOut);
    }

    retVal = DPC_ObjDet_intersectDopAndRangeCFAR(objDetObj, subFrmObj, outDopplerProc.numObjOut, detObjList, &result->dopNumObjOut) ;
    if (retVal < 0)
    {
        goto exit;
    }

    /********************************
     * Prepare for subFrame switch
     *******************************/
    if (objDetObj->commonCfg.numSubFrames > 1U)
    {
        uint8_t nextSubFrameIdx;
        SubFrameObj *nextSubFrmObj;

        DPC_ObjectDetection_ADCBufConfig nextSubFrameADCBufConfig;

        if (objDetObj->subFrameIndx == (objDetObj->commonCfg.numSubFrames - 1U))
        {
            nextSubFrameIdx = 0;
        }
        else
        {
            nextSubFrameIdx = objDetObj->subFrameIndx + 1U;
        }
        /* get next subframe objDetObj */
        nextSubFrmObj = &objDetObj->subFrameObj[nextSubFrameIdx];

        if(objDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_FRAME_DC_MODE)
        {
            /* In this rangeProcChain, if subframe switching is happening, 
             * corresponding subframe indices' DC Estimation statistics need to be loaded and stored. */
            saveRestoreDataSize = (uint32_t)subFrmObj->staticCfg.ADCBufData.dataProperty.numRxAntennas * 4U;
            if(objDetObj->commonCfg.rangeProcCfg.isReal2XEnabled)
            {
                /* In Real2X mode, number of elements is halved */
                saveRestoreDataSize >>= 1;
            }
            rangeProcHWA_storePreProcStats(&subFrmObj->dpuCfg.rangeCfg, saveRestoreDataSize, 0, 0);
            rangeProcHWA_loadPreProcStats(&nextSubFrmObj->dpuCfg.rangeCfg, saveRestoreDataSize, 0, 0);
        }

        nextSubFrameADCBufConfig = nextSubFrmObj->staticCfg.ADCBufConfig;
        /* Configure ADC for next sub-frame */
        DPC_ObjectDetection_ConfigureADCBuf(
            nextSubFrameADCBufConfig.rxChannelEn,
            nextSubFrameADCBufConfig.adcBufChanDataSize);
        (void)DPC_ObjDet_reconfigSubFrame(objDetObj, nextSubFrameIdx);

        /* Trigger Range DPU for the next sub frame */
        retVal = DPU_RangeProcHWA_control(nextSubFrmObj->dpuRangeObj, &nextSubFrmObj->dpuCfg.rangeCfg,
                                          DPU_RangeProcHWA_Cmd_triggerProc, NULL, 0);
        if (retVal < 0)
        {
            goto exit;
        }
    }
    else
    {
        /* Trigger Range DPU for the next frame */
        retVal = DPU_RangeProcHWA_control(subFrmObj->dpuRangeObj, &subFrmObj->dpuCfg.rangeCfg,
                                          DPU_RangeProcHWA_Cmd_triggerProc, NULL, 0);
        if (retVal < 0)
        {
            goto exit;
        }
    }

#ifdef SUBSYS_DSS
#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    gTimingInfo.aoaStartTimes[gTimingInfo.aoaStartCnt % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE] = CycleCounterP_getCount32();
    gTimingInfo.aoaStartCnt++;
#endif

    retVal = DPC_ObjDet_estimateXYZ(subFrmObj, objDetObj, subFrmObj->dpuCfg.dopplerCfg.hwRes.finalDetObjList, objOut, result->dopNumObjOut, &result->numObjOut);
    if (retVal != 0)
    {
        goto exit;
    }
#endif
#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    gTimingInfo.aoaEndTimes[gTimingInfo.aoaEndCnt % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE] = CycleCounterP_getCount32();
    gTimingInfo.aoaEndCnt++;
#endif

	/* Set DPM result */
    result->subFrameIdx = objDetObj->subFrameIndx;
    result->objOut      = objOut;
    result->objOutSideInfo  = subFrmObj->detObjOutSideInfo;
    result->detMatrix   = subFrmObj->dpuCfg.dopplerCfg.hwRes.detMatrix;
    result->detObjList = subFrmObj->dpuCfg.dopplerCfg.hwRes.finalDetObjList;

    if (objDetObj->commonCfg.measureRxChannelBiasCfg.enabled == 1U)
    {
        result->compRxChanBiasMeasurement = &objDetObj->compRxChanCfgMeasureOut;
    }
    else
    {
        result->compRxChanBiasMeasurement = NULL;
    }

    /* For rangeProcHwa, interChirpProcessingMargin is not available */
    objDetObj->stats.interChirpProcessingMargin = 0;

    objDetObj->stats.interFrameEndTimeStamp = CycleCounterP_getCount32();
    result->stats = (DPC_ObjectDetection_Stats *)((uint32_t)SOC_virtToPhy((void*)&objDetObj->stats));

    /* populate DPM_resultBuf - first pointer and size are for results of the
     * processing */
    ptrResult->ptrBuffer[0] = (uint8_t *)result;
    ptrResult->size[0] = sizeof(DPC_ObjectDetection_ExecuteResult);

    /* clear rest of the result */
    for (i = 1; i < DPM_MAX_BUFFER; i++)
    {
        ptrResult->ptrBuffer[i] = NULL;
        ptrResult->size[i] = 0;
    }

#ifndef INCLUDE_DPM
#ifdef SOC_AWR2X44P
    /* HWA Clk Gate */
    subFrmObj = &objDetObj->subFrameObj[objDetObj->subFrameIndx];
    if (subFrmObj->staticCfg.powerOptCfg.hwaStateAfterFrameProc == DPC_OBJDET_HWA_CG_ENABLE)
    {
        /* Gate HWA Peripheral Clock */
        ptrDssRcmRegs->DSS_HWA_CLK_GATE = DPC_OBJDET_HWA_CLOCK_GATE;
    }
#endif
    /* Increment the subframe index to get the next subrame object */
    if (objDetObj->commonCfg.numSubFrames > 1U)
    {
        /* Next sub-frame */
        objDetObj->subFrameIndx++;
        if (objDetObj->subFrameIndx == objDetObj->commonCfg.numSubFrames)
        {
            objDetObj->subFrameIndx = 0;
        }
    }
    /* Mark the end of DPC Processing for the current frame/ subframe. */
    objDetObj->interSubFrameProcToken--;
#endif

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      Sub-frame reconfiguration, used when switching sub-frames. Invokes the
 *      DPU configuration using the configuration that was stored during the
 *      pre-start configuration so reconstruction time is saved  because this will
 *      happen in real-time.
 *  @param[in]  objDetObj Pointer to DPC object
 *  @param[in]  subFrameIndx Sub-frame index.
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 *
 * \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjDet_reconfigSubFrame(ObjDetObj *objDetObj, uint8_t subFrameIndx)
{
    int32_t retVal = 0;
    SubFrameObj *subFrmObj;

    subFrmObj = &objDetObj->subFrameObj[subFrameIndx];

    DPC_ObjDet_GenRangeWindow(&subFrmObj->dpuCfg.rangeCfg);

    retVal = DPU_RangeProcHWA_config(subFrmObj->dpuRangeObj, &subFrmObj->dpuCfg.rangeCfg);
    if (retVal != 0)
    {
        goto exit;
    }

    (void)DPC_ObjDet_GenDopplerWindow(&subFrmObj->dpuCfg.dopplerCfg);
    retVal = DPU_DopplerProcHWA_config(subFrmObj->dpuDopplerObj, &subFrmObj->dpuCfg.dopplerCfg, 1);
    if (retVal != 0)
    {
        goto exit;
    }

    if (subFrmObj->staticCfg.rangeCfarCfg.cfg.isEnabled)
    {
        retVal = DPU_RangeCFARProcHWA_config(subFrmObj->dpuRangeCfarObj, &subFrmObj->dpuCfg.rangeCfarCfg);
        if (retVal != 0)
        {
            goto exit;
        }
    }

    retVal = DPC_ObjectDetection_configEdmaDetObjsOut(objDetObj->edmaHandle[DPC_OBJDET_DPU_DOPPLERPROC_EDMA_INST_ID],
                                                    &subFrmObj->dpuCfg.dopplerCfg.hwRes,
                                                    &objDetObj->edmaDetObjs);
    if(retVal != 0)
    {
        goto exit;
    }

exit:
    return(retVal);
}

/**
 *  @b Description
 *  @n
 *      DPC's (DPM registered) start function which is invoked by the
 *      application using DPM_start API.
 *
 *  @param[in]  handle  DPM's DPC handle
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t DPC_ObjectDetection_start (DPM_DPCHandle handle)
{
    ObjDetObj   *objDetObj;
    SubFrameObj *subFrmObj;
    uint32_t dcEstBufSize = 0;
    int32_t retVal = 0;

    objDetObj = (ObjDetObj *) handle;
    DebugP_assert (objDetObj != NULL);

    objDetObj->stats.frameStartIntCounter = 0;
    objDetObj->stats.subframeStartIntCounter = 0;
    objDetObj->numTimesResultExported = 0;
    (void)memset((void*)&objDetObj->executeResult.FFTClipCount[0], 0, sizeof(objDetObj->executeResult.FFTClipCount));

#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    (void)memset((void*)&gTimingInfo, 0, sizeof(timingInfo));
#endif

    /* Start marks consumption of all pre-start configs, reset the flag to check
     * if pre-starts were issued only after common config was issued for the next
     * time full configuration happens between stop and start */
    objDetObj->isCommonCfgReceived = false;

    /* App must issue export of last frame after stop which will switch to sub-frame 0,
     * so start should always see sub-frame indx of 0, check */
    DebugP_assert(objDetObj->subFrameIndx == 0U);

    /* Pre-start cfgs for sub-frames may have come in any order, so need
     * to ensure we reconfig for the current (0) sub-frame before starting */
    (void)DPC_ObjDet_reconfigSubFrame(objDetObj, objDetObj->subFrameIndx);

    /* Initialize HWA DC estimate register for the first subframe
     * For subsequent subframes, this is done in the DPU.
    */
    if((objDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_FRAME_DC_MODE)
        && (objDetObj->commonCfg.numSubFrames > 1U))
    {
        dcEstBufSize = objDetObj->subFrameObj[0].staticCfg.ADCBufData.dataProperty.numRxAntennas * sizeof(uint32_t);
        if(objDetObj->commonCfg.rangeProcCfg.isReal2XEnabled)
        {
            dcEstBufSize >>= 1;
        }
        rangeProcHWA_loadPreProcStats(&objDetObj->subFrameObj[0].dpuCfg.rangeCfg, dcEstBufSize, 0, 0);
    }

    /* Trigger Range DPU, related to reconfig above */
    subFrmObj = &objDetObj->subFrameObj[objDetObj->subFrameIndx];

    retVal = DPU_RangeProcHWA_control(subFrmObj->dpuRangeObj, &subFrmObj->dpuCfg.rangeCfg,
                 DPU_RangeProcHWA_Cmd_triggerProc, NULL, 0);
    if(retVal < 0)
    {
        goto exit;
    }

    DebugP_logInfo("ObjDet DPC: Start done\n");
exit:
    return(retVal);
}

static void ObjectDetection_freeDmaChannels(EDMA_Handle  edmaHandle)
{
    uint32_t   index;
    uint32_t  dmaCh, tcc, pram, shadow;

    for(index = 0; index < 64U; index++)
    {
        dmaCh = index;
        tcc = index;
        pram = index;
        shadow = index;

        DPEDMA_freeEDMAChannel(edmaHandle, &dmaCh, &tcc, &pram, &shadow);

    }

    for(index = 0; index < 128U; index++)
    {
        shadow = index;
        DebugP_assert(EDMA_freeParam(edmaHandle, &shadow) == SystemP_SUCCESS);
    }

    return;
}

/**
 *  @b Description
 *  @n
 *      DPC's (DPM registered) stop function which is invoked by the
 *      application using DPM_stop API.
 *
 *  @param[in]  handle  DPM's DPC handle
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t DPC_ObjectDetection_stop (DPM_DPCHandle handle)
{
    ObjDetObj   *objDetObj;

    objDetObj = (ObjDetObj *) handle;
    DebugP_assert (objDetObj != NULL);

    /* print the FFT clip status */
    if(objDetObj->executeResult.FFTClipCount[0]>0U)
    {
        DebugP_log("Warning! FFT clipping happened for %d times in Range FFT Stage. \n", objDetObj->executeResult.FFTClipCount[0]);
    }
    if(objDetObj->executeResult.FFTClipCount[1]>0U)
    {
        DebugP_log("Warning! FFT clipping happened for %d times in Doppler or Azimuth FFT Stage. \n", objDetObj->executeResult.FFTClipCount[1]);
    }

#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
    uint32_t i, frame0StartTime;
    frame0StartTime = gTimingInfo.frameStartTimes[(gTimingInfo.frameCnt) % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE];
    for (i = 0; i < OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE; i++)
    {
        gTimingInfo.frameStartTimes[i] -= frame0StartTime;
        gTimingInfo.frameStartTimes[i] /= OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ;
        gTimingInfo.rangeEndTimes[i] -= frame0StartTime;
        gTimingInfo.rangeEndTimes[i] /= OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ;
        gTimingInfo.dopEndTimes[i] -= frame0StartTime;
        gTimingInfo.dopEndTimes[i] /= OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ;
        gTimingInfo.aoaStartTimes[i] -= frame0StartTime;
        gTimingInfo.aoaStartTimes[i] /= OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ;
        gTimingInfo.aoaEndTimes[i] -= frame0StartTime;
        gTimingInfo.aoaEndTimes[i] /= OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ;
        gTimingInfo.resEndTimes[i] -= frame0StartTime;
        gTimingInfo.resEndTimes[i] /= OBJECTDETHWA_TIMING_CPU_CLK_FREQ_KHZ;
    }

    DebugP_logInfo("\n");
    DebugP_logInfo("----DPU Timing Info (ms)----\n");
    DebugP_logInfo("%10s|%10s|%10s|%10s|%10s|%10s\n","FrameStart","RangeEnd","DopEnd","AoAStart","AoAEnd","ResEnd");

    // CacheP_wbInv((void *)&gDpc[0], sizeof(gFrameStartTimes), CacheP_TYPE_ALL);
    for (i = 0; i < OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE; i++)
    {
        DebugP_logInfo("%10d|%10d|%10d|%10d|%10d|%10d\n", gTimingInfo.frameStartTimes[i], gTimingInfo.rangeEndTimes[i], gTimingInfo.dopEndTimes[i], gTimingInfo.aoaStartTimes[i], gTimingInfo.aoaEndTimes[i], gTimingInfo.resEndTimes[i]);
    }
    DebugP_logInfo("-----------\n");
#endif

    /* We can be here only after complete frame processing is done, which means
     * processing token must be 0 and subFrameIndx also 0  */
    return(0);
}

/**
 *  @b Description
 *  @n
 *     Configure range DPU.
 *
 *  @param[in]  dpuHandle Handle to DPU
 *  @param[in]  staticCfg Pointer to static configuration of the sub-frame
 *  @param[in]  edmaHandle Handle to edma driver to be used for the DPU
 *  @param[in]  radarCube Pointer to DPIF radar cube, which is output of range
 *                        processing.
 *  @param[in]  CoreLocalRamObj Pointer to core local RAM object to allocate local memory
 *              for the DPU, only for scratch purposes
 *  @param[in]  WinBufRamObj Pointer to core local RAM object to allocate window memory
 *              for the DPU
 *  @param[in,out]  windowOffset Window coefficients that are generated by this function
 *                               (in heap memory) are passed to DPU configuration API to
 *                               configure the HWA window RAM starting from this offset.
 *                               The end offset after this configuration will be returned
 *                               in this variable which could be the begin offset for the
 *                               next DPU window RAM.
 *  @param[out]  CoreLocalRamScratchUsage Core Local RAM's scratch usage in bytes
 *  @param[out] cfgSave Configuration that is built in local
 *                      (stack) variable is saved here. This is for facilitating
 *                      quick reconfiguration later without having to go through
 *                      the construction of the configuration.
 *  @param[in]  ptrObjDetObj Pointer to object detection object
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjDet_rangeConfig(DPU_RangeProcHWA_Handle dpuHandle,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
                   EDMA_Handle                   edmaHandle,
                   DPIF_RadarCube                *radarCube,
                   MemPoolObj                    *CoreLocalRamObj,
                   MemPoolObj                    *WinBufRamObj,
                   uint32_t                      *windowOffset,
                   uint32_t                      *CoreLocalRamScratchUsage,
                   DPU_RangeProcHWA_Config       *cfgSave,
                   ObjDetObj                     *ptrObjDetObj)
{
    int32_t retVal = 0;
    DPU_RangeProcHWA_HW_Resources *hwRes = &cfgSave->hwRes;
    DPU_RangeProcHWA_EDMAInputConfig *edmaIn = &hwRes->edmaInCfg;
    DPU_RangeProcHWA_EDMAOutputConfig *edmaOut = &hwRes->edmaOutCfg;
    DPU_RangeProcHWA_HwaConfig *hwaCfg = &hwRes->hwaCfg;
    int32_t *windowBuffer;
    uint32_t winGenLen;
    uint32_t dcEstNumSamples, interfStatsNumSamples;

    (void)memset(cfgSave, 0, sizeof(DPU_RangeProcHWA_Config));

    /* static configuration */
    cfgSave->staticCfg.ADCBufData         = staticCfg->ADCBufData;
    cfgSave->staticCfg.numChirpsPerFrame  = staticCfg->numChirpsPerFrame;
    cfgSave->staticCfg.numRangeBins       = staticCfg->numRangeBins;
    cfgSave->staticCfg.numFFTBins         = staticCfg->numRangeFFTBins;
    cfgSave->staticCfg.numTxAntennas      = staticCfg->numTxAntennas;
    cfgSave->staticCfg.numVirtualAntennas = staticCfg->numVirtualAntennas;
    cfgSave->staticCfg.numBandsTotal      = staticCfg->numBandsTotal;

    if (cfgSave->staticCfg.numRangeBins == cfgSave->staticCfg.numFFTBins)
    {
        cfgSave->staticCfg.isChirpDataReal    = 0;
    }
    else if (cfgSave->staticCfg.numRangeBins == cfgSave->staticCfg.numFFTBins / 2U)
    {
        cfgSave->staticCfg.isChirpDataReal    = 1;
    }
    else
    {
        retVal = DPC_OBJECTDETECTION_RANGE_BINS_ERR;
        goto exit;
    }
    cfgSave->staticCfg.resetDcRangeSigMeanBuffer = 1;
    cfgSave->staticCfg.rangeFFTtuning.fftOutputDivShift =
                                    staticCfg->rangeFFTtuning.fftOutputDivShift;
    cfgSave->staticCfg.rangeFFTtuning.numLastButterflyStagesToScale =
                                    staticCfg->rangeFFTtuning.numLastButterflyStagesToScale;

    (void)memcpy(&cfgSave->staticCfg.compressionCfg,
            &staticCfg->compressionCfg,
            sizeof(DPU_RangeProcHWA_CompressionCfg));
    (void)memcpy(&cfgSave->staticCfg.rangeProcCfg, 
            &ptrObjDetObj->commonCfg.rangeProcCfg, 
            sizeof(DPU_RangeProcHWADDMA_rangeProcCfg));

    /* radarCube */
    cfgSave->hwRes.radarCube = *radarCube;

    /* static configuration - windows */
    /* Generating 1D window, allocate first */
    winGenLen = DPC_ObjDet_GetRangeWinGenLen(cfgSave);
    cfgSave->staticCfg.windowSize = winGenLen * sizeof(uint32_t);
    windowBuffer = (int32_t *)DPC_ObjDet_MemPoolAlloc(WinBufRamObj, cfgSave->staticCfg.windowSize, (uint8_t)sizeof(uint32_t));
    if (windowBuffer == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__CORE_LOCAL_RAM_RANGE_HWA_WINDOW;
        goto exit;
    }
    cfgSave->staticCfg.window = windowBuffer;
    DPC_ObjDet_GenRangeWindow(cfgSave);

    /* hwres - edma */
    hwRes->edmaHandle = edmaHandle;
    /* We have choosen ISOLATE mode, so we have to fill in dataIn */

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAIN_CH,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SHADOW,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAIN_EVENT_QUE,
                                       &edmaIn->dataIn);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SIG_CH,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SIG_SHADOW,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SIG_EVENT_QUE,
                                       &edmaIn->dataInSignature);



    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_SIG_CH,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_SIG_SHADOW,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_SIG_EVENT_QUE,
                                       &edmaOut->dataOutSignature);

    /* Ping */
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PING_CH,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PING_SHADOW,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PING_EVENT_QUE,
                                       &edmaOut->u.fmt1.dataOutPing);

    /* Pong */
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PONG_CH,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PONG_SHADOW,
                                       DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PONG_EVENT_QUE,
                                       &edmaOut->u.fmt1.dataOutPong);

    {
        {

        uint32_t intrIdx = 0;

        /* Allocate interrupt object */
        cfgSave->hwRes.edmaTransferCompleteIntrObj = &ptrObjDetObj->rangProcIntrObj[intrIdx++];
        }
    }

    dcEstNumSamples = cfgSave->staticCfg.ADCBufData.dataProperty.numAdcSamples;
    if(ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_FRAME_DC_MODE)
    {
        dcEstNumSamples *= staticCfg->numChirpsPerFrame;
    }
    /* DC Est shift and scale */
    retVal = DPU_RangeProcHWA_findDCEstStaticParams(dcEstNumSamples,
                                                    &cfgSave->staticCfg.dcEstShiftScaleCfg.scale, &cfgSave->staticCfg.dcEstShiftScaleCfg.shift);
    if (retVal != 0)
    {
        goto exit;
    }

    interfStatsNumSamples = cfgSave->staticCfg.ADCBufData.dataProperty.numAdcSamples;
    if(ptrObjDetObj->commonCfg.rangeProcCfg.isReal2XEnabled)
    {
        interfStatsNumSamples *= staticCfg->ADCBufData.dataProperty.numRxAntennas;
    }

    /* Interf config */
    retVal = DPU_RangeProcHWA_findIntfStatsStaticParams(interfStatsNumSamples,
                                                        staticCfg->intfStatsdBCfg.intfMitgMagSNRdB,
                                                        &cfgSave->staticCfg.intfStatsMagShiftScaleCfg.scale, &cfgSave->staticCfg.intfStatsMagShiftScaleCfg.shift);
    if (retVal != 0)
    {
        goto exit;
    }
    retVal = DPU_RangeProcHWA_findIntfStatsStaticParams(interfStatsNumSamples,
                                                        staticCfg->intfStatsdBCfg.intfMitgMagDiffSNRdB,
                                                        &cfgSave->staticCfg.intfStatsMagDiffShiftScaleCfg.scale, &cfgSave->staticCfg.intfStatsMagDiffShiftScaleCfg.shift);
    if (retVal != 0)
    {
        goto exit;
    }

    uint32_t dcEstBufSize, intfThresBufSize;
    if((ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_FRAME_DC_MODE)
        && (ptrObjDetObj->commonCfg.numSubFrames > 1U))
    {
        /* In 1st RangeProcChain, only DC estimates of every subframe index's last subframe is stored */
        dcEstBufSize = staticCfg->ADCBufData.dataProperty.numRxAntennas * sizeof(uint32_t);
        if(ptrObjDetObj->commonCfg.rangeProcCfg.isReal2XEnabled)
        {
            dcEstBufSize >>= 1; /* In Real 2X Mode, the DC estimate of other half of channels is stored as imaginary part of complex estimate */
        }
    }
    else if(ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_NTH_CHIRP_ESTIMATES_MODE)
    {
        /* In 2nd RangeProcChain, DC estimates and interference statistics are stored for every subbands last chirp */
        dcEstBufSize = (uint32_t)staticCfg->numBandsTotal * (uint32_t)staticCfg->ADCBufData.dataProperty.numRxAntennas * sizeof(uint32_t);
        intfThresBufSize = dcEstBufSize; /* Interference Thresholds are stored only in 2nd RangeProcChain */
    }

    if(((ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_FRAME_DC_MODE)
        && (ptrObjDetObj->commonCfg.numSubFrames > 1U))
        || (ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_NTH_CHIRP_ESTIMATES_MODE))
    {
        hwRes->dcEstIVal = (uint32_t *)DPC_ObjDet_MemPoolAlloc(&ptrObjDetObj->preProcBufObj, dcEstBufSize, (uint8_t)sizeof(uint32_t));
        if (hwRes->dcEstIVal == NULL)
        {
            retVal = DPC_OBJECTDETECTION_PREPROCBUF_ERR;
            goto exit;
        }
        (void)memset(hwRes->dcEstIVal, 0, dcEstBufSize);
        if(ptrObjDetObj->commonCfg.rangeProcCfg.isReal2XEnabled)
        {
            hwRes->dcEstQVal = (uint32_t *)DPC_ObjDet_MemPoolAlloc(&ptrObjDetObj->preProcBufObj, dcEstBufSize, (uint8_t)sizeof(uint32_t));
            if (hwRes->dcEstQVal == NULL)
            {
                retVal = DPC_OBJECTDETECTION_PREPROCBUF_ERR;
                goto exit;
            }
            (void)memset(hwRes->dcEstQVal, 0, dcEstBufSize);
        }
    }
    else
    {
        hwRes->dcEstIVal = NULL;
        hwRes->dcEstQVal = NULL;
    }

    if(ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain == DPU_RANGEPROCHWA_PREVIOUS_NTH_CHIRP_ESTIMATES_MODE)
    {
        hwRes->intfThresMagVal = (uint32_t *)DPC_ObjDet_MemPoolAlloc(&ptrObjDetObj->preProcBufObj, intfThresBufSize, (uint8_t)sizeof(uint32_t));
        if (hwRes->intfThresMagVal == NULL)
        {
            retVal = DPC_OBJECTDETECTION_PREPROCBUF_ERR;
            goto exit;
        }

        hwRes->intfThresMagDiffVal = (uint32_t *)DPC_ObjDet_MemPoolAlloc(&ptrObjDetObj->preProcBufObj, intfThresBufSize, (uint8_t)sizeof(uint32_t));
        if (hwRes->intfThresMagDiffVal == NULL)
        {
            retVal = DPC_OBJECTDETECTION_PREPROCBUF_ERR;
            goto exit;
        }

        (void)memset(hwRes->intfThresMagVal, 0, intfThresBufSize);
        (void)memset(hwRes->intfThresMagDiffVal, 0, intfThresBufSize);
    }
    
    /* In this case HWA hardware trigger source is equal to HWA param index value*/
    hwaCfg->dataInputMode = DPU_RangeProcHWA_InputMode_ISOLATED;

#ifdef DPC_USE_SYMMETRIC_WINDOW_RANGE_DPU
    hwaCfg->hwaWinSym = HWA_FFT_WINDOW_SYMMETRIC;
#else
    hwaCfg->hwaWinSym = HWA_FFT_WINDOW_NONSYMMETRIC;
#endif
    hwaCfg->hwaWinRamOffset = (uint16_t) *windowOffset;
    if ((hwaCfg->hwaWinRamOffset + winGenLen) > DPC_OBJDET_HWA_MAX_WINDOW_RAM_SIZE_IN_SAMPLES)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM_HWA_WINDOW_RAM;
        goto exit;
    }
    *windowOffset += winGenLen;

    hwaCfg->numParamSet = DPU_RANGEPROCHWADDMA_NUM_HWA_PARAM_SETS + staticCfg->compressionCfg.bfpCompExtraParamSets;
    /* Twice the value of rangeProcChain equals the number of paramsets saved compared to the default mode */
    hwaCfg->numParamSet -= ptrObjDetObj->commonCfg.rangeProcCfg.rangeProcChain * 2U;
    hwaCfg->paramSetStartIdx = DPC_OBJDET_DPU_RANGEPROC_PARAMSET_START_IDX;

    retVal = DPU_RangeProcHWA_config(dpuHandle, cfgSave);
    if (retVal != 0)
    {
        goto exit;
    }

    /* store configuration for use in intra-sub-frame processing and
     * inter-sub-frame switching, although window will need to be regenerated and
     * dc range sig should not be reset. */
    cfgSave->staticCfg.resetDcRangeSigMeanBuffer = 0;

    /* report scratch usage */
    *CoreLocalRamScratchUsage = cfgSave->staticCfg.windowSize;
exit:

    return retVal;
}

/**
 *  @b Description
 *  @n
 *     Configure Doppler DPU.
 *
 *  @param[in]  obj Pointer to sub-frame object
 *  @param[in]  dpuHandle Handle to DPU
 *  @param[in]  staticCfg Pointer to static configuration of the sub-frame
 *  @param[in]  log2NumDopplerBins log2 of numDopplerBins of the static config.
 *  @param[in]  antennaCalibParamsPtr Pointer to antenna calibration parameters
 *  @param[in]  edmaHandle Handle to edma driver to be used for the DPU
 *  @param[in]  radarCubeDecompressedSizeInBytes Size of radar cube if it were
 *              not compressed
 *  @param[in]  radarCube Pointer to DPIF radar cube, which will be the input
 *              to doppler processing
 *  @param[in]  detMatrix Pointer to DPIF detection matrix, which will be the output
 *              of doppler processing
 *  @param[in]  CoreLocalRamObj Pointer to core local RAM object to allocate local memory
 *              for the DPU, only for scratch purposes
 *  @param[in]  L3ramObj Pointer to L3 RAM memory pool object
 *  @param[in]  WinBufRamObj Pointer to core local RAM object to allocate window memory
 *              for the DPU
 *  @param[in]  CoreLocalScratchStartPoolAddr Core Local RAM's scratch start address
 *  @param[out] CoreLocalScratchStartPoolAddrNextDPU Core Local RAM's scratch start address for Next DPU
 *  @param[out] l3RamStartPoolAddrNextDPU L3 RAM's start address for Next DPU
 *  @param[in,out]  windowOffset Window coefficients that are generated by this function
 *                               (in heap memory) are passed to DPU configuration API to
 *                               configure the HWA window RAM starting from this offset.
 *                               The end offset after this configuration will be returned
 *                               in this variable which could be the begin offset for the
 *                               next DPU window RAM.
 *  @param[out]  CoreLocalRamScratchUsage Core Local RAM's scratch usage in bytes
 *  @param[out] cfgSave Configuration that is built in local
 *                      (stack) variable is saved here. This is for facilitating
 *                      quick reconfiguration later without having to go through
 *                      the construction of the configuration.
 *  @param[in]  objDetObj Pointer to DPC object
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjDet_dopplerConfig(SubFrameObj *obj,
                   DPU_DopplerProcHWA_Handle dpuHandle,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
                   uint8_t log2NumDopplerBins,
                   float *                       antennaCalibParamsPtr,
                   EDMA_Handle                   edmaHandle,
                   uint32_t                      radarCubeDecompressedSizeInBytes,
                   DPIF_RadarCube                *radarCube,
                   DPIF_DetMatrix                *detMatrix,
                   MemPoolObj                    *CoreLocalRamObj,
                   MemPoolObj                    *L3ramObj,
                   MemPoolObj                    *WinBufRamObj,
                   void *                        CoreLocalScratchStartPoolAddr,
                   volatile void *               CoreLocalScratchStartPoolAddrNextDPU,
                   volatile void *               l3RamStartPoolAddrNextDPU,
                   uint32_t                      *windowOffset,
                   uint32_t                      *CoreLocalRamScratchUsage,
                   DPU_DopplerProcHWA_Config     *cfgSave,
                   ObjDetObj                     *objDetObj)
{

    int32_t retVal = 0;
    DPU_DopplerProcHWA_Config *dopCfg = cfgSave;
    DPU_DopplerProcHWA_HW_Resources  *hwRes;
    DPU_DopplerProcHWA_StaticConfig  *dopStaticCfg;
    DPU_DopplerProcHWA_EdmaCfg *edmaCfg;
    DPU_DopplerProcHWA_HwaCfg *hwaCfg;
    uint32_t *windowBuffer, winGenLen, winType;
    uint32_t detObjListSizeInBytes, dopMaxSubBandScratchBufferSizeBytes;
    uint32_t objOutSizeInBytes, sideInfoSizeInBytes;
    void * scratchBufMem;
    uint8_t pingPongIdx;

    hwRes = &dopCfg->hwRes;
    dopStaticCfg = &dopCfg->staticCfg;
    edmaCfg = &hwRes->edmaCfg;
    hwaCfg = &hwRes->hwaCfg;

    (void)memset((void*)dopCfg, 0, sizeof(DPU_DopplerProcHWA_Config));

    dopStaticCfg->numTxAntennas         = staticCfg->numTxAntennas;
    dopStaticCfg->numAzimTxAntennas     = staticCfg->numVirtualAntAzim / staticCfg->ADCBufData.dataProperty.numRxAntennas;
    dopStaticCfg->numRxAntennas         = staticCfg->ADCBufData.dataProperty.numRxAntennas;
    dopStaticCfg->numVirtualAntennas    = staticCfg->numVirtualAntennas;
    dopStaticCfg->numRangeBins          = staticCfg->numRangeBins;
    dopStaticCfg->numChirps             = staticCfg->numChirps;
    dopStaticCfg->numDopplerFFTBins     = staticCfg->numDopplerBins;
    dopStaticCfg->numBandsTotal         = staticCfg->numBandsTotal;
    dopStaticCfg->log2NumDopplerBins    = log2NumDopplerBins;
    dopStaticCfg->isSumTxEnabled        = staticCfg->isSumTxEnabled;

#ifdef OBJ_DETECTION_DDMA_TEST
    dopStaticCfg->numAzimFFTBins = 4 * mathUtils_getValidFFTSize(staticCfg->ADCBufData.dataProperty.numRxAntennas *\
                                                staticCfg->numVirtualAntAzim / \
                                                staticCfg->ADCBufData.dataProperty.numRxAntennas);

#else
    /* Setting numAzimFFTBins to 32 for 3 Azim Tx (as opposed to 48) as well as 2 Azim Tx case for optimization
       (instead of using the following formula:) */
    dopStaticCfg->numAzimFFTBins = OBJECTDETECTION_NUM_AZIM_FFT_BINS;
#endif

#ifdef ENABLE_HISTOGRAM_BASED_DOP_AZIM_DETECTION
    /* HWA supports maximum 64 number of histograms, thus, azimfft size should <=64 if this feature is enabled.*/
    if(dopStaticCfg->numAzimFFTBins > 64U)
    {
        retVal = DPC_OBJECTDETECTION_HIST_AZIMFFT_SIZE;
        goto exit;
    }
#endif
    /* Zero Insertion Mask in Azimuth dimension */
    dopStaticCfg->zeroInsrtMaskAzim = objDetObj->commonCfg.zeroInsrtMaskCfg.zeroInsrtMaskAzim;

    /* The compression config structure in rangeproc and decompression config
       structure in dopplerproc are the same. */
    (void)memcpy((void*)&dopStaticCfg->decompCfg, (void*)&staticCfg->compressionCfg, sizeof(DPU_DopplerProc_DecompressionCfg));

#ifndef ENABLE_HISTOGRAM_BASED_DOP_AZIM_DETECTION
    /* Cfar Cfg */
    (void)memcpy((void*)&dopStaticCfg->cfarCfg, (void*)&staticCfg->cfarCfg.cfg, sizeof(DPU_DopplerProc_CfarCfg));
#endif

    /* Local Max cfg */
    (void)memcpy((void*)&dopStaticCfg->localMaxCfg, (void*)&staticCfg->localMaxCfg, sizeof(DPU_DopplerProc_LocalMaxCfg));

    /* Antenna Calib Cfg */
    (void)memcpy((void*)&dopStaticCfg->antennaCalibParams, (void*)antennaCalibParamsPtr, sizeof(dopStaticCfg->antennaCalibParams));

    /* Antenna Geometry Pattern */
    (void)memcpy((void*)&dopStaticCfg->antennaGeometryCfg, (void*)objDetObj->commonCfg.antennaGeometryCfg, sizeof(objDetObj->commonCfg.antennaGeometryCfg));

    edmaCfg->edmaHandle = edmaHandle;

    /* hwaCfg - window */
    winGenLen = DPC_ObjDet_GetDopplerWinGenLen(dopCfg);
    hwaCfg->windowSize = winGenLen * sizeof(int32_t);
    DPC_ObjDet_MemPoolReset(WinBufRamObj);
    windowBuffer = DPC_ObjDet_MemPoolAlloc(WinBufRamObj, hwaCfg->windowSize, (uint8_t)sizeof(uint32_t));
    if (windowBuffer == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__CORE_LOCAL_RAM_DOPPLER_HWA_WINDOW;
        goto exit;
    }
    hwaCfg->window = (int32_t *)windowBuffer;
    hwaCfg->winRamOffset = (uint16_t) *windowOffset;
    winType = DPC_ObjDet_GenDopplerWindow(dopCfg);
    if (winType != DPC_DPU_DOPPLERPROC_FFT_WINDOW_TYPE)
    {
        retVal = DPC_OBJECTDETECTION_WIN_ERR;
        goto exit;
    }

#ifdef DPC_USE_SYMMETRIC_WINDOW_DOPPLER_DPU
    hwaCfg->winSym = HWA_FFT_WINDOW_SYMMETRIC;
#else
    hwaCfg->winSym = HWA_FFT_WINDOW_NONSYMMETRIC;
#endif
    if ((hwaCfg->winRamOffset + winGenLen) > DPC_OBJDET_HWA_MAX_WINDOW_RAM_SIZE_IN_SAMPLES)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM_HWA_WINDOW_RAM;
        goto exit;
    }
    *windowOffset += winGenLen;

    /********************************************
     * Allocating memory resources              *
     *******************************************/
    /* DPU Output Resource */

    /* Max Number of objects that can be stored per range gate. */
    hwRes->maxNumDetObjsPerRGate = (radarCube->dataSize/staticCfg->numRangeBins)/sizeof(DetObjParams);

    /* Following is required to ensure local array SNRList in DPU_DopplerProcHWA_extractObjectList does not overflow.
     * This also allows user to set a lower limit for object detection per range gate, if desired. */
    if (hwRes->maxNumDetObjsPerRGate > MAX_NUM_OBJ_PER_RANGE_BIN)
    {
        hwRes->maxNumDetObjsPerRGate = MAX_NUM_OBJ_PER_RANGE_BIN;
    }

    dopStaticCfg->maxNumObj = (uint32_t)hwRes->maxNumDetObjsPerRGate * (uint32_t)staticCfg->numRangeBins;
    /* Doppler stage output - List of detected objects is stored on the radar cube. */
    hwRes->detObjList = (DetObjParams *)radarCube->data;

    /* Maximum number of objects that can be stored on L2 for angle processing - after intersection with range CFAR. */
    hwRes->finalMaxNumDetObjs = (CoreLocalRamObj->cfg.size)/sizeof(DetObjParams);

    /* If Range CFAR is not enabled, limit the maximum number of objects after doppler stage to finalMaxNumDetObjs*/
    if(obj->staticCfg.rangeCfarCfg.cfg.isEnabled == 0U)
    {
        dopStaticCfg->maxNumObj = hwRes->finalMaxNumDetObjs;
    }

    /* Allocate an intersected shorter list in L2 RAM. */
    detObjListSizeInBytes = sizeof(DetObjParams) * hwRes->finalMaxNumDetObjs;
    DPC_ObjDet_MemPoolSet(CoreLocalRamObj, CoreLocalScratchStartPoolAddr);
    scratchBufMem = DPC_ObjDet_MemPoolAlloc(CoreLocalRamObj, detObjListSizeInBytes, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__OBJ_PARAMS_RAM_DOPPLER_DECOMP_BUF;
        goto exit;
    }
    DPC_ObjDet_MemPoolSet(CoreLocalRamObj, CoreLocalScratchStartPoolAddr);
    hwRes->finalDetObjList = (DetObjParams *)scratchBufMem;

    objOutSizeInBytes = sizeof(DPIF_PointCloudCartesian) * hwRes->finalMaxNumDetObjs;
    scratchBufMem = DPC_ObjDet_MemPoolAlloc(L3ramObj, objOutSizeInBytes, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__OBJ_PARAMS_RAM_DOPPLER_DECOMP_BUF;
        goto exit;
    }
    hwRes->objOut = (DPIF_PointCloudCartesian *)scratchBufMem;

    sideInfoSizeInBytes = sizeof(DPIF_PointCloudSideInfo) * hwRes->finalMaxNumDetObjs;
    scratchBufMem = DPC_ObjDet_MemPoolAlloc(L3ramObj, sideInfoSizeInBytes, (uint8_t)DOUBLEWORD_ALIGNED);
    if (scratchBufMem == NULL){
        retVal = DPC_OBJECTDETECTION_ENOMEM__OBJ_PARAMS_SIDEINFO;
        goto exit;
    }
    obj->detObjOutSideInfo = (DPIF_PointCloudSideInfo *)scratchBufMem;

    l3RamStartPoolAddrNextDPU = DPC_ObjDet_MemPoolGet(L3ramObj);
    if (l3RamStartPoolAddrNextDPU == NULL)
    {
        retVal = DPC_OBJECTDETECTION_EINVAL;
        goto exit;
    }

    /* Resetting to buffer start */
    DPC_ObjDet_MemPoolSet(CoreLocalRamObj, CoreLocalScratchStartPoolAddr);

    /* We don't need any L2 resources to be retained till the end of the next DPU */
    CoreLocalScratchStartPoolAddrNextDPU = DPC_ObjDet_MemPoolGet(CoreLocalRamObj);
    if (CoreLocalScratchStartPoolAddrNextDPU == NULL)
    {
        retVal = DPC_OBJECTDETECTION_EINVAL;
        goto exit;
    }
    /* Resources to be saved through the doppler stage */
{{

    /* This resource needs to be saved through the doppler stage */
    if (staticCfg->compressionCfg.rangeBinsPerBlock < 8U)
    {
         hwRes->decompScratchBufferSizeBytes = radarCubeDecompressedSizeInBytes /
                                ((uint32_t)staticCfg->numRangeBins / 8U);
    }
    else
    {
        hwRes->decompScratchBufferSizeBytes = radarCubeDecompressedSizeInBytes /
                                ((uint32_t)staticCfg->numRangeBins / (uint32_t)staticCfg->compressionCfg.rangeBinsPerBlock);
    }

    scratchBufMem = DPC_ObjDet_MemPoolAlloc(L3ramObj, hwRes->decompScratchBufferSizeBytes, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__CORE_LOCAL_RAM_DOPPLER_DECOMP_BUF;
        goto exit;
    }
    hwRes->decompScratchBuf = (uint8_t *)scratchBufMem;

    /* Allocate memory for Max Doppler Sub Band Buffers */
    dopMaxSubBandScratchBufferSizeBytes = ((uint32_t)staticCfg->numDopplerBins / (uint32_t)staticCfg->numBandsTotal) * (uint32_t)sizeof(uint8_t) * 2U; /* Ping and Pong */
    if(dopMaxSubBandScratchBufferSizeBytes > DOPPLER_MAXDOP_SUBBAND_BUFFER_SIZE) 
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__RAM_DOPPLER_MAXDOP_SUBBAND;
        goto exit;
    }
    for (pingPongIdx = 0; pingPongIdx < 2U; pingPongIdx++)
    {
        hwRes->dopMaxSubBandScratchBuf[pingPongIdx] = (uint8_t *)dopMaxSubBandScratchBuf + pingPongIdx * dopMaxSubBandScratchBufferSizeBytes / 2U;
    }

#ifndef ENABLE_HISTOGRAM_BASED_DOP_AZIM_DETECTION
    hwRes->maxCfarPeaksToDetect = DPC_OBJDET_MAX_NUM_CFAR_PEAKS;
#endif

    /* Assign the detection matrix, radar cube */
    hwRes->detMatrix = *detMatrix;
    hwRes->radarCube = *radarCube;

}}
    /********************************************
     * Allocating hw resources (decomp stage)   *
     *******************************************/
{{
	DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DECOMP_IN_PING,
                                       EDMA_DOPPLERPROC_DECOMP_IN_PING_SHADOW,
                                       0,
                                       &edmaCfg->decompEdmaCfg.edmaIn.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DECOMP_IN_PONG,
                                       EDMA_DOPPLERPROC_DECOMP_IN_PONG_SHADOW,
                                       0,
                                       &edmaCfg->decompEdmaCfg.edmaIn.pingPong[1]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DECOMP_OUT_PING,
                                       EDMA_DOPPLERPROC_DECOMP_OUT_PING_SHADOW,
                                       0,
                                       &edmaCfg->decompEdmaCfg.edmaOut.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DECOMP_OUT_PONG,
                                       EDMA_DOPPLERPROC_DECOMP_OUT_PONG_SHADOW,
                                       0,
                                       &edmaCfg->decompEdmaCfg.edmaOut.pingPong[1]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DECOMP_IN_HOTSIG_PING,
                                       EDMA_DOPPLERPROC_DECOMP_IN_HOTSIG_PING_SHADOW,
                                       0,
                                       &edmaCfg->decompEdmaCfg.edmaInSignature.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DECOMP_IN_HOTSIG_PONG,
                                       EDMA_DOPPLERPROC_DECOMP_IN_HOTSIG_PONG_SHADOW,
                                       0,
                                       &edmaCfg->decompEdmaCfg.edmaInSignature.pingPong[1]);

    hwaCfg->decompStageHwaStateMachineCfg.paramSetStartIdx = DPC_OBJDET_DPU_DOPPLERPROCHWADDMA_PARAMSET_START_IDX + staticCfg->compressionCfg.bfpCompExtraParamSets;
    hwaCfg->decompStageHwaStateMachineCfg.paramSetStartIdx -= objDetObj->commonCfg.rangeProcCfg.rangeProcChain * 2U;
    hwaCfg->decompStageHwaStateMachineCfg.numParamSets = DPU_DOPPLERPOCHWADDMA_DECOMP_NUM_HWA_PARAMSETS + staticCfg->compressionCfg.bfpCompExtraParamSets;
}}


    /********************************************
     * Allocating hw resources (doppler stage)  *
     *******************************************/
    {{

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_PING,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_PING_SHADOW,
                                       0,
                                       &edmaCfg->dopplerEdmaCfg.edmaIn.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_PONG,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_PONG_SHADOW,
                                       0,
                                       &edmaCfg->dopplerEdmaCfg.edmaIn.pingPong[1]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_HOTSIG_PING,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_HOTSIG_PING_SHADOW,
                                       0,
                                       &edmaCfg->dopplerEdmaCfg.edmaInSignature.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_HOTSIG_PONG,
                                       EDMA_DOPPLERPROC_DOPPLER_IN_HOTSIG_PONG_SHADOW,
                                       0,
                                       &edmaCfg->dopplerEdmaCfg.edmaInSignature.pingPong[1]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_MAXSUBBAND_OUT_PING,
                                       EDMA_DOPPLERPROC_MAXSUBBAND_OUT_PING_SHADOW,
                                       0,
                                       &edmaCfg->dopplerEdmaCfg.edmaMaxSubbandOut.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_DOPPLERPROC_MAXSUBBAND_OUT_PONG,
                                       EDMA_DOPPLERPROC_MAXSUBBAND_OUT_PONG_SHADOW,
                                       0,
                                       &edmaCfg->dopplerEdmaCfg.edmaMaxSubbandOut.pingPong[1]);

	if (dopStaticCfg->isSumTxEnabled)
	{
        DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                        EDMA_DOPPLERPROC_SUMTX_OUT_PING,
                                        EDMA_DOPPLERPROC_SUMTX_OUT_PING_SHADOW,
                                        0,
                                        &edmaCfg->dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[0]);
        DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                        EDMA_DOPPLERPROC_SUMTX_OUT_PONG,
                                        EDMA_DOPPLERPROC_SUMTX_OUT_PONG_SHADOW,
                                        0,
                                        &edmaCfg->dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[1]);
    }

	hwaCfg->dopplerStageHwaStateMachineCfg.paramSetStartIdx = hwaCfg->decompStageHwaStateMachineCfg.paramSetStartIdx + hwaCfg->decompStageHwaStateMachineCfg.numParamSets;
	if (dopStaticCfg->isSumTxEnabled)
	{
        hwaCfg->dopplerStageHwaStateMachineCfg.numParamSets = DPU_DOPPLERPOCHWADDMA_DOPPLER_NUM_HWA_PARAMSETS;
    }
	else
	{
        hwaCfg->dopplerStageHwaStateMachineCfg.numParamSets = DPU_DOPPLERPOCHWADDMA_DOPPLER_NUM_HWA_PARAMSETS - DPU_DOPPLERPOCHWADDMA_SUMTX_NUM_HWA_PARAMSETS;
    }

    }}

    /********************************************
     * Allocating hw resources (azim stage)     *
     *******************************************/
    {{
    hwaCfg->azimCfarStageHwaStateMachineCfg.paramSetStartIdx = hwaCfg->dopplerStageHwaStateMachineCfg.paramSetStartIdx + hwaCfg->dopplerStageHwaStateMachineCfg.numParamSets;
    hwaCfg->azimCfarStageHwaStateMachineCfg.numParamSets = DPU_DOPPLERPOCHWADDMA_AZIM_NUM_HWA_PARAMSETS + 2U * (dopStaticCfg->numRxAntennas - MAX_NUM_RX);

    /* Allocate the EDMA channel to copy the antenna samples of detected object. */
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                    EDMA_DOPPLERPROC_EXTRACT_OBJECT_LIST,
                                    EDMA_DOPPLERPROC_EXTRACT_OBJECT_LIST_SHADOW,
                                    0,
                                    &edmaCfg->edmaDetObjAntSamples);
    }}

{
    {

        uint32_t intrIdx = 0;

        /* Allocating interrupt objects */
        edmaCfg->decompEdmaCfg.edmaIntrObjDecompOut                  = &objDetObj->dopplerProcIntrObj[intrIdx++];

        edmaCfg->dopplerEdmaCfg.edmaIntrObjMaxSubbandOut.pingPong[0] = &objDetObj->dopplerProcIntrObj[intrIdx++];
        edmaCfg->dopplerEdmaCfg.edmaIntrObjMaxSubbandOut.pingPong[1] = &objDetObj->dopplerProcIntrObj[intrIdx++];
    }
}
	retVal = DPU_DopplerProcHWA_config(dpuHandle, dopCfg, 1); // 1 ->  full config
    if (retVal != 0)
    {
        goto exit;
    }
    /* report scratch usage */
    *CoreLocalRamScratchUsage = hwaCfg->windowSize;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *     Configure range cfar DPU.
 *
 *  @param[in]  dpuHandle Handle to DPU
 *  @param[in]  staticCfg Pointer to static configuration of the sub-frame
 *  @param[in]  edmaHandle Handle to edma driver to be used for the DPU
 *  @param[in]  detMatrix Pointer to detection matrix, which will be the output
 *  @param[in]  CoreLocalRamObj Pointer to core local RAM object to allocate local memory
 *              for the DPU, only for scratch purposes
 *  @param[in]  L3ramObj Pointer to L3 RAM object to allocate L3 memory for the DPU
 *  @param[out] CoreLocalScratchStartPoolAddrNextDPU Core Local RAM's scratch start address for Next DPU
 *  @param[out] l3RamStartPoolAddrNextDPU L3 RAM's start address for Next DPU
 *  @param[out] cfgSave Configuration that is built in local
 *                      (stack) variable is saved here. This is for facilitating
 *                      quick reconfiguration later without having to go through
 *                      the construction of the configuration.
 *  @param[in]  ptrObjDetObj Pointer to object detection object
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjDet_rangeCfarConfig(DPU_RangeCFARProcHWA_Handle dpuHandle,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
                   EDMA_Handle                   edmaHandle,
                   DPIF_DetMatrix                *detMatrix,
                   MemPoolObj                    *CoreLocalRamObj,
                   MemPoolObj                    *L3ramObj,
                   void                          *CoreLocalScratchStartPoolAddrNextDPU,
                   void                          *l3RamStartPoolAddrNextDPU,
                   DPU_RangeCfarProcHWA_Config   *cfgSave,
                   ObjDetObj                     *ptrObjDetObj)
{
    int32_t retVal = 0;
    void * scratchBufMem;
    DPU_RangeCFARProcHWA_HW_Resources *res = &cfgSave->res;

    (void)memset(cfgSave, 0, sizeof(DPU_RangeCfarProcHWA_Config));

    cfgSave->staticCfg.numDopplerBins     = staticCfg->numChirpsPerFrame;
    cfgSave->staticCfg.numRangeBins       = staticCfg->numRangeBins;
    cfgSave->staticCfg.numSubBandsTotal   = staticCfg->numBandsTotal;
    (void)memcpy(&cfgSave->staticCfg.cfarCfg,
            &staticCfg->rangeCfarCfg.cfg,
            sizeof(DPU_CFARProc_CfarCfg));

    /********************************************
     * Allocating memory resources              *
     *******************************************/
    {
        {

    /* DPU Output Resource */
    res->rangeCfarListSizeBytes = sizeof(RangeCfarListObj) * DPC_OBJDET_RANGECFAR_MAX_NUM_OBJECTS;
    scratchBufMem = DPC_ObjDet_MemPoolAlloc(L3ramObj, res->rangeCfarListSizeBytes, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__OBJ_PARAMS_RAM_RANGE_CFAR_BUF;
        goto exit;
    }
    res->rangeCfarList = (RangeCfarListObj *)scratchBufMem;

    l3RamStartPoolAddrNextDPU = DPC_ObjDet_MemPoolGet(L3ramObj);
    if (l3RamStartPoolAddrNextDPU == NULL)
    {
        retVal = DPC_OBJECTDETECTION_EINVAL;
        goto exit;
    }

    CoreLocalScratchStartPoolAddrNextDPU = DPC_ObjDet_MemPoolGet(CoreLocalRamObj);
    if (CoreLocalScratchStartPoolAddrNextDPU == NULL)
    {
        retVal = DPC_OBJECTDETECTION_EINVAL;
        goto exit;
    }

    res->rangeCfarScratchBufSizeBytes = sizeof(cmplx32ImRe_t) * DPC_OBJDET_RANGECFAR_MAX_NUM_OBJECTS;

    scratchBufMem = DPC_ObjDet_MemPoolAlloc(CoreLocalRamObj, res->rangeCfarScratchBufSizeBytes / 2U, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__CORE_LOCAL_RAM_RANGECFAR_SCRATCH_BUF;
        goto exit;
    }
    res->rangeCfarScratchBuf[0] = (uint8_t *)scratchBufMem;

    scratchBufMem = DPC_ObjDet_MemPoolAlloc(CoreLocalRamObj, res->rangeCfarScratchBufSizeBytes / 2U, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__CORE_LOCAL_RAM_RANGECFAR_SCRATCH_BUF;
        goto exit;
    }
    res->rangeCfarScratchBuf[1] = (uint8_t *)scratchBufMem;

    res->rangeCfarNumObjPerDopplerBinSizeBytes = sizeof(uint16_t) * staticCfg->numChirpsPerFrame / staticCfg->numBandsTotal;

    /* Allocating in L3 as this is required till the doppler and range CFAR intersecton stage.*/
    scratchBufMem = DPC_ObjDet_MemPoolAlloc(L3ramObj, res->rangeCfarNumObjPerDopplerBinSizeBytes, (uint8_t)sizeof(uint32_t));
    if (scratchBufMem == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__CORE_LOCAL_RAM_RANGECFAR_NUMOBJ_PER_DOPPLER_BUF;
        goto exit;
    }
    res->rangeCfarNumObjPerDopplerBinBuf = (uint8_t *)scratchBufMem;

    /* Assign the detection matrix */
    res->detMatrix = *detMatrix;
	    }
	}

    /* hwres - edma */
    res->edmaHandle = edmaHandle;
    res->detMatBytesPerSample = sizeof(uint16_t);
    res->maxNumCFARObj = DPC_OBJDET_RANGECFAR_MAX_NUM_OBJECTS;

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_RANGECFARPROC_CFAR_IN_PING,
                                       EDMA_RANGECFARPROC_CFAR_IN_PING_SHADOW,
                                       DPC_OBJDET_DPU_RANGECFARPROC_EVENT_QUE,
                                       &res->edmaIn.pingPong[0]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_RANGECFARPROC_CFAR_IN_HOTSIG_PING,
                                       EDMA_RANGECFARPROC_CFAR_IN_HOTSIG_PING_SHADOW,
                                       DPC_OBJDET_DPU_RANGECFARPROC_EVENT_QUE,
                                       &res->edmaInSignature.pingPong[0]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_RANGECFARPROC_CFAR_IN_PONG,
                                       EDMA_RANGECFARPROC_CFAR_IN_PONG_SHADOW,
                                       DPC_OBJDET_DPU_RANGECFARPROC_EVENT_QUE,
                                       &res->edmaIn.pingPong[1]);
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_RANGECFARPROC_CFAR_IN_HOTSIG_PONG,
                                       EDMA_RANGECFARPROC_CFAR_IN_HOTSIG_PONG_SHADOW,
                                       DPC_OBJDET_DPU_RANGECFARPROC_EVENT_QUE,
                                       &res->edmaInSignature.pingPong[1]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_RANGECFARPROC_CFAR_OUT_PING,
                                       EDMA_RANGECFARPROC_CFAR_OUT_PING_SHADOW,
                                       DPC_OBJDET_DPU_RANGECFARPROC_EVENT_QUE,
                                       &res->edmaOut.pingPong[0]);

    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle,
                                       EDMA_RANGECFARPROC_CFAR_OUT_PONG,
                                       EDMA_RANGECFARPROC_CFAR_OUT_PONG_SHADOW,
                                       DPC_OBJDET_DPU_RANGECFARPROC_EVENT_QUE,
                                       &res->edmaOut.pingPong[1]);

    {
        {

        uint32_t intrIdx = 0;

        /* Allocate interrupt object */
        cfgSave->res.edmaIntrObj.pingPong[0] = &ptrObjDetObj->rangeCfarProcIntrObj[intrIdx++];
        cfgSave->res.edmaIntrObj.pingPong[1] = &ptrObjDetObj->rangeCfarProcIntrObj[intrIdx++];
        }
    }

    res->hwaCfg.numParamSet = DPU_RANGECFARPROCHWADDMA_NUM_HWA_PARAMSETS;
    res->hwaCfg.paramSetStartIdx = (uint8_t)DPC_OBJDET_DPU_RANGECFARPROCHWADDMA_PARAMSET_START_IDX + 2U * staticCfg->compressionCfg.bfpCompExtraParamSets;

    retVal = DPU_RangeCFARProcHWA_config(dpuHandle, cfgSave);
    if (retVal != 0)
    {
        goto exit;
    }

exit:

    return retVal;
}



/**
 *  @b Description
 *  @n
 *     Performs processing related to pre-start configuration, which is per sub-frame,
 *     by configuring each of the DPUs involved in the processing chain.
 *  Memory management notes:
 *  1. Core Local Memory that needs to be preserved across sub-frames (such as range DPU's calib DC buffer)
 *     will be allocated using MemoryP_alloc.
 *  2. Core Local Memory that needs to be preserved within a sub-frame across DPU calls
 *     (the DPIF * type memory) or for intermediate private scratch memory for
 *     DPU (i.e no preservation is required from process call to process call of the DPUs
 *     within the sub-frame) will be allocated from the Core Local RAM configuration supplied in
 *     @ref DPC_ObjectDetection_InitParams given to @ref DPC_ObjectDetection_init API
 *  3. L3 memory will only be allocated from the L3 RAM configuration supplied in
 *     @ref DPC_ObjectDetection_InitParams given to @ref DPC_ObjectDetection_init API
 *     No L3 buffers are presently required that need to be preserved across sub-frames
 *     (type described in #1 above), neither are L3 scratch buffers required for
 *     intermediate processing within DPU process call.
 *
 *  @param[in]  obj Pointer to sub-frame object
 *  @param[in]  commonCfg Pointer to pre-start common configuration
 *  @param[in]  staticCfg Pointer to static configuration of the sub-frame
 *  @param[in]  edmaHandle Pointer to array of EDMA handles for the device, this
 *              can be distributed among the DPUs, although presently we only
 *              use the first handle for all DPUs.
 *  @param[in]  L3ramObj Pointer to L3 RAM memory pool object
 *  @param[in]  CoreLocalRamObj Pointer to Core Local RAM memory pool object
 *  @param[in]  hwaMemBankAddr pointer to HWA Memory Bank addresses that will be used
 *              to allocate various scratch areas for the DPU processing
 *  @param[in]  hwaMemBankSize Size in bytes of each of HWA memory banks
 *  @param[out] L3RamUsage Net L3 RAM memory usage in bytes as a result of allocation
 *              by the DPUs.
 *  @param[out] CoreLocalRamUsage Net Core Local RAM memory usage in bytes as a
 *              result of allocation by the DPUs.
 *  @param[in]  ptrObjDetObj Pointer to object detection object
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 */
static int32_t DPC_ObjDet_preStartConfig(SubFrameObj *obj,
                  DPC_ObjectDetection_PreStartCommonCfg *commonCfg,
                   DPC_ObjectDetection_StaticCfg *staticCfg,
                   EDMA_Handle                   edmaHandle[EDMA_NUM_CC],
                   MemPoolObj                    *L3ramObj,
                   MemPoolObj                    *CoreLocalRamObj,
                   uint32_t                      *hwaMemBankAddr,
                   uint16_t                      hwaMemBankSize,
                   uint32_t                      *L3RamUsage,
                   uint32_t                      *CoreLocalRamUsage,
                   ObjDetObj                    *ptrObjDetObj)
{
    int32_t retVal = 0;
    DPIF_RadarCube radarCube;
    DPIF_DetMatrix detMatrix;
    uint32_t hwaWindowOffset;
    uint32_t rangeCoreLocalRamScratchUsage,
             dopplerCoreLocalRamScratchUsage;
    void *CoreLocalScratchStartPoolAddr;
    float achievedCompressionRatio;
    uint32_t outputBytesPerBlock, inputBytesPerBlock;
    uint32_t radarCubeDecompressedSizeInBytes;
    void *CoreLocalScratchStartPoolAddrNextDPU = NULL;
    void *l3RamStartPoolAddrNextDPU = NULL;
    float temp;
#ifdef SUBSYS_DSS
    float radConversionFactor;
    radConversionFactor = PI_/180.0f;
#endif

    MemPoolObj WinBufRamObj;
    WinBufRamObj.cfg.addr = &gWinBuf[0];
    WinBufRamObj.cfg.size = sizeof(gWinBuf);
    DPC_ObjDet_MemPoolReset(&WinBufRamObj);

    /* save configs to object. We need to pass this stored config (instead of
       the input arguments to this function which will be in stack) to
       the DPU config functions inside of this function because the DPUs
       have pointers to dynamic configurations which are later going to be
       reused during re-configuration (intra sub-frame or inter sub-frame)
     */
    obj->staticCfg = *staticCfg;

    hwaWindowOffset = DPC_OBJDET_HWA_WINDOW_RAM_OFFSET;

    /* derived config */
    obj->log2NumDopplerBins = mathUtils_floorLog2(staticCfg->numDopplerBins);

    DPC_ObjDet_MemPoolReset(L3ramObj);
    DPC_ObjDet_MemPoolReset(CoreLocalRamObj);

    /* L3 allocations */
    /* L3 - radar cube */
    /* Input and output samples out of the rangeproc/compression DPU */
    if(staticCfg->compressionCfg.compressionMethod==HWA_COMPRESS_METHOD_BFP)
    {
        inputBytesPerBlock = 4U * (uint32_t)staticCfg->compressionCfg.rangeBinsPerBlock;
    }
    else
    {
        inputBytesPerBlock = 4U * (uint32_t)staticCfg->compressionCfg.numRxAntennaPerBlock * (uint32_t)staticCfg->compressionCfg.rangeBinsPerBlock;
    }
    temp = (((staticCfg->compressionCfg.compressionRatio * (float)inputBytesPerBlock) + 3.99F) / 4.0F);
    outputBytesPerBlock = (uint32_t)temp * 4U; /* Word aligned */
    achievedCompressionRatio = (float) outputBytesPerBlock / (float) inputBytesPerBlock;

    radarCubeDecompressedSizeInBytes = (uint32_t)staticCfg->numRangeBins * (uint32_t)staticCfg->numChirps *
                                        (uint32_t)staticCfg->ADCBufData.dataProperty.numRxAntennas * sizeof(cmplx16ReIm_t);
    temp = (float) radarCubeDecompressedSizeInBytes * achievedCompressionRatio;
    radarCube.dataSize = (uint32_t)temp;
    radarCube.data = DPC_ObjDet_MemPoolAlloc(L3ramObj, radarCube.dataSize,
                                            (uint8_t)DPC_OBJDET_RADAR_CUBE_DATABUF_BYTE_ALIGNMENT);

    if (radarCube.data == NULL)
    {
        retVal = DPC_OBJECTDETECTION_ENOMEM__L3_RAM_RADAR_CUBE;
        goto exit;
    }
    radarCube.datafmt = DPIF_RADARCUBE_FORMAT_2;

    if (staticCfg->isSumTxEnabled)
    {
        /* L3 - detection matrix */
        detMatrix.dataSize = (uint32_t)staticCfg->numRangeBins * ((uint32_t)staticCfg->numDopplerBins / (uint32_t)staticCfg->numBandsTotal) * sizeof(uint16_t);
        detMatrix.data = DPC_ObjDet_MemPoolAlloc(L3ramObj, detMatrix.dataSize,
                                                (uint8_t)DPC_OBJDET_DET_MATRIX_DATABUF_BYTE_ALIGNMENT);
        if (detMatrix.data == NULL)
        {
            retVal = DPC_OBJECTDETECTION_ENOMEM__L3_RAM_DET_MATRIX;
            goto exit;
        }
        detMatrix.datafmt = DPIF_DETMATRIX_FORMAT_1;
    }

    /* Remember pool position */
    CoreLocalScratchStartPoolAddr = DPC_ObjDet_MemPoolGet(CoreLocalRamObj);

    retVal = DPC_ObjDet_rangeConfig(obj->dpuRangeObj, &obj->staticCfg,
                 edmaHandle[DPC_OBJDET_DPU_RANGEPROC_EDMA_INST_ID],
                 &radarCube, CoreLocalRamObj, &WinBufRamObj, &hwaWindowOffset,
                 &rangeCoreLocalRamScratchUsage, &obj->dpuCfg.rangeCfg,
                 ptrObjDetObj);
    if (retVal != 0)
    {
        goto exit;
    }

    if (obj->staticCfg.rangeCfarCfg.cfg.isEnabled)
    {
        /* allocating range cfar resources after range DPU resource allocation because of the following reasons:
         * 1. range DPU resources + range cfar resources are mostly less than dppler DPU resources size; so,
         * instead of allocating at the end of doppler dpu, allocating here, will save around 4-5KB of L2RAM.
         */
        retVal = DPC_ObjDet_rangeCfarConfig(obj->dpuRangeCfarObj, &obj->staticCfg,
                                    edmaHandle[DPC_OBJDET_DPU_RANGECFARPROC_EDMA_INST_ID],
                                    &detMatrix,
                                    CoreLocalRamObj,
                                    L3ramObj,
                                    CoreLocalScratchStartPoolAddrNextDPU,
                                    l3RamStartPoolAddrNextDPU,
                                    &obj->dpuCfg.rangeCfarCfg,
                                    ptrObjDetObj);
        if (retVal != 0)
        {
            goto exit;
        }
    }

    /* Rewind to the scratch beginning */
    DPC_ObjDet_MemPoolSet(CoreLocalRamObj, CoreLocalScratchStartPoolAddr);
    CoreLocalScratchStartPoolAddrNextDPU = CoreLocalScratchStartPoolAddr;
    l3RamStartPoolAddrNextDPU = DPC_ObjDet_MemPoolGet(L3ramObj);

    retVal = DPC_ObjDet_dopplerConfig(obj, obj->dpuDopplerObj, &obj->staticCfg,
                 obj->log2NumDopplerBins,
                 commonCfg->antennaCalibParams,
                 edmaHandle[DPC_OBJDET_DPU_DOPPLERPROC_EDMA_INST_ID],
                 radarCubeDecompressedSizeInBytes,
                 &radarCube, &detMatrix, CoreLocalRamObj, L3ramObj, &WinBufRamObj,
                 CoreLocalScratchStartPoolAddr,
                 (volatile void *)CoreLocalScratchStartPoolAddrNextDPU,
                 (volatile void *)l3RamStartPoolAddrNextDPU, &hwaWindowOffset,
                 &dopplerCoreLocalRamScratchUsage, &obj->dpuCfg.dopplerCfg,
                 ptrObjDetObj);
    if (retVal != 0)
    {
        goto exit;
    }

    /* Allocate the EDMA channel to copy the object found in both doppler and range cfar list from L3 to L2. */
    DPC_ObjDet_EDMAChannelConfigAssist(edmaHandle[DPC_OBJDET_DPU_DOPPLERPROC_EDMA_INST_ID],
                                    EDMA_OBJECTDETECTIONDPC_INTERSECT_DETOBJS,
                                    EDMA_OBJECTDETECTIONDPC_INTERSECT_DETOBJS_SHADOW,
                                    0,
                                    &ptrObjDetObj->edmaDetObjs);

    retVal = DPC_ObjectDetection_configEdmaDetObjsOut(edmaHandle[DPC_OBJDET_DPU_DOPPLERPROC_EDMA_INST_ID],
                                                    &obj->dpuCfg.dopplerCfg.hwRes,
                                                    &ptrObjDetObj->edmaDetObjs);
    if(retVal != 0)
    {
        goto exit;
    }

#ifdef SUBSYS_DSS
    /* Sin values of FOV */
    obj->aoaFovSinVal.minAzimuthSinVal   = sinsp(radConversionFactor * obj->staticCfg.aoaFovCfg.minAzimuthDeg);
    obj->aoaFovSinVal.maxAzimuthSinVal   = sinsp(radConversionFactor * obj->staticCfg.aoaFovCfg.maxAzimuthDeg);
    obj->aoaFovSinVal.minElevationSinVal = sinsp(radConversionFactor * obj->staticCfg.aoaFovCfg.minElevationDeg);
    obj->aoaFovSinVal.maxElevationSinVal = sinsp(radConversionFactor * obj->staticCfg.aoaFovCfg.maxElevationDeg);
#endif

    /* Report RAM usage */
    *CoreLocalRamUsage = DPC_ObjDet_MemPoolGetMaxUsage(CoreLocalRamObj);
    *L3RamUsage = DPC_ObjDet_MemPoolGetMaxUsage(L3ramObj);

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      DPC IOCTL commands configuration API which will be invoked by the
 *      application using DPM_ioctl API
 *
 *  @param[in]  handle   DPM's DPC handle
 *  @param[in]  cmd      Capture DPC specific commands
 *  @param[in]  arg      Command specific arguments
 *  @param[in]  argLen   Length of the arguments which is also command specific
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t DPC_ObjectDetection_ioctl(
    DPM_DPCHandle   handle,
    uint32_t            cmd,
    void*               arg,
    uint32_t argLen)
{
    ObjDetObj   *objDetObj;
    SubFrameObj *subFrmObj;
    int32_t      retVal = 0;

    /* Get the DSS MCB: */
    objDetObj = (ObjDetObj *) handle;
    DebugP_assert(objDetObj != NULL);

    /* Process the commands. Process non sub-frame specific ones first
     * so the sub-frame specific ones can share some code. */
    if (cmd == DPC_OBJDET_IOCTL__TRIGGER_FRAME)
    {
        DPC_ObjectDetection_frameStart(handle);
    }
    else if (cmd == DPC_OBJDET_IOCTL__STATIC_PRE_START_COMMON_CFG)
    {
        DPC_ObjectDetection_PreStartCommonCfg *cfg;

        DebugP_assert(argLen == sizeof(DPC_ObjectDetection_PreStartCommonCfg));

        cfg = (DPC_ObjectDetection_PreStartCommonCfg*)arg;

        objDetObj->commonCfg = *cfg;
        objDetObj->isCommonCfgReceived = true;
        
        objDetObj->preProcBufObj.cfg.addr = &preProcBuffer[0];
        objDetObj->preProcBufObj.cfg.size = sizeof(preProcBuffer);
        DPC_ObjDet_MemPoolReset(&objDetObj->preProcBufObj);

#if defined(SOC_AWR2X44P)
        DPC_ObjectDetection_ElevEstCommonCfg elevEstCommonCfg;
        elevEstCommonCfg.numSubFrames = cfg->numSubFrames;
        (void)memcpy(elevEstCommonCfg.antennaCalibParams, cfg->antennaCalibParams, sizeof(elevEstCommonCfg.antennaCalibParams));
        (void)memcpy(elevEstCommonCfg.antennaGeometryCfg, cfg->antennaGeometryCfg, sizeof(elevEstCommonCfg.antennaGeometryCfg));
        elevEstCommonCfg.antennaSpacing = cfg->antennaSpacing;
        elevEstCommonCfg.zeroInsrtMaskCfg = cfg->zeroInsrtMaskCfg;
        elevEstCommonCfg.result = (DPC_ObjectDetection_ExecuteResult *)((uint32_t)SOC_virtToPhy((void*)&gObjDetObj.executeResult));
        
        /* Copy elevEstCommonCfg to message buffer to send it to MSS */
        *(DPC_ObjectDetection_ElevEstCommonCfg *) arg = elevEstCommonCfg; 
#endif

        DebugP_logInfo("ObjDet DPC: Pre-start Common Config IOCTL processed\n");
    }
    else if (cmd == DPC_OBJDET_IOCTL__DYNAMIC_EXECUTE_RESULT_EXPORTED)
    {
#ifdef INCLUDE_DPM
        DPC_ObjectDetection_ExecuteResultExportedInfo *inp;
        volatile uint32_t startTime;

        startTime = CycleCounterP_getCount32();

        DebugP_assert(argLen == sizeof(DPC_ObjectDetection_ExecuteResultExportedInfo));

        inp = (DPC_ObjectDetection_ExecuteResultExportedInfo *)arg;

        /* input sub-frame index must match current sub-frame index */
        DebugP_assert(inp->subFrameIdx == objDetObj->subFrameIndx);

        /* Increment the subframe index to get the next subrame object */
        if (objDetObj->commonCfg.numSubFrames > 1)
        {
            /* Next sub-frame */
            objDetObj->subFrameIndx++;
            if (objDetObj->subFrameIndx == objDetObj->commonCfg.numSubFrames)
            {
                objDetObj->subFrameIndx = 0;
            }
        }
        objDetObj->stats.subFramePreparationCycles =
            CycleCounterP_getCount32() - startTime;

        /* mark end of processing of the frame/sub-frame by the DPC and the app */
        objDetObj->interSubFrameProcToken--;

#ifdef OBJECTDETHWA_PRINT_DPC_TIMING_INFO
        gTimingInfo.resEndTimes[gTimingInfo.resEndCnt % OBJECTDETHWA_NUM_FRAME_TIMING_TO_STORE] = CycleCounterP_getCount32();
        gTimingInfo.resEndCnt++;
#endif
        objDetObj->numTimesResultExported++;

        /* Perform DPM_notifyExecute only if the next frame was received before while
           the processing of the previous frame was not complete, i.e., the frame start
           ISR for the next frame was entered before the result exported ioctl was received
           for the previous frame. Otherwise, perform DPM_notifyExecute in the frame start
           ISR. */
        if (objDetObj->numTimesResultExported == objDetObj->stats.subframeStartIntCounter - 1)
        {
            DebugP_assert(DPM_notifyExecute(objDetObj->dpmHandle, handle) == 0);
        }
        else
        {
            // Do Nothing
        }
#endif
    }
    else
    {
        uint8_t subFrameNum;

        /* First argument is sub-frame number */
        DebugP_assert(arg != NULL);
        subFrameNum = *(uint8_t *)arg;
        subFrmObj = &objDetObj->subFrameObj[subFrameNum];

        switch (cmd)
        {
            /* Related to pre-start configuration */
            case DPC_OBJDET_IOCTL__STATIC_PRE_START_CFG:
            {
                DPC_ObjectDetection_PreStartCfg *cfg;
                DPC_ObjectDetection_DPC_IOCTL_preStartCfg_memUsage *memUsage;
#if defined(SOC_AWR2X44P)
                DPC_ObjectDetection_ElevEstSubframeCfg elevEstSubframeCfg;
                float radConversionFactor;
                radConversionFactor = PI_/180.0f;
#endif

                /* Pre-start common config must be received before pre-start configs
                 * are received. */
                if (objDetObj->isCommonCfgReceived == false)
                {
                    retVal = DPC_OBJECTDETECTION_PRE_START_CONFIG_BEFORE_PRE_START_COMMON_CONFIG;
                    goto exit;
                }

                DebugP_assert(argLen == sizeof(DPC_ObjectDetection_PreStartCfg));

                cfg = (DPC_ObjectDetection_PreStartCfg*)arg;
                memUsage = &cfg->memUsage;
                memUsage->L3RamTotal = objDetObj->L3RamObj.cfg.size;
                memUsage->CoreLocalRamTotal = objDetObj->CoreLocalRamObj.cfg.size;
                retVal = DPC_ObjDet_preStartConfig(subFrmObj,
                             &objDetObj->commonCfg, &cfg->staticCfg,
                             &objDetObj->edmaHandle[0],
                             &objDetObj->L3RamObj,
                             &objDetObj->CoreLocalRamObj,
                             &objDetObj->hwaMemBankAddr[0],
                             objDetObj->hwaMemBankSize,
                             &memUsage->L3RamUsage,
                             &memUsage->CoreLocalRamUsage,
                             objDetObj);
                if (retVal != 0)
                {
                    goto exit;
                }

#if defined(SOC_AWR2X44P)
                /* Populate the configs requirred for AoA estimation */
                elevEstSubframeCfg.numAzimFFTBins = subFrmObj->dpuCfg.dopplerCfg.staticCfg.numAzimFFTBins;
                elevEstSubframeCfg.numDopplerBins = subFrmObj->staticCfg.numDopplerBins;
                elevEstSubframeCfg.rangeStep = subFrmObj->staticCfg.rangeStep;
                elevEstSubframeCfg.dopplerStep = subFrmObj->staticCfg.dopplerStep;

                /* Sin values of FOV */
                elevEstSubframeCfg.aoaFovSinVal.minAzimuthDeg   = (float)sin((double)radConversionFactor * (double)subFrmObj->staticCfg.aoaFovCfg.minAzimuthDeg);
                elevEstSubframeCfg.aoaFovSinVal.maxAzimuthDeg   = (float)sin((double)radConversionFactor * (double)subFrmObj->staticCfg.aoaFovCfg.maxAzimuthDeg);
                elevEstSubframeCfg.aoaFovSinVal.minElevationDeg = (float)sin((double)radConversionFactor * (double)subFrmObj->staticCfg.aoaFovCfg.minElevationDeg);
                elevEstSubframeCfg.aoaFovSinVal.maxElevationDeg = (float)sin((double)radConversionFactor * (double)subFrmObj->staticCfg.aoaFovCfg.maxElevationDeg);

                /* Copy elevEstSubframeCfg to message buffer to send it to MSS */
                *(DPC_ObjectDetection_ElevEstSubframeCfg *)arg = elevEstSubframeCfg;
#endif
                DebugP_logInfo("ObjDet DPC: Pre-start Config IOCTL processed (subFrameIndx = %d)\n", subFrameNum);
                break;
            }

            default:
            {
                /* Error: This is an unsupported command */
                retVal = DPC_OBJECTDETECTION_EINVAL__COMMAND;
                break;
            }
        }
    }

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      DPC's (DPM registered) initialization function which is invoked by the
 *      application using DPM_init API. Among other things, this API allocates DPC instance
 *      and DPU instances (by calling DPU's init APIs) from the MemoryP osal
 *      heap. If this API returns an error of any type, the heap is not guaranteed
 *      to be in the same state as before calling the API (i.e any allocations
 *      from the heap while executing the API are not guaranteed to be deallocated
 *      in case of error), so any error from this API should be considered fatal and
 *      if the error is of _ENOMEM type, the application will
 *      have to be built again with a bigger heap size to address the problem.
 *
 *  @param[in]  dpmHandle   DPM's DPC handle
 *  @param[in]  ptrInitCfg  Handle to the framework semaphore
 *  @param[out] errCode     Error code populated on error
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
DPM_DPCHandle DPC_ObjectDetection_init(
#ifdef INCLUDE_DPM
    DPM_Handle          dpmHandle,
    DPM_InitCfg*        ptrInitCfg,
#else
    DPC_ObjectDetection_InitParams* dpcInitParams,
#endif

    int32_t *errCode)
{
    uint32_t i;
    volatile uint8_t subFrameIdx=0U;
    ObjDetObj     *objDetObj = NULL;
    SubFrameObj   *subFrmObj = NULL;
#ifdef INCLUDE_DPM
    DPC_ObjectDetection_InitParams *dpcInitParams;
#endif
    DPU_RangeProcHWA_InitParams rangeInitParams;
    DPU_DopplerProcHWA_InitParams dopplerInitParams;
    DPU_RangeCFARProcHWA_InitParams rangeCfarInitParams;
    HWA_MemInfo         hwaMemInfo;

    *errCode = 0;

#ifdef INCLUDE_DPM
    if ((ptrInitCfg == NULL) || (ptrInitCfg->arg == NULL))
#else
    if (dpcInitParams == NULL)
#endif
    {
        *errCode = DPC_OBJECTDETECTION_EINVAL;
        goto exit;
    }

#ifdef INCLUDE_DPM
    if (ptrInitCfg->argSize != sizeof(DPC_ObjectDetection_InitParams))
    {
        *errCode = DPC_OBJECTDETECTION_EINVAL__INIT_CFG_ARGSIZE;
        goto exit;
    }

    dpcInitParams = (DPC_ObjectDetection_InitParams *) ptrInitCfg->arg;
#endif

    objDetObj = &gObjDetObj;

    /* Initialize memory */
    (void)memset((void *)objDetObj, 0, sizeof(ObjDetObj));

#ifdef INCLUDE_DPM
    /* Copy over the DPM configuration: */
    (void)memcpy ((void*)&objDetObj->dpmInitCfg, (void*)ptrInitCfg, sizeof(DPM_InitCfg));

    objDetObj->dpmHandle = dpmHandle;
#endif

    objDetObj->hwaHandle = dpcInitParams->hwaHandle;
    objDetObj->L3RamObj.cfg = dpcInitParams->L3ramCfg;
    objDetObj->CoreLocalRamObj.cfg = dpcInitParams->CoreLocalRamCfg;

    for(i = 0; i < (uint32_t)EDMA_NUM_CC; i++)
    {
        objDetObj->edmaHandle[i] = dpcInitParams->edmaHandle[i];
    }

    objDetObj->processCallBackCfg = dpcInitParams->processCallBackCfg;

#ifndef INCLUDE_DPM
    /* Create Semaphore to trigger DPC execution on frame start */
    *errCode = SemaphoreP_constructBinary(&objDetObj->dpcExecSemHandle, 0);
    if (*errCode != 0)
    {
        goto exit;
    }
#endif

    /* Set HWA bank memory address */
    *errCode =  HWA_getHWAMemInfo(dpcInitParams->hwaHandle, &hwaMemInfo);
    if (*errCode != 0)
    {
        goto exit;
    }

    objDetObj->hwaMemBankSize = hwaMemInfo.bankSize;

    for (i = 0; i < hwaMemInfo.numBanks; i++)
    {
        objDetObj->hwaMemBankAddr[i] = hwaMemInfo.baseAddress +
            i * hwaMemInfo.bankSize;
    }

    rangeInitParams.hwaHandle = dpcInitParams->hwaHandle;
    dopplerInitParams.hwaHandle = dpcInitParams->hwaHandle;
    rangeCfarInitParams.hwaHandle = dpcInitParams->hwaHandle;

    for (subFrameIdx = 0; subFrameIdx < RL_MAX_SUBFRAMES; subFrameIdx++)
    {
        subFrmObj = &objDetObj->subFrameObj[subFrameIdx];
        subFrmObj->dpuRangeObj = DPU_RangeProcHWA_init(&rangeInitParams, subFrameIdx, errCode);
        if (*errCode != 0)
        {
            goto exit;
        }

        subFrmObj->dpuDopplerObj = DPU_DopplerProcHWA_init(&dopplerInitParams, subFrameIdx, errCode);
        if (*errCode != 0)
        {
            goto exit;
        }

        subFrmObj->dpuRangeCfarObj = DPU_RangeCFARProcHWA_init(&rangeCfarInitParams, subFrameIdx, errCode);
        if (*errCode != 0)
        {
            goto exit;
        }
    }

exit:

    if(*errCode != 0)
    {
        if(objDetObj != NULL)
        {

            objDetObj = NULL;
        }
    }

    return ((DPM_DPCHandle)objDetObj);
}

/**
 *  @b Description
 *  @n
 *      DPC's (DPM registered) de-initialization function which is invoked by the
 *      application using DPM_deinit API.
 *
 *  @param[in]  handle  DPM's DPC handle
 *
 *  \ingroup DPC_OBJDET__INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t DPC_ObjectDetection_deinit (DPM_DPCHandle handle)
{
    ObjDetObj *objDetObj = (ObjDetObj *) handle;
    SubFrameObj   *subFrmObj;
    int32_t retVal = 0;
    uint32_t i;

    if (handle == NULL)
    {
        retVal = DPC_OBJECTDETECTION_EINVAL;
        goto exit;
    }

    ObjectDetection_freeDmaChannels(objDetObj->edmaHandle[0]);

    for(i = 0; i < RL_MAX_SUBFRAMES; i++)
    {
        subFrmObj = &objDetObj->subFrameObj[i];

        retVal = DPU_RangeProcHWA_deinit(subFrmObj->dpuRangeObj);

        if (retVal != 0)
        {
            goto exit;
        }

        retVal = DPU_DopplerProcHWA_deinit(subFrmObj->dpuDopplerObj);

        if (retVal != 0)
        {
            goto exit;
        }

        retVal = DPU_RangeCFARProcHWA_deinit(subFrmObj->dpuRangeCfarObj);

        if (retVal != 0)
        {
            goto exit;
        }

    }

exit:

    return (retVal);
}
