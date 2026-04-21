/*
 *   @file  cli_mmwave.c
 *
 *   @brief
 *      CLI Extension which handles the mmWave configuration
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

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include <drivers/uart.h>

/* mmWave SDK Include Files: */
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/common/syscommon.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/common/mmwavesdk_version.h>
#include "mmwavelink.h"
#include "mmwave.h"
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/utils/cli/cli.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/utils/cli/include/cli_internal.h>

/**************************************************************************
 ******************* CLI mmWave Extension Local Definitions****************
 **************************************************************************/

/*This is used in several formulas that translate CLI input to mmwavelink units.
  It must be defined as double to achieve the correct precision
  on the formulas (if defined as float there will be small precision errors
  that may result in the computed value being out of mmwavelink range if the
  CLI input is a borderline value).

  The vaiable is initialized in @ref CLI_MMWaveExtensionInit() */
double gCLI_mmwave_freq_scale_factor;


/**************************************************************************
 ******************* CLI mmWave Extension Local Functions *****************
 **************************************************************************/

/* CLI Command Functions */
static int32_t CLI_MMWaveVersion (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveFlushCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveDataOutputMode (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveChannelCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveADCCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveProfileCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveChirpCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveAdvChirpCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveLUTDataCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveFrameCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveAdvFrameCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveSubFrameCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveAdvFrameDataCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveLowPowerCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveContModeCfg (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveBPMCfgAdvanced (int32_t argc, char* argv[]);
static int32_t CLI_MMWaveSetLdoBypassConfig(int32_t argc, char* argv[]);
static int32_t CLI_MMWaveApllSynthBwConfig(int32_t argc, char* argv[]);
#ifdef CLI_ADDITIONAL_COMMANDS
static int32_t CLI_setAsyncEventDir(int32_t argc, char* argv[]);
static int32_t CLI_setMiscConfig(int32_t argc, char* argv[]);
static int32_t CLI_gpadcMeasConfig(int32_t argc, char* argv[]);
static int32_t CLI_powerOff(int32_t argc, char* argv[]);
static int32_t CLI_progFiltConfig(int32_t argc, char* argv[]);
#endif

#define CLI_MMWAVE_HELP_SUPPORT  1
#undef CLI_ADDITIONAL_COMMANDS

/* OSC_CLK_ETH configuration Macros */
#define CLI_OSCCLKOUT_DIS_SHIFT       (0x00000005U)
#if (defined(SOC_AWR2544) || defined(SOC_AWR2X44P))
#define CLI_OSCCLKOUTETH_EN_SHIFT     (0x00000007U)
#define CLI_OSCCLKDIV_SHIFT           (0x00000008U)
#define CLI_OSCLKOUTETH_DRV_VAL_SHIFT (0x00000009U)
#endif

/**************************************************************************
 ************************ CLI mmWave Extension Globals ********************
 **************************************************************************/

/**
 * @brief
 *  This is the mmWave extension table added to the CLI.
 */
CLI_CmdTableEntry gCLIMMWaveExtensionTable[] =
{
    {
        "version",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "No arguments",
#else
        NULL,
#endif
        CLI_MMWaveVersion
    },
    {
        "flushCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "No arguments",
#else
        NULL,
#endif
        CLI_MMWaveFlushCfg
    },
    {
        "dfeDataOutputMode",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<modeType>   1-Chirp and 2-Continuous",
#else
        NULL,
#endif
        CLI_MMWaveDataOutputMode
    },
    {
        "channelCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
    #if defined(SOC_AWR2544)
            "<rxChannelEn> <txChannelEn> <cascading> <ethOscClkEn> <driveStrength>",
    #else
            "<rxChannelEn> <txChannelEn> <cascading>",
    #endif
#else
        NULL,
#endif
        CLI_MMWaveChannelCfg
    },
    {
        "adcCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<numADCBits> <adcOutputFmt>",
#else
        NULL,
#endif
        CLI_MMWaveADCCfg
    },
    {
        "profileCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<profileId> <startFreq> <idleTime> <adcStartTime> <rampEndTime> <txOutPower> <txPhaseShifter> <freqSlopeConst> <txStartTime> <numAdcSamples> <digOutSampleRate> <hpfCornerFreq1> <hpfCornerFreq2> <rxGain>",
#else
        NULL,
#endif
        CLI_MMWaveProfileCfg
    },
    {
        "chirpCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<startIdx> <endIdx> <profileId> <startFreqVar> <freqSlopeVar> <idleTimeVar> <adcStartTimeVar> <txEnable>",
#else
        NULL,
#endif
        CLI_MMWaveChirpCfg
    },
    {
        "advChirpCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<chirpParamIdx> <resetMode> <deltaResetPeriod> <deltaParamUpdatePeriod> <sf0ChirpParamDelta> <sf1ChirpParamDelta> \
        <sf2ChirpParamDelta> <sf3ChirpParamDelta> <lutResetPeriod> <lutParamUpdatePeriod> <lutPatternAddressOffset> <numOfPatterns> \
        <lutSfIndexOffset> <lutChirpParamSize> <lutChirpParamScale> <maxTxPhShiftIntDither>",
#else
        NULL,
#endif
        CLI_MMWaveAdvChirpCfg
    },
    {
        "LUTDataCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<chirpParamIdx> <LUTData_{0}>.....<LUTData_{numOfPatterns-1}>",
#else
        NULL,
#endif
        CLI_MMWaveLUTDataCfg
    },
    {
        "frameCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<chirpStartIdx> <chirpEndIdx> <numLoops> <numFrames><numAdcSamples><framePeriodicity> <triggerSelect> <frameTriggerDelay>",
#else
        NULL,
#endif
        CLI_MMWaveFrameCfg
    },
    {
        "advFrameCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<numOfSubFrames> <forceProfile> <numFrames> <triggerSelect> <frameTrigDelay> <numOfSubFrames> ",
#else
        NULL,
#endif
        CLI_MMWaveAdvFrameCfg
    },
    {
        "subFrameCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<subFrameNum> <forceProfileIdx> <chirpStartIdx> <numOfChirps> <numLoops> <burstPeriodicity> <chirpStartIdxOffset> <numOfBurst> <numOfBurstLoops> <subFramePeriodicity>",
#else
        NULL,
#endif
        CLI_MMWaveSubFrameCfg
    },
    {
        "subDataFrameCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<subFrameNum> <numAdcSamples> <totalChirps> <numChirpsInDataPacket> ",
#else
        NULL,
#endif
        CLI_MMWaveAdvFrameDataCfg
    },
    {
        "lowPower",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<reserved> <lpAdcMode>",
#else
        NULL,
#endif
        CLI_MMWaveLowPowerCfg
    },
    {
        "contModeCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<startFreq> <txOutPower> <txPhaseShifter> <digOutSampleRate> <hpfCornerFreq1> <hpfCornerFreq2> <rxGain> <reserved> <numSamples>",
#else
        NULL,
#endif
        CLI_MMWaveContModeCfg,
    },
    {
        "bpmCfgAdvanced",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<chirpStartIdx> <chirpEndIdx> <constBpmVal>",
#else
        NULL,
#endif
        CLI_MMWaveBPMCfgAdvanced
    },
    {
      "LdoBypassConfig",
#ifdef CLI_MMWAVE_HELP_SUPPORT
      "<deviceMap> <ldoBypassEnable> <supplyMonIrDrop> <ioSupplyIndicator>",
#else
        NULL,
#endif
      CLI_MMWaveSetLdoBypassConfig,
    },
    {
      "ApllSynthBwConfig",
#ifdef CLI_MMWAVE_HELP_SUPPORT
      "<devicemap> <synthIcpTrimVco1> <synthRzTrimVco1> <apllIcpTrim > <apllRzTrimLpf> <synthIcpTrimVco2> <apllRzTrimVco> <synthRzTrimVco2>",
#else
        NULL,
#endif
      CLI_MMWaveApllSynthBwConfig,
    },
#if CLI_ADDITIONAL_COMMANDS
    {
      "setAsyncEventDir",
#ifdef CLI_MMWAVE_HELP_SUPPORT
       "<devicemap> <aeDirection> <aeControl> <bssAnaControl> <bssDigCtrl> <aeCrcConfig>",
#else
        NULL,
#endif
        CLI_setAsyncEventDir,
    },
    {
      "setMiscConfig",
#ifdef CLI_MMWAVE_HELP_SUPPORT
       "<devicemap> <miscCtl> ",
#else
        NULL,
#endif
        CLI_setMiscConfig,
    },
    {
      "gpAdcMeasConfig",
#ifdef CLI_MMWAVE_HELP_SUPPORT
       "<devicemap> <enable> <bufferEnable> <sampleCnt> ",
#else
        NULL,
#endif
        CLI_gpadcMeasConfig,
    },
    {
      "powerOff",
      NULL,
      CLI_powerOff,
    },
    {
      "progFilterConfig",
#ifdef CLI_MMWAVE_HELP_SUPPORT
      "<deviceMap> <profileId> <coeffStartIdx> <progFiltLen> <progFiltFreqShift>",
#else
    NULL,
#endif
      CLI_progFiltConfig,
    },
#endif //    CLI_ADDITIONAL_COMMANDS
    {
        NULL,
        NULL,
        NULL
    }
};

/**
 * @brief
 *  Global MMWave configuration tracked by the module.
 */
MMWave_CtrlCfg      gCLIMMWaveControlCfg;

/**
 * @brief
 *  Global MMWave open configuration tracked by the module.
 */
MMWave_OpenCfg      gCLIMMWaveOpenCfg;

/**************************************************************************
 ********************** CLI mmWave Extension Functions ********************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the version command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveVersion (int32_t argc, char* argv[])
{
    rlVersion_t    verArgs;
    rlReturnVal_t retVal;
    uint8_t        rfPatchBuildVer, rfPatchDebugVer;
    rlRfDieIdCfg_t  dieId = { 0 };

    if(gCLI.cfg.overridePlatform == false)
    {
        /* print the platform */
#if defined(SOC_AWR294X)
        CLI_write ("Platform                : AWR294X\r\n");
#elif defined(SOC_AWR2544)
        CLI_write ("Platform                : AWR2544\r\n");
#elif defined(SOC_AWR2X44P)
        CLI_write ("Platform                : AWR2X44P\r\n");
#else
        CLI_write ("Platform                : unknown\r\n");
#endif
    }
    else
    {
        CLI_write ("Platform                : %s\r\n", gCLI.cfg.overridePlatformString);
    }
    memset(&verArgs,0,sizeof(rlVersion_t));
    /* Get the version string: */
    retVal = rlDeviceGetVersion(RL_DEVICE_MAP_CASCADED_1, &verArgs);
    if (retVal < 0)
    {
        CLI_write ("Error: Unable to get the device version from mmWave link [Error %d]\r\n", retVal);
        return -1;
    }

    CLI_write ("RF F/W Version          : %02d.%02d.%02d.%02d.%02d.%02d.%02d\r\n",
                verArgs.rf.fwMajor, verArgs.rf.fwMinor, verArgs.rf.fwBuild, verArgs.rf.fwDebug,
                verArgs.rf.fwYear, verArgs.rf.fwMonth, verArgs.rf.fwDay);

    rfPatchDebugVer = ((verArgs.rf.patchBuildDebug) & 0x0F);
    rfPatchBuildVer = (((verArgs.rf.patchBuildDebug) & 0xF0) >> 4);

    CLI_write ("RF F/W Patch            : %02d.%02d.%02d.%02d.%02d.%02d.%02d\r\n",
                verArgs.rf.patchMajor, verArgs.rf.patchMinor, rfPatchBuildVer, rfPatchDebugVer,
                verArgs.rf.patchYear, verArgs.rf.patchMonth, verArgs.rf.patchDay);
    CLI_write ("mmWaveLink Version      : %02d.%02d.%02d.%02d\r\n",
                verArgs.mmWaveLink.major, verArgs.mmWaveLink.minor,
                verArgs.mmWaveLink.build, verArgs.mmWaveLink.debug);

    /* Get the die ID: */
    retVal = rlGetRfDieId(RL_DEVICE_MAP_CASCADED_1, &dieId);
    if (retVal < 0)
    {
        CLI_write ("Error: Unable to get the device die ID from mmWave link [Error %d]\r\n", retVal);
        return 0;
    }

    CLI_write ("Die ID Val0 : %d\r\n", dieId.dieIDHexVal0);
    CLI_write ("Die ID Val1 : %d\r\n", dieId.dieIDHexVal1);
    CLI_write ("Die ID Val2 = %d, Die ID Val3 = %d\r\n",
    dieId.dieIDHexVal2, dieId.dieIDHexVal3);


#ifndef SOC_AWR2544
    if(gCLI.cfg.procChain == 1){
        CLI_write ("ProcChain               : DDM\r\n");
    }
    else{
        CLI_write ("ProcChain               : TDM\r\n");
    }
#endif
    /* Display the version information on the CLI Console: */
    CLI_write ("mmWave SDK Version      : %02d.%02d.%02d.%02d\r\n",
                            MMWAVE_SDK_VERSION_MAJOR,
                            MMWAVE_SDK_VERSION_MINOR,
                            MMWAVE_SDK_VERSION_BUGFIX,
                            MMWAVE_SDK_VERSION_BUILD);
    /* Version string has been formatted successfully. */
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the flush configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveFlushCfg (int32_t argc, char* argv[])
{
    int32_t errCode;

    /* Flush the configuration in the MMWave */
    if (MMWave_flushCfg (gCLI.cfg.mmWaveHandle, &errCode) < 0)
    {
        /* Error: Flushing the configuration failed. Return the error code back to the callee */
        return errCode;
    }

    /* Reset the global configuration: */
    memset ((void*)&gCLIMMWaveControlCfg, 0, sizeof(MMWave_CtrlCfg));

    /* Reset the open configuration: */
    memset ((void*)&gCLIMMWaveOpenCfg, 0, sizeof(MMWave_OpenCfg));
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the DFE Data Output mode.
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveDataOutputMode (int32_t argc, char* argv[])
{
    uint32_t cfgMode;

    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Get the configuration mode: */
    cfgMode = atoi (argv[1]);
    switch (cfgMode)
    {
        case 1U:
        {
            gCLIMMWaveControlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_FRAME;
            break;
        }
        case 2U:
        {
            gCLIMMWaveControlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_CONTINUOUS;
            break;
        }
        case 3U:
        {
            gCLIMMWaveControlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_ADVANCED_FRAME;
            break;
        }
        case 4U:
        {
            gCLIMMWaveControlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME;
            break;
        }
        case 5U:
        {
            gCLIMMWaveControlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME;
            break;
        }
        default:
        {
            /* Error: Invalid argument. */
            CLI_write ("Error: Invalid mode.\n");
            return -1;
        }
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the channel configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveChannelCfg (int32_t argc, char* argv[])
{
    rlChanCfg_t     chCfg;

    /* Sanity Check: Minimum argument check */
#if (defined(SOC_AWR2544) || defined (SOC_AWR2X44P))
    if (argc != 6)
    {
        CLI_write ("Error: Invalid usage of the CLI command\r\n");
        return -1;
    }

    if (((uint32_t)atoi(argv[5])) > 15U)
    {
        CLI_write ("Error: Invalid argument value for OSCLKOUTETH_DRV_VAL\r\n");
        return -1;
    }

    if (((uint32_t)atoi(argv[4])) > 1U)
    {
        CLI_write ("Error: Invalid argument value for OSCCLKOUTETH_EN\r\n");
        return -1;
    }
#else
if (argc != 4)
    {
        CLI_write ("Error: Invalid usage of the CLI command\r\n");
        return -1;
    }
#endif

    /* Initialize the channel configuration: */
    memset ((void *)&chCfg, 0, sizeof(rlChanCfg_t));

    /* Populate the channel configuration: */
    chCfg.rxChannelEn = atoi (argv[1]);
    chCfg.txChannelEn = atoi (argv[2]);
    chCfg.cascading   = atoi (argv[3]);
    chCfg.cascadingPinoutCfg   = (1U << CLI_OSCCLKOUT_DIS_SHIFT); /* Disbale OSC_CLKOUT */

#if defined(SOC_AWR2544) || defined (SOC_AWR2X44P)
    chCfg.cascadingPinoutCfg = (chCfg.cascadingPinoutCfg |
                                (((uint32_t)atoi(argv[4])) << CLI_OSCCLKOUTETH_EN_SHIFT) |
                                (((uint32_t)atoi(argv[5])) << CLI_OSCLKOUTETH_DRV_VAL_SHIFT) |
                                (1U << CLI_OSCCLKDIV_SHIFT));
#endif

    /* Save Configuration to use later */
    memcpy((void *)&gCLIMMWaveOpenCfg.chCfg, (void *)&chCfg, sizeof(rlChanCfg_t));
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the ADC configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveADCCfg (int32_t argc, char* argv[])
{
    rlAdcOutCfg_t   adcOutCfg;
    int32_t         retVal = 0;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize the ADC Output configuration: */
    memset ((void *)&adcOutCfg, 0, sizeof(rlAdcOutCfg_t));

    /* Populate the ADC Output configuration: */
    adcOutCfg.fmt.b2AdcBits   = atoi (argv[1]);
    adcOutCfg.fmt.b2AdcOutFmt = atoi (argv[2]);

    /* Save Configuration to use later */
    memcpy((void *)&gCLIMMWaveOpenCfg.adcOutCfg, (void *)&adcOutCfg, sizeof(rlAdcOutCfg_t));
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the profile configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveProfileCfg (int32_t argc, char* argv[])
{
    rlProfileCfg_t          profileCfg;
    uint8_t                 index;
    int32_t                 errCode;
    MMWave_ProfileHandle    profileHandle;
    MMWave_ProfileHandle*   ptrBaseCfgProfileHandle;

    /* Sanity Check: Minimum argument check */
    if (argc != 15)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Profile configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Frame or Advanced Frame \n");
        return -1;
    }

    if ((gCLIMMWaveControlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_FRAME)||
        (gCLIMMWaveControlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME))
    {
        ptrBaseCfgProfileHandle = &gCLIMMWaveControlCfg.u.frameCfg[0].profileHandle[0U];
    }
    else
    {
        ptrBaseCfgProfileHandle = &gCLIMMWaveControlCfg.u.advancedFrameCfg[0].profileHandle[0U];
    }

    /* Initialize the profile configuration: */
    memset ((void *)&profileCfg, 0, sizeof(rlProfileCfg_t));

    /* Populate the profile configuration: */
    profileCfg.profileId             = atoi (argv[1]);

    /* Translate from GHz to [1 LSB = gCLI_mmwave_freq_scale_factor * 1e9 / 2^26 Hz] units
     * of mmwavelink format */
    profileCfg.startFreqConst        =  (uint32_t) (atof(argv[2]) * (1U << 26) /
                                            gCLI_mmwave_freq_scale_factor);

    /* Translate below times from us to [1 LSB = 10 ns] units of mmwavelink format */
    profileCfg.idleTimeConst         = (uint32_t)((float)atof(argv[3]) * 1000 / 10);
    profileCfg.adcStartTimeConst     = (uint32_t)((float)atof(argv[4]) * 1000 / 10);
    profileCfg.rampEndTime           = (uint32_t)((float)atof(argv[5]) * 1000 / 10);

    profileCfg.txOutPowerBackoffCode = atoi (argv[6]);
    profileCfg.txPhaseShifter        = atoi (argv[7]);

    /* Translate from MHz/us to [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e6 * 900) / 2^26 KHz/uS]
     * units of mmwavelink format */
    profileCfg.freqSlopeConst        = (int16_t)(atof(argv[8]) * (1U << 26) /
                                              ((gCLI_mmwave_freq_scale_factor * 1e3) * 900.0));

    /* Translate from us to [1 LSB = 10 ns] units of mmwavelink format */
    profileCfg.txStartTime           = (int32_t)((float)atof(argv[9]) * 1000 / 10);

    profileCfg.numAdcSamples         = atoi (argv[10]);
    profileCfg.digOutSampleRate      = atoi (argv[11]);
    profileCfg.hpfCornerFreq1        = atoi (argv[12]);
    profileCfg.hpfCornerFreq2        = atoi (argv[13]);
    profileCfg.rxGain                = atoi (argv[14]);

    /* Search for a free space in the mmWave configuration block: */
    for (index = 0U; index < MMWAVE_MAX_PROFILE; index++)
    {
        /* Did we get a free entry? */
        if (ptrBaseCfgProfileHandle[index] == NULL)
        {
            /* YES: We can add the profile. */
            break;
        }
    }
    if (index == MMWAVE_MAX_PROFILE)
    {
        /* Error: All the profiles have been exhausted */
        return -1;
    }

    /* Add the profile to the mmWave module: */
    profileHandle = MMWave_addProfile (gCLI.cfg.mmWaveHandle, &profileCfg, &errCode);
    if (profileHandle == NULL)
    {
        /* Error: Unable to add the profile. Return the error code back */
        return errCode;
    }

    /* Record the profile: */
    ptrBaseCfgProfileHandle[index] = profileHandle;
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the chirp configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveChirpCfg (int32_t argc, char* argv[])
{
    rlChirpCfg_t            chirpCfg;
    MMWave_ProfileHandle    profileHandle;
    int32_t                 errCode;

    /* Sanity Check: Minimum argument check */
    if (argc != 9)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Chirp configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Chirp\n");
        return -1;
    }

    /* Initialize the chirp configuration: */
    memset ((void *)&chirpCfg, 0, sizeof(rlChirpCfg_t));

    /* Populate the chirp configuration: */
    chirpCfg.chirpStartIdx   = atoi (argv[1]);
    chirpCfg.chirpEndIdx     = atoi (argv[2]);
    chirpCfg.profileId       = atoi (argv[3]);

    /* Translate from Hz to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e9) / 2^26 Hz]
     * units of mmwavelink format */
    chirpCfg.startFreqVar    = (uint32_t) ((float)atof(argv[4]) * (1U << 26) /
                                            (gCLI_mmwave_freq_scale_factor * 1e9));

    /* Translate from KHz/us to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e6) * 900 /2^26 KHz/us]
     * units of mmwavelink format */
    chirpCfg.freqSlopeVar    = (uint16_t) ((float)atof(argv[5]) * (1U << 26) /
                                           ((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));

    /* Translate from us to [1 LSB = 10ns] units of mmwavelink format */
    chirpCfg.idleTimeVar     = (uint32_t)((float)atof (argv[6]) * 1000.0 / 10.0);

    /* Translate from us to [1 LSB = 10ns] units of mmwavelink format */
    chirpCfg.adcStartTimeVar = (uint32_t)((float)atof (argv[7]) * 1000.0 / 10.0);

    chirpCfg.txEnable        = atoi (argv[8]);

    /* Get the profile handle to which the chirp is to be added: */
    if (MMWave_getProfileHandle (gCLI.cfg.mmWaveHandle, chirpCfg.profileId,
                                 &profileHandle, &errCode) < 0)
    {
        /* Error: Unable to get the profile handle. Return the error code */
        return errCode;
    }

    /* Add the chirp to the profile */
    if (MMWave_addChirp (profileHandle, &chirpCfg, &errCode) == NULL)
    {
        /* Error: Unable to add the chirp. Return the error code. */
        return errCode;
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the advance chirp configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveAdvChirpCfg (int32_t argc, char* argv[])
{
    rlAdvChirpCfg_t     advChirpCfg;
    float               tempVal;
    int32_t             errCode=0;

    /* Sanity Check: Minimum argument check */
    if (argc != 17)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Advanced chirp configuration is valid only for the Advance chirp Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Advanced Chirp (4 or 5)\n");
        return -1;
    }

    /* Reset the advance chirp configuration params and LUT data for each parameter  */
    memset ((void*)&advChirpCfg, 0, sizeof(rlAdvChirpCfg_t));

    advChirpCfg.chirpParamIdx           = (rlUInt8_t) (atoi (argv[1]));
    advChirpCfg.resetMode               = (rlUInt8_t) (atoi (argv[2]));
    advChirpCfg.lutResetPeriod          = (rlUInt16_t) (atoi(argv[9]));
    advChirpCfg.lutParamUpdatePeriod    = (rlUInt16_t) (atoi(argv[10]));

    advChirpCfg.lutPatternAddressOffset = (rlUInt16_t) (atoi(argv[11]));
    /* LUT pattern address offset should be multiple of 4 */
    if((advChirpCfg.lutPatternAddressOffset % 4U) != 0U)
    {
        CLI_write ("Error: LUT pattern address offset should be multiple of 4\n");
        return -1;
    }

    advChirpCfg.numOfPatterns           = (rlUInt16_t) (atoi(argv[12]));
    /* atleast one pattern should be programmed */
    if(advChirpCfg.numOfPatterns==0U){
        CLI_write("Error: Minimum one LUT pattern required\n");
        return -1;
    }
    if(advChirpCfg.numOfPatterns>MMWAVE_EACH_CHIRP_PARAM_MAX_LUT_BUF_SIZE){
        CLI_write("Error: Maximum %d LUT patterns can be configured\n", MMWAVE_EACH_CHIRP_PARAM_MAX_LUT_BUF_SIZE);
    }

    advChirpCfg.lutSfIndexOffset        = (rlUInt16_t) (atoi(argv[13]));

    switch (advChirpCfg.chirpParamIdx)
    {
        case RL_LUT_CHIRP_PROFILE_VAR:
        case RL_LUT_CHIRP_TX_EN_VAR:
        case RL_LUT_CHIRP_BPM_VAL_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Profile_Var / CHIRP_TX_EN / CHIRP_BPM_VAL
             **************************************************************************/

            /* delta dither is not supported for profile/ TX enable/ BPM enable parameter */
            break;
        }
        case RL_LUT_CHIRP_FREQ_START_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Freq_Start_Var
             **************************************************************************/
            advChirpCfg.deltaResetPeriod        = (rlUInt16_t) (atoi(argv[3]));
            advChirpCfg.deltaParamUpdatePeriod  = (rlUInt16_t) (atoi(argv[4]));

            /* Translate from Hz to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e9) / 2^26 Hz]
                units of mmwavelink format */
            tempVal= ((float)atof(argv[5]) * (1U << 26) /(gCLI_mmwave_freq_scale_factor * 1e9));
            advChirpCfg.sf0ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal= ((float)atof(argv[6]) * (1U << 26) /(gCLI_mmwave_freq_scale_factor * 1e9));
            advChirpCfg.sf1ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal= ((float)atof(argv[7]) * (1U << 26) /(gCLI_mmwave_freq_scale_factor * 1e9));
            advChirpCfg.sf2ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal= ((float)atof(argv[8]) * (1U << 26) /(gCLI_mmwave_freq_scale_factor * 1e9));
            advChirpCfg.sf3ChirpParamDelta  = (rlInt32_t)tempVal;

            advChirpCfg.lutChirpParamSize   = (rlUInt8_t) (atoi(argv[14]));
            advChirpCfg.lutChirpParamScale  = (rlUInt8_t) (atoi(argv[15]));

            break;
        }
        case RL_LUT_CHIRP_FREQ_SLOPE_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Freq_Slope_Var
             **************************************************************************/
            advChirpCfg.deltaResetPeriod        = (rlUInt16_t) (atoi(argv[3]));
            advChirpCfg.deltaParamUpdatePeriod  = (rlUInt16_t) (atoi(argv[4]));

            /* Translate from KHz/us to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e6) * 900 /2^26 KHz/us]
                * units of mmwavelink format */
            tempVal = ((float)atof(argv[5]) * (1U << 26) / ((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));
            advChirpCfg.sf0ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal= ((float)atof(argv[6]) *  (1U << 26) / ((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));
            advChirpCfg.sf1ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal= ((float)atof(argv[7]) *  (1U << 26) / ((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));
            advChirpCfg.sf2ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal= ((float)atof(argv[8]) *  (1U << 26) / ((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));
            advChirpCfg.sf3ChirpParamDelta  = (rlInt32_t)tempVal;

            break;
        }
        case RL_LUT_CHIRP_IDLE_TIME_VAR:
        case RL_LUT_CHIRP_ADC_START_TIME_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Idle_Time_Var / Chirp_ADC_Start_Time_Var
             **************************************************************************/
            advChirpCfg.deltaResetPeriod        = (rlUInt16_t) (atoi(argv[3]));
            advChirpCfg.deltaParamUpdatePeriod  = (rlUInt16_t) (atoi(argv[4]));

            /* Translate from us to [1 LSB = 10ns] units of mmwavelink format */
            tempVal = ((float)atof (argv[5]) * 1000.0 / 10.0);
            advChirpCfg.sf0ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal = ((float)atof (argv[6]) * 1000.0 / 10.0);
            advChirpCfg.sf1ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal = ((float)atof (argv[7]) * 1000.0 / 10.0);
            advChirpCfg.sf2ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal = ((float)atof (argv[8]) * 1000.0 / 10.0);
            advChirpCfg.sf3ChirpParamDelta  = (rlInt32_t)tempVal;

            advChirpCfg.lutChirpParamSize   = (rlUInt8_t) (atoi(argv[14]));
            advChirpCfg.lutChirpParamScale  = (rlUInt8_t) (atoi(argv[15]));

            break;
        }
        case RL_LUT_TX0_PHASE_SHIFT_VAR:
        case RL_LUT_TX1_PHASE_SHIFT_VAR:
        case RL_LUT_TX2_PHASE_SHIFT_VAR:
        case RL_LUT_TX3_PHASE_SHIFT_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_TXN_Phase_Shifter
             **************************************************************************/
            advChirpCfg.deltaResetPeriod        = (rlUInt16_t) (atoi(argv[3]));
            advChirpCfg.deltaParamUpdatePeriod  = (rlUInt16_t) (atoi(argv[4]));

            /* Translate from degree to number of [1 LSB = (360°/2^16)] units of mmwavelink format */
            tempVal = ((float)atof (argv[5]) * (1U << 16) / 360);
            advChirpCfg.sf0ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal = ((float)atof (argv[6]) * (1U << 16) / 360);
            advChirpCfg.sf1ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal = ((float)atof (argv[7]) * (1U << 16) / 360);
            advChirpCfg.sf2ChirpParamDelta  = (rlInt32_t)tempVal;

            tempVal = ((float)atof (argv[8]) * (1U << 16) / 360);
            advChirpCfg.sf3ChirpParamDelta  = (rlInt32_t)tempVal;

            advChirpCfg.maxTxPhShiftIntDither = (rlUInt16_t) (atoi(argv[16]));

            break;
        }
        default:
        {
            CLI_write("Error: Invalid chirp param index!\n");
        }
    }/* end of switch case */

    /* Add the advanced chirp parameters to the list */
    if(MMWave_addAdvChirpParam(gCLI.cfg.mmWaveHandle, &advChirpCfg, &errCode) == NULL)
    {
        MMWave_ErrorLevel   errorLevel;
        int16_t             mmWaveErrorCode;
        int16_t             subsysErrorCode;
        /* Error: Unable to add the chirp parameter configurations. Return the error code. */
        MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);
        CLI_write ("Error: Configuration failed [Error code: %d Subsystem: %d]\n",
                        mmWaveErrorCode, subsysErrorCode);
        return errCode;
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the LUT Data configuration command for advanced chirp
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveLUTDataCfg (int32_t argc, char* argv[])
{
    uint8_t                 chirpParamIdx;
    rlUInt16_t              numOfPatterns;
    MMWave_AdvChirpHandle   advChirpHandle;
    rlAdvChirpCfg_t         advChirpCfg;
    rlInt32_t               lutDataBuf[MMWAVE_EACH_CHIRP_PARAM_MAX_LUT_BUF_SIZE]={0};
    uint32_t                lutDataIdx =0;
    float                   tempVal;
    int32_t                 errCode=0;

    /* Sanity Check: Advanced Chirp configuration is valid only for the Advance chirp Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Advanced Chirp (4 or 5)\n");
        return -1;
    }


    chirpParamIdx = (rlUInt8_t) (atoi (argv[1]));
    if(chirpParamIdx >= MMWAVE_ADVANCE_CHIRP_NUM_PARAMS)
    {
        CLI_write ("Error: Invalid chirp parameter index!\n");
        return -1;
    }

    /* Get the advance chirp handle */
    if(MMWave_getAdvChirpHandle(gCLI.cfg.mmWaveHandle, chirpParamIdx, &advChirpHandle, &errCode)<0)
    {
        /* Error: Unable to get the advance chirp handle. Return the error code */
        CLI_write("Error: Chirp paramater %d handle not found!", chirpParamIdx);
        return errCode;
    }

    /* get the advance chirp configs */
    if(MMWave_getAdvChirpParams(advChirpHandle, &advChirpCfg, &errCode) <0)
    {
        /* Error: Unable to get the advance chirp configurations. Return the error code */
        CLI_write("Error: Chirp paramater %d configurations not found!", chirpParamIdx);
        return errCode;
    }

    numOfPatterns = advChirpCfg.numOfPatterns;

    /* Sanity Check: Minimum argument check */
    if (argc != ( numOfPatterns+ 2))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }


    switch (chirpParamIdx)
    {
        case RL_LUT_CHIRP_PROFILE_VAR:
        case RL_LUT_CHIRP_TX_EN_VAR:
        case RL_LUT_CHIRP_BPM_VAL_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Profile_Var / CHIRP_TX_EN / CHIRP_BPM_VAL
             **************************************************************************/
            for(lutDataIdx = 0; lutDataIdx < numOfPatterns; lutDataIdx++)
            {
                lutDataBuf[lutDataIdx] = (rlInt32_t) atoi(argv[2+lutDataIdx]);
            }
            break;
        }
        case RL_LUT_CHIRP_FREQ_START_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Freq_Start_Var
             **************************************************************************/
            /* Translate from Hz to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e9 * 2^Scale) / 2^26 Hz]
             * units of mmwavelink format */
            for(lutDataIdx = 0; lutDataIdx < numOfPatterns; lutDataIdx++)
            {
                tempVal =  ((float)atof(argv[2+lutDataIdx]) * (1U << 26) /
                                (gCLI_mmwave_freq_scale_factor * 1e9 * (1U << advChirpCfg.lutChirpParamScale )));
                lutDataBuf[lutDataIdx] = (rlInt32_t)tempVal;
            }
            break;
        }
        case RL_LUT_CHIRP_FREQ_SLOPE_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Freq_Slope_Var
             **************************************************************************/
            /* Translate from KHz/us to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e6) * 900 /2^26 Hz/us]
             * units of mmwavelink format */
            for(lutDataIdx = 0; lutDataIdx < numOfPatterns; lutDataIdx++)
            {
                tempVal =  ((float)atof(argv[2+lutDataIdx]) * (1U << 26) /((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));
                lutDataBuf[lutDataIdx] = (rlInt32_t)tempVal;
            }
            break;
        }
        case RL_LUT_CHIRP_IDLE_TIME_VAR:
        case RL_LUT_CHIRP_ADC_START_TIME_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_Idle_Time_Var / Chirp_ADC_Start_Time_Var
             **************************************************************************/
            /* Translate from us to [1 LSB = 10ns * 2^Scale] units of mmwavelink format */
            for(lutDataIdx = 0; lutDataIdx < numOfPatterns; lutDataIdx++)
            {
                tempVal = ((float)atof(argv[2+lutDataIdx]) * 1000.0 / (10.0 * (1U << advChirpCfg.lutChirpParamScale)));
                lutDataBuf[lutDataIdx] = (rlInt32_t)tempVal;
            }
            break;
        }
        case RL_LUT_TX0_PHASE_SHIFT_VAR:
        case RL_LUT_TX1_PHASE_SHIFT_VAR:
        case RL_LUT_TX2_PHASE_SHIFT_VAR:
        case RL_LUT_TX3_PHASE_SHIFT_VAR:
        {
            /**************************************************************************
             * Parameter: Chirp_TXN_Phase_Shifter
             **************************************************************************/
            /* Translate from degree to number of [1 LSB = (360°/2^6)] units of mmwavelink format */
            for(lutDataIdx = 0; lutDataIdx < numOfPatterns; lutDataIdx++)
            {
                tempVal = ((float)atof(argv[2+lutDataIdx]) * (1U << 6) / 360);
                lutDataBuf[lutDataIdx] = (rlInt32_t)tempVal;
            }

            break;
        }
        default:
        {
            CLI_write("Error: Invalid chirp param index!\n");
        }
    }/* end of switch case */

    /* Add the LUT data to the advanced chirp config */
    if(MMWave_addAdvChirpLUTData(advChirpHandle, &lutDataBuf[0], &errCode) == NULL)
    {
        /* Error: Unable to add the LUT data. Return the error code. */
        MMWave_ErrorLevel   errorLevel;
        int16_t             mmWaveErrorCode;
        int16_t             subsysErrorCode;
        /* Error: Unable to add the chirp parameter configurations. Return the error code. */
        MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);
        CLI_write ("Error: Configuration failed [Error code: %d Subsystem: %d]\n",
                        mmWaveErrorCode, subsysErrorCode);
        return errCode;
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the frame configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveFrameCfg (int32_t argc, char* argv[])
{
    rlFrameCfg_t    frameCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 9)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }
#if defined(SOC_AWR2544)
    /* [AWR2544] Sanity Check: triggerSelect
     * 1   -   SW TRIGGER
     * 2   -   HW TRIGGER
     * 3   -   CPTS Trigger
     */
    if ((atoi (argv[7]) != 0) && (atoi (argv[7]) > 3U))
    {
        CLI_write ("Error: Invalid triggerSelect argument of the CLI command\r\n");
        return -1;
    }
#else
    /* [AWR294x] Sanity Check: triggerSelect
     * 1   -   SW TRIGGER
     * 2   -   HW TRIGGER
     */
    if ((atoi (argv[7]) != 0) && (atoi (argv[7]) > 2U))
    {
        CLI_write ("Error: Invalid triggerSelect argument of the CLI command\n");
        return -1;
    }
#endif

    /* Sanity Check: Frame configuration is valid only for the Frame Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME)&&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_LEGACY_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Chirp\n");
        return -1;
    }

    /* Initialize the frame configuration: */
    memset ((void *)&frameCfg, 0, sizeof(rlFrameCfg_t));

    /* Populate the frame configuration: */
    frameCfg.chirpStartIdx      = atoi (argv[1]);
    frameCfg.chirpEndIdx        = atoi (argv[2]);
    frameCfg.numLoops           = atoi (argv[3]);
    frameCfg.numFrames          = atoi (argv[4]);
    frameCfg.numAdcSamples      = atoi (argv[5]) * 2;
    frameCfg.framePeriodicity   = (uint32_t)((float)atof(argv[6]) * 1000000 / 5);
    frameCfg.triggerSelect      = atoi (argv[7]);
    frameCfg.frameTriggerDelay  = (uint32_t)((float)atof(argv[8]) * 1000000 / 5);

    /* Save Configuration to use later */
    memcpy((void *)&gCLIMMWaveControlCfg.u.frameCfg[0].frameCfg, (void *)&frameCfg, sizeof(rlFrameCfg_t));
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the advanced frame configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveAdvFrameCfg (int32_t argc, char* argv[])
{
    rlAdvFrameCfg_t  advFrameCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 7)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Frame configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME)&&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Advanced Frame\n");
        return -1;
    }

    /* Initialize the frame configuration: */
    memset ((void *)&advFrameCfg, 0, sizeof(rlAdvFrameCfg_t));

    /* Populate the frame configuration: */
    advFrameCfg.frameSeq.numOfSubFrames      = atoi (argv[1]);
    advFrameCfg.frameSeq.forceProfile        = atoi (argv[2]);
    advFrameCfg.frameSeq.numFrames           = atoi (argv[3]);
    advFrameCfg.frameSeq.triggerSelect       = atoi (argv[4]);
    advFrameCfg.frameSeq.frameTrigDelay      = (uint32_t)((float)atof(argv[5]) * 1000000 / 5);
    advFrameCfg.frameData.numSubFrames       = atoi (argv[6]);

    /* Save Configuration to use later */
    memcpy ((void *)&gCLIMMWaveControlCfg.u.advancedFrameCfg[0].frameCfg,
            (void *)&advFrameCfg, sizeof(rlAdvFrameCfg_t));
    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the subframe configuration command.
 *      Only valid when used in conjunction with the advanced frame configuration.
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveSubFrameCfg (int32_t argc, char* argv[])
{
    rlSubFrameCfg_t  subFrameCfg;
    uint8_t          subFrameNum;

    /* Sanity Check: Minimum argument check */
    if (argc != 11)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Sub Frame configuration is valid only for the Advanced Frame Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME)&&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_CHIRP_ADVANCED_FRAME))
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Advanced Frame\n");
        return -1;
    }

    /* Initialize the frame configuration: */
    memset ((void *)&subFrameCfg, 0, sizeof(rlSubFrameCfg_t));

    /* Populate the frame configuration: */
    subFrameNum                                  = (uint8_t)atoi (argv[1]);
    if (subFrameNum > gCLIMMWaveControlCfg.u.advancedFrameCfg[0].frameCfg.frameSeq.numOfSubFrames)
    {
        CLI_write ("Error: Invalid subframe number.\n");
        return -1;
    }
    subFrameCfg.forceProfileIdx     = atoi (argv[2]);
    subFrameCfg.chirpStartIdx       = atoi (argv[3]);
    subFrameCfg.numOfChirps         = atoi (argv[4]);
    subFrameCfg.numLoops            = atoi (argv[5]);
    subFrameCfg.burstPeriodicity    = (uint32_t)((float)atof(argv[6]) * 1000000 / 5);
    subFrameCfg.chirpStartIdxOffset = atoi (argv[7]);
    subFrameCfg.numOfBurst          = atoi (argv[8]);
    subFrameCfg.numOfBurstLoops     = atoi (argv[9]);
    subFrameCfg.subFramePeriodicity = (uint32_t)((float)atof(argv[10]) * 1000000 / 5);

    /* Save Configuration to use later */
    memcpy((void *)&gCLIMMWaveControlCfg.u.advancedFrameCfg[0].frameCfg.frameSeq.subFrameCfg[subFrameNum],
        (void *)&subFrameCfg, sizeof(rlSubFrameCfg_t));
    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the advanced Frame Data configuration command.
 *      Only valid when used in conjunction with the advanced frame configuration.
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */

static int32_t CLI_MMWaveAdvFrameDataCfg (int32_t argc, char* argv[])
{
    rlSubFrameDataCfg_t  subFrameDataCfg;
    uint8_t          subFrameNum;

    /* Sanity Check: Minimum argument check */
    if (argc != 5)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Sub Frame configuration is valid only for the Advanced Frame Mode: */
    if (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME)
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Advanced Frame\n");
        return -1;
    }

    /* Initialize the frame configuration: */
    memset ((void *)&subFrameDataCfg, 0, sizeof(rlSubFrameDataCfg_t));

    /* Populate the frame configuration: */
    subFrameNum                                  = (uint8_t)atoi (argv[1]);
    if (subFrameNum > gCLIMMWaveControlCfg.u.advancedFrameCfg[0].frameCfg.frameData.numSubFrames)
    {
        CLI_write ("Error: Invalid subframe number.\n");
        return -1;
    }

    subFrameDataCfg.numAdcSamples = atoi (argv[2]) * 2 ;
    subFrameDataCfg.totalChirps   = atoi (argv[3]);
    subFrameDataCfg.numChirpsInDataPacket = atoi (argv[4]);

    /* Save Configuration to use later */
    memcpy((void *)&gCLIMMWaveControlCfg.u.advancedFrameCfg[0].frameCfg.frameData.subframeDataCfg[subFrameNum],
        (void *)&subFrameDataCfg, sizeof(rlSubFrameDataCfg_t));

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the low power command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveLowPowerCfg (int32_t argc, char* argv[])
{
    rlLowPowerModeCfg_t     lowPowerCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize the channel configuration: */
    memset ((void *)&lowPowerCfg, 0, sizeof(rlLowPowerModeCfg_t));

    /* Populate the channel configuration: */
    lowPowerCfg.lpAdcMode     = atoi (argv[2]);

    /* Save Configuration to use later */
    memcpy((void *)&gCLIMMWaveOpenCfg.lowPowerMode, (void *)&lowPowerCfg, sizeof(rlLowPowerModeCfg_t));
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the continuous mode
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveContModeCfg (int32_t argc, char* argv[])
{
    MMWave_ContModeCfg *contCfg = &gCLIMMWaveControlCfg.u.continuousModeCfg[0];

    /* Sanity Check: Minimum argument check */
    if (argc != 10)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: Continuous configuration is valid only for the Continuous Mode: */
    if (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_CONTINUOUS)
    {
        CLI_write ("Error: Configuration is valid only if the DFE Output Mode is Continuous\n");
        return -1;
    }

    /* Populate the configuration: */
    contCfg->cfg.startFreqConst        = (uint32_t) (atof(argv[1]) * (1U << 26) /
                                                     gCLI_mmwave_freq_scale_factor);
    contCfg->cfg.txOutPowerBackoffCode = (uint32_t) atoi (argv[2]);
    contCfg->cfg.txPhaseShifter        = (uint32_t) atoi (argv[3]);
    contCfg->cfg.digOutSampleRate      = (uint16_t) atoi (argv[4]);
    contCfg->cfg.hpfCornerFreq1        = (uint8_t)  atoi (argv[5]);
    contCfg->cfg.hpfCornerFreq2        = (uint8_t)  atoi (argv[6]);
    contCfg->cfg.rxGain                = (uint16_t) atoi (argv[7]);
    /*argv[8] is reserved*/
    contCfg->dataTransSize             = (uint16_t) atoi (argv[9]);
    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the LDO Bypass function .
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveSetLdoBypassConfig(int32_t argc, char* argv[])
{
  rlReturnVal_t retVal = 0 ;

  rlUInt8_t deviceMap = (rlUInt8_t)atoi(argv[1]);
  rlRfLdoBypassCfg_t rfLdoBypassCfgArgs = { 0 };

  /* Sanity Check: Minimum argument check */
  if (argc != 5)
  {
    CLI_write("Error: Invalid usage of the CLI command\n");
    return -1;
  }
  rfLdoBypassCfgArgs.ldoBypassEnable = (rlUInt16_t)atoi(argv[2]);
  rfLdoBypassCfgArgs.supplyMonIrDrop = (rlUInt8_t)atoi(argv[3]);
  rfLdoBypassCfgArgs.ioSupplyIndicator = (rlUInt8_t)atoi(argv[4]);

  retVal = rlRfSetLdoBypassConfig(deviceMap, &rfLdoBypassCfgArgs);
  return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the Apll Synth BW Config function .
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveApllSynthBwConfig(int32_t argc, char* argv[]) {
  rlReturnVal_t retVal = 0;

  rlUInt8_t deviceMap = (rlUInt8_t)atoi(argv[1]);
  rlRfApllSynthBwControl_t rfApllSynthBwCfgArgs = { 0 };

  /* Sanity Check: Minimum argument check */
  if (argc != 9)
  {
    CLI_write("Error: Invalid usage of the CLI command\n");
    return -1;
  }

  rfApllSynthBwCfgArgs.synthIcpTrimVco1 = (rlUInt16_t)atoi(argv[2]);
  rfApllSynthBwCfgArgs.synthRzTrimVco1 = (rlUInt16_t)atoi(argv[3]);
  rfApllSynthBwCfgArgs.apllIcpTrim = (rlUInt16_t)atoi(argv[4]);
  rfApllSynthBwCfgArgs.apllRzTrimLpf = (rlUInt16_t)atoi(argv[5]);
  rfApllSynthBwCfgArgs.synthIcpTrimVco2 = (rlUInt16_t)atoi(argv[6]);
  rfApllSynthBwCfgArgs.apllRzTrimVco = (rlUInt16_t)atoi(argv[7]);
  rfApllSynthBwCfgArgs.synthRzTrimVco2 = (rlUInt16_t)atoi(argv[8]);

  retVal = rlRfApllSynthBwCtlConfig(deviceMap, &rfApllSynthBwCfgArgs);

  return retVal;
}


#if CLI_ADDITIONAL_COMMANDS
/**
 *  @b Description
 *  @n
 *      This is the CLI command Handler for setting the async direction.
 *
 *  @param[in]  argc
 *      Number of detected arguments
 *  @param[in] argv
 *      Detected arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      0   -   Success
 *  @retval
 *      -1  -   Setting of the async direction command failed.
 */
static int32_t CLI_setAsyncEventDir(int32_t argc, char* argv[])
{
  rlReturnVal_t retVal = 0;
  rlUInt8_t deviceMap = (rlUInt8_t)atoi(argv[1]);
  rlRfDevCfg_t rfDevCfg = { 0x0 };

  /* Sanity Check: Minimum argument check */
  if (argc != 7)
  {
    CLI_write("Error: Invalid usage of the CLI command\n");
    return -1;
  }

  /* set global and monitoring async event direction to Host */
  /* Set the CRC type of Async event received from radarSS */

  rfDevCfg.aeDirection = (rlUInt8_t)atoi(argv[2]);
  rfDevCfg.aeControl = (rlUInt8_t)atoi(argv[3]);
  rfDevCfg.bssAnaControl = (rlUInt8_t)atoi(argv[4]);
  rfDevCfg.bssDigCtrl = (rlUInt8_t)atoi(argv[5]);
  rfDevCfg.aeCrcConfig = (rlUInt8_t)atoi(argv[6]);


  retVal = rlRfSetDeviceCfg(deviceMap, &rfDevCfg);
  if (retVal != RL_RET_CODE_OK)
  {
    printf("rlRfSetDeviceCfg failed for deviceMap %u with error code %d\n\n",
      deviceMap, retVal);
    return -1;
  }
  else
  {
    printf("rlRfSetDeviceCfg success for deviceMap %u\n\n", deviceMap);
  }
  return retVal;

}


/**
 *  @b Description
 *  @n
 *      This is the CLI command Handler for the setting Miscellaneous configuration.
 *
 *  @param[in]  argc
 *      Number of detected arguments
 *  @param[in] argv
 *      Detected arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      0   -   Success
 *  @retval
 *      -1  -   Setting of the MiscConfig failed.
 */
static int32_t	CLI_setMiscConfig(int32_t argc, char* argv[])
{
  rlReturnVal_t retVal = 0;
  rlUInt8_t deviceMap = (rlUInt8_t)atoi(argv[1]);
  rlRfMiscConf_t MiscCfg = { 0 };

  /* Sanity Check: Minimum argument check */
  if (argc != 3)
  {
    CLI_write("Error: Invalid usage of the CLI command\n");
    return -1;
  }
  MiscCfg.miscCtl = (rlUInt8_t)atoi(argv[2]);

  retVal = rlRfSetMiscConfig(deviceMap, &MiscCfg);
  if (retVal != RL_RET_CODE_OK)
  {
    printf("rlRfSetMiscConfig failed for deviceMap %u with error code %d\n\n",
      deviceMap, retVal);
    return -1;
  }
  else
  {
    printf("rlRfSetMiscConfig success for deviceMap %u\n\n", deviceMap);
  }
  return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI command Handler for gpADC measurement config.
 *
 *  @param[in]  argc
 *      Number of detected arguments
 *  @param[in] argv
 *      Detected arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      0   -   Success
 *  @retval
 *      -1  -   Setting of the gpADC config failed.
 */
static int32_t CLI_gpadcMeasConfig(int32_t argc, char* argv[])
{
  rlReturnVal_t retVal = 0;

  rlUInt8_t deviceMap = (rlUInt8_t)atoi(argv[1]);
  rlGpAdcCfg_t gpadcCfg = { 0 };
  rlUInt8_t sampleCnt = 0;

  /* Sanity Check: Minimum argument check */
  if (argc != 5)
  {
    CLI_write("Error: Invalid usage of the CLI command\n");
    return -1;
  }

  gpadcCfg.enable = (rlUInt8_t)atoi(argv[2]);
  gpadcCfg.bufferEnable = (rlUInt8_t)atoi(argv[3]);
  sampleCnt = (rlUInt8_t)atoi(argv[4]);
  gpadcCfg.numOfSamples[0].sampleCnt = sampleCnt;
  gpadcCfg.numOfSamples[1].sampleCnt = sampleCnt;
  gpadcCfg.numOfSamples[2].sampleCnt = sampleCnt;
  gpadcCfg.numOfSamples[3].sampleCnt = sampleCnt;
  gpadcCfg.numOfSamples[4].sampleCnt = sampleCnt;
  gpadcCfg.numOfSamples[5].sampleCnt = sampleCnt;
  //gpadcCfg.numOfSamples[6].sampleCnt = sampleCnt;

  retVal = rlSetGpAdcConfig(deviceMap, &gpadcCfg);

  if (retVal != RL_RET_CODE_OK)
  {
    printf("rlRfSetDeviceCfg failed for deviceMap %u with error code %d\n\n",
      deviceMap, retVal);
    return -1;
  }
  else
  {
    printf("rlRfSetDeviceCfg success for deviceMap %u\n\n", deviceMap);

#if 0
    rlUInt32_t timeOutCnt = 0;
    if (retVal == 0)
    {
      while (gcli_bGpadcDataRcv == 0U)
      {
        osiSleep(1); /*Sleep 1 msec*/
        timeOutCnt++;
        if (timeOutCnt > CLI_API_RF_INIT_TIMEOUT)
        {
          retVal = RL_RET_CODE_RESP_TIMEOUT;
          break;
        }
      }
#endif
    }
    return retVal;
  }


  /**
    *  @b Description
    *  @n
    *      This is the CLI command Handler for power off.
    *
    *  @param[in]  argc
    *      Number of detected arguments
    *  @param[in] argv
    *      Detected arguments
    *
    *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
    *
    *  @retval
    *      0   -   Success
    *  @retval
    *      -1  -   Power Off failed.
    */

  static int32_t  CLI_powerOff(int32_t argc, char* argv[])
  {
    rlReturnVal_t retVal = 0;

    /* Sanity Check: Minimum argument check */
    if (argc != 1)
    {
      CLI_write("Error: Invalid usage of the CLI command\n");
      return -1;
    }
    #if 0
    retVal = rlDevicePowerOff();
    if (retVal != RL_RET_CODE_OK)
    {
      printf("rlDevicePowerOff failed with error code %d\n\n", retVal);
      return -1;
    }
    else
    {
      printf("rlDevicePowerOff success \n\n");
    }
    #endif
    return retVal;
  }


  /**
    *  @b Description
    *  @n
    *      This is the CLI command Handler for programming the Filter Config.
    *
    *  @param[in]  argc
    *      Number of detected arguments
    *  @param[in] argv
    *      Detected arguments
    *
    *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
    *
    *  @retval
    *      0   -   Success
    *  @retval
    *      -1  -   Power Off failed.
    */
    //deviceMap profileId coeffStartIdx progFiltLen progFiltFreqShift
  static int32_t  CLI_progFiltConfig(int32_t argc, char* argv[])
  {
    rlReturnVal_t retVal = 0;
    rlUInt8_t deviceMap = (rlUInt8_t)atoi(argv[1]);
    rlRfProgFiltConf_t progFiltCnfgArgs = { 0 };

    /* Sanity Check: Minimum argument check */
    if (argc != 6)
    {
      CLI_write("Error: Invalid usage of the CLI command\n");
      return -1;
    }

    progFiltCnfgArgs.profileId = (rlUInt8_t)atoi(argv[2]);
    progFiltCnfgArgs.coeffStartIdx = (rlUInt8_t)atoi(argv[3]);
    progFiltCnfgArgs.progFiltLen = (rlUInt8_t)atoi(argv[4]);
    progFiltCnfgArgs.progFiltFreqShift = (rlUInt8_t)atoi(argv[5]);

    printf("Calling rlRfSetProgFiltConfig with \ncoeffStartIdx[%d]\nprogFiltLen[%d] GHz\nprogFiltFreqShift[%d] MHz/uS \n\n",
      progFiltCnfgArgs.coeffStartIdx, progFiltCnfgArgs.progFiltLen, progFiltCnfgArgs.progFiltFreqShift);

    retVal = rlRfSetProgFiltConfig(deviceMap, &progFiltCnfgArgs);
    if (retVal != RL_RET_CODE_OK)
    {
      printf("rlRfSetProgFiltConfig failed with error code %d\n\n", retVal);
      return -1;
    }
    else
    {
      printf("rlRfSetProgFiltConfig success \n\n");
    }

    return retVal;
  }
#endif //CLI_ADDITIONAL_COMMANDS
/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the BPM configuration.
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveBPMCfgAdvanced (int32_t argc, char* argv[])
{
    rlBpmChirpCfg_t         bpmChirpCfg;
    int32_t                 errCode;

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Sanity Check: BPM Chirp configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if ((gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME) &&
        (gCLIMMWaveControlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME))
    {
        CLI_write ("Error: BPM Configuration is valid only if the DFE Output Mode is frame or advanced frame\n");
        return -1;
    }

    /* Initialize the chirp configuration: */
    memset ((void *)&bpmChirpCfg, 0, sizeof(rlBpmChirpCfg_t));

    /* Populate the chirp configuration: */
    bpmChirpCfg.chirpStartIdx   = atoi (argv[1]);
    bpmChirpCfg.chirpEndIdx     = atoi (argv[2]);
    bpmChirpCfg.constBpmVal     = atoi (argv[3]);

    /* Add the BPM chirp configuration to the list */
    if (MMWave_addBpmChirp (gCLI.cfg.mmWaveHandle, &bpmChirpCfg, &errCode) == NULL)
    {
        /* Error: Unable to add the BPM configuration. Return the error code. */
        return errCode;
    }

    return 0;

}


/**
 *  @b Description
 *  @n
 *      This is the mmWave extension initialization API
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t CLI_MMWaveExtensionInit(CLI_Cfg* ptrCLICfg)
{
    /* Sanity Check: We need the mmWave handle to work. */
    if (ptrCLICfg->mmWaveHandle == NULL)
    {
        return -1;
    }

    /* Initialize the mmWave control configuration: */
    memset ((void *)&gCLIMMWaveControlCfg, 0, sizeof(MMWave_CtrlCfg));
    gCLI_mmwave_freq_scale_factor = 3.6;
    return 0;
}



/**
 *  @b Description
 *  @n
 *      This is the mmWave extension handler which executes mmWave extension
 *      commands. This is invoked by the main CLI wrapper only if the extension
 *      was enabled.
 *
 *  @param[in]  argc
 *      Number of detected arguments
 *  @param[in] argv
 *      Detected arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      0   -   Matching mmWave extension command found
 *  @retval
 *      -1  -   No Matching mmWave extension command
 */
int32_t CLI_MMWaveExtensionHandler(int32_t argc, char* argv[])
{
    CLI_CmdTableEntry*  ptrCLICommandEntry;
    int32_t             cliStatus;
    int32_t             retVal = 0;

    /* Get the pointer to the mmWave extension table */
    ptrCLICommandEntry = &gCLIMMWaveExtensionTable[0];

    /* Cycle through all the registered externsion CLI commands: */
    while (ptrCLICommandEntry->cmdHandlerFxn != NULL)
    {
        /* Do we have a match? */
        if (strcmp(ptrCLICommandEntry->cmd, argv[0]) == 0)
        {
            /* YES: Pass this to the CLI registered function */
            cliStatus = ptrCLICommandEntry->cmdHandlerFxn (argc, argv);
            if (cliStatus == 0)
            {
                /* Successfully executed the CLI command: */
                CLI_write ("Done\r\n");
            }
            else
            {
                /* Error: The CLI command failed to execute */
                CLI_write ("Error %d\r\n", cliStatus);
            }
            break;
        }

        /* Get the next entry: */
        ptrCLICommandEntry++;
    }

    /* Was this a valid CLI command? */
    if (ptrCLICommandEntry->cmdHandlerFxn == NULL)
    {
        /* NO: The command was not a valid CLI mmWave extension command. Setup
         * the return value correctly. */
        retVal = -1;
    }
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      This is the mmWave extension handler which is invoked by the
 *      CLI Help command handler only if the extension was enabled.
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void CLI_MMWaveExtensionHelp(void)
{
    CLI_CmdTableEntry*  ptrCLICommandEntry;

    /* Get the pointer to the mmWave extension table */
    ptrCLICommandEntry = &gCLIMMWaveExtensionTable[0];

    /* Display the banner: */
    CLI_write ("\t****************************************************\r\n");
    CLI_write ("\tmmWave Extension Help\r\n");
    CLI_write ("\t****************************************************\r\n");

    /* Cycle through all the registered externsion CLI commands: */
    while (ptrCLICommandEntry->cmdHandlerFxn != NULL)
    {
        /* Display the help string*/
        CLI_write ("\t%s:\t%s\r\n",
                    ptrCLICommandEntry->cmd,
                   (ptrCLICommandEntry->helpString == NULL) ?
                    "No help available" :
                    ptrCLICommandEntry->helpString);

        /* Get the next entry: */
        ptrCLICommandEntry++;
    }
    return;
}

/**
 *  @b Description
 *  @n
 *      This is an API provided by the CLI mmWave extension handler to get
 *      the mmWave control configuration.
 *
 *  @param[out]  ptrCtrlCfg
 *      Pointer to the control configuration populated by the API
 *
 *  \ingroup CLI_UTIL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void CLI_getMMWaveExtensionConfig(MMWave_CtrlCfg* ptrCtrlCfg)
{
    memcpy ((void*)ptrCtrlCfg, (void*)&gCLIMMWaveControlCfg, sizeof(MMWave_CtrlCfg));
    return;
}

/**
 *  @b Description
 *  @n
 *      This is an API provided by the CLI mmWave extension handler to get
 *      the mmWave control configuration.
 *
 *  @param[out]  ptrOpenCfg
 *      Pointer to the open configuration populated by the API
 *
 *  \ingroup CLI_UTIL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void CLI_getMMWaveExtensionOpenConfig(MMWave_OpenCfg* ptrOpenCfg)
{
    memcpy ((void*)ptrOpenCfg, (void*)&gCLIMMWaveOpenCfg, sizeof(MMWave_OpenCfg));
    return;
}
