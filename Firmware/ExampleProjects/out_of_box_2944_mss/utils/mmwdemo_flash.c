/**
 *   @file  mmwdemo_flash.c
 *
 *   @brief
 *      The file implements the functions which are required to access QSPI flash
 *   from mmw demo.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2020 - 2025 Texas Instruments, Inc.
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

/* MCU+SDK Include Files: */
#include <board/flash.h>
#include <kernel/dpl/CacheP.h>

/* mmWave SDK Include Files: */
#include <ti/common/syscommon.h>
#include <ti/demo/utils/mmwdemo_flash.h>

#ifdef SOC_AWR2X44P
#if MMWDEMO_TDM
#include <ti/demo/awr2x44P/mmw_tdm/mss/mmw_mss.h>
#else
#include <ti/demo/awr2x44P/mmw_ddm/mss/mmw_mss.h>
#endif
#elif SOC_AWR294X
#include <ti/demo/awr294x/mmw/mss/mmw_mss.h>
#elif SOC_AWR2544
#if defined CSI_STREAM
#include <ti/demo/awr2544/mmw_csitx/mmw_common.h>
#else
#include <ti/demo/awr2544/mmw/mmw_common.h>
#endif
#endif

/**************************************************************************
 **************************** Local Functions *****************************
 **************************************************************************/
typedef struct mmwDemo_Flash_t
{

    /*! @brief   QSPI flash driver handle */
    Flash_Handle      QSPIFlashHandle;

    /*! @brief   Module initialized flag */
    bool              initialized;
}mmwDemo_Flash;

mmwDemo_Flash gMmwDemoFlash;

/**************************************************************************
 **************************** Monitor Functions *****************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is used to initialize QSPI and Flash interface.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashInit(void)
{
    int32_t          retVal = 0;

    gMmwDemoFlash.QSPIFlashHandle = gFlashHandle[0];

    if(gMmwDemoFlash.QSPIFlashHandle != NULL)
    {
        gMmwDemoFlash.initialized = true;
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL__QSPI;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to close Flash interface.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void mmwDemo_flashClose(void)
{
    gMmwDemoFlash.initialized = false;

    /* Graceful shutdown */
    Board_flashClose();

    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to read data from flash.
 *
 *  @param[in]  flashOffset
 *      Flash Offset to read data from
 *  @param[in]  readBuf
 *      Pointer to buffer that hold data read from flash
 *  @param[in]  size
 *      Size in bytes to be read from flash
 *
 *  @pre
 *      mmwDemo_flashInit
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashRead(uint32_t flashOffset, uint8_t *readBuf, uint32_t size)
{
    int32_t retVal = 0;
    int32_t status = SystemP_FAILURE;

    if(gMmwDemoFlash.initialized == true)
    {
        /* Read flash memory */
        status = Flash_read(gMmwDemoFlash.QSPIFlashHandle, flashOffset, readBuf, size);
        CacheP_wb(readBuf, size, CacheP_TYPE_ALL);

        if(status == SystemP_FAILURE)
        {
            retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
        }
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to write data to flash.
 *
 *  @param[in]  flashOffset
 *      Flash Offset to write data to
 *  @param[in]  writeBuf
 *      Pointer to buffer that hold data to be written to flash
 *  @param[in]  size
 *      Size in bytes to be written to flash
 *
 *  @pre
 *      mmwDemo_flashInit
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashWrite(uint32_t flashOffset, uint8_t *writeBuf, uint32_t size)
{
    int32_t           retVal = 0;
    uint32_t          blockNum = 0;     /* flash block number */
    uint32_t          pageNum = 0;      /* flash page number */
    int32_t           status = SystemP_SUCCESS;


    if(gMmwDemoFlash.initialized == true)
    {
        if(mmwDemo_flashEraseOneSector(flashOffset, &blockNum, &pageNum) < 0)
        {
            retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
        }
        else
        {
            /* Write buffer to flash */
            status = Flash_write(gMmwDemoFlash.QSPIFlashHandle, flashOffset, writeBuf, size);
            if(status != SystemP_SUCCESS)
            {
                retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
            }
        }
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to write data to flash.
 *
 *  @param[in]  flashOffset
 *      Flash Offset to write data to.
 *  @param[out]  blockNum
 *      Flash block number returned based on flash offset.
 *  @param[out]  pageNum
 *      Flash page number returned based on flash offset.
 *
 *  @pre
 *      mmwDemo_flashInit
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashEraseOneSector(uint32_t flashOffset, uint32_t* blockNum, uint32_t* pageNum)
{
    int32_t           retVal = 0;
    int32_t           status = SystemP_SUCCESS;

    if(gMmwDemoFlash.initialized == true)
    {
        status = Flash_offsetToBlkPage(gMmwDemoFlash.QSPIFlashHandle, flashOffset, blockNum, pageNum);
        if (status != SystemP_SUCCESS)
        {
            retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
        }
        else
        {
            /* Erase block, to which data has to be written */
            status = Flash_eraseBlk(gMmwDemoFlash.QSPIFlashHandle, *blockNum);
            if (status != SystemP_SUCCESS)
            {
                retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
            }
        }
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL;
    }

    return retVal;
}
