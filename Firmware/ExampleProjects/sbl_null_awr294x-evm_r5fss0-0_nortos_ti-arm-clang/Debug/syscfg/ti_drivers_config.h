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

#ifndef TI_DRIVERS_CONFIG_H_
#define TI_DRIVERS_CONFIG_H_

#include <stdint.h>
#include <drivers/hw_include/cslr_soc.h>
#include "ti_dpl_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Common Functions
 */

void System_init(void);
void System_deinit(void);

/*
 * BOOTLOADER
 */
#include <drivers/bootloader.h>

/* BOOTLOADER Instance Macros */
#define CONFIG_BOOTLOADER0 (0U)
#define CONFIG_BOOTLOADER_NUM_INSTANCES (1U)

/*
 *  HSM Client 
 */
#include <drivers/hsmclient.h>

/* sysconfig generated parameter QUEUE LENGTH */
#define SIPC_QUEUE_LENGTH   (32u)
/* Total number of secure cores */ 
#define SIPC_NUM_R5_CORES   (2u)


/*
 * UART
 */
#include <drivers/uart.h>

/* UART Instance Macros */
#define CONFIG_UART0 (0U)
#define CONFIG_UART_NUM_INSTANCES (1U)
#define CONFIG_UART_NUM_DMA_INSTANCES (0U)
/*
 * EDMA
 */
#include <drivers/edma.h>
#include <drivers/soc.h>

/* EDMA Instance Macros */
#define CONFIG_EDMA0_BASE_ADDR (CSL_RSS_TPCC_A_U_BASE)
#define CONFIG_EDMA0 (0U)
#define CONFIG_EDMA_NUM_INSTANCES (1U)

#include <drivers/soc.h>
#include <kernel/dpl/CycleCounterP.h>


#ifdef __cplusplus
}
#endif

#endif /* TI_DRIVERS_CONFIG_H_ */
