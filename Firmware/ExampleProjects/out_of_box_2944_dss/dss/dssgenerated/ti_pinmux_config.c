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
#include <drivers/pinmux.h>



static Pinmux_PerCfg_t gPinMuxMainDomainCfg1[] =
{
    /* Unused PAD registers configuration - Start */
    {
        PIN_PAD_BQ,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_BT,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_BX,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_BY,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CS,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_DG,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    /* Unused PAD registers configuration - End */
    {PINMUX_END, PINMUX_END}
};

static Pinmux_PerCfg_t gPinMuxMainDomainCfg[] = {
    /* Unused PAD registers configuration - Start */
    {
        PIN_PAD_AD,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_AE,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_AF,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_AG,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_AR,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_BU,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_BV,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_BW,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CT,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CU,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CV,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CW,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CX,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CY,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    {
        PIN_PAD_CZ,
        ( PIN_FORCE_INPUT_DISABLE | PIN_FORCE_OUTPUT_DISABLE )
    },
    /* Unused PAD registers configuration - End */
    {PINMUX_END, PINMUX_END}
};



/*
 * Pinmux
 */


void Pinmux_init(void)
{
    Pinmux_config(gPinMuxMainDomainCfg, PINMUX_DOMAIN_ID_MAIN);


    /* Check if additional PAD registers needs to be disabled for this device */
    if(SOC_rcmIsPadDisableRequired() == 0x1)
    {
        Pinmux_config(gPinMuxMainDomainCfg1, PINMUX_DOMAIN_ID_MAIN);
    }
}

