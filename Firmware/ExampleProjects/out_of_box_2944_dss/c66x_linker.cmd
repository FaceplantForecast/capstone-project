/*
 * Copyright (c) 2020-2025, Texas Instruments Incorporated
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * *  Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * *  Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

--stack_size=8192

--heap_size=4096
--retain=_vectors

MEMORY
{
    DSS_L2:   ORIGIN = 0x800000, LENGTH = 0x60000
    /* Total of 2.5 MB of DSS L3 is available */
    DSS_L3:   ORIGIN = 0x88000000, LENGTH = 0x00280000
    HWA_RAM:  ORIGIN = 0x82000000, LENGTH = 0x00020000

    /* 1st 512 B of DSS mailbox memory and MSS mailbox memory is used for IPC with R4 and should not be used by application */
    /* MSS mailbox memory is used as shared memory, we dont use bottom 32*6 bytes, since its used as SW queue by ipc_notify */
    RTOS_NORTOS_IPC_SHM_MEM : ORIGIN = 0xC5000200, LENGTH = 0x1D40
}


SECTIONS
{
    /* hard addresses forces vecs to be allocated there */
    .text:vectors: {. = align(1024); } > 0x00800000
    .text:      {} palign(8) > DSS_L2
    .const:     {} palign(8) > DSS_L2
    .cinit:     {} palign(8) > DSS_L2
    .data:      {} palign(8) > DSS_L2
    .stack:     {} palign(8) > DSS_L2
    .switch:    {} palign(8) > DSS_L2
    .cio:       {} palign(8) > DSS_L2
    .sysmem:    {} palign(8) > DSS_L2
    .fardata:   {} palign(8) > DSS_L2
    .far:       {} palign(8) > DSS_L2

    /* These should be grouped together to avoid STATIC_BASE relative relocation linker error */
    GROUP {
        .rodata:    {} palign(8)
        .bss:       {} palign(8)
        .neardata:  {} palign(8)
    } > DSS_L2

    /* any data buffer needed to be put in L3 can be assigned this section name */
    .bss.dss_l3 {} palign(8) > DSS_L3

    /* this is used only when IPC RPMessage is enabled, else this is not used */
    .bss.ipc_vring_mem   (NOLOAD) : {} palign(8) > RTOS_NORTOS_IPC_SHM_MEM
}

