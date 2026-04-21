/*----------------------------------------------------------------------------*/
/* r5f_linker.cmd                                                                 */
/*                                                                            */
/* (c) Texas Instruments 2020, All rights reserved.                           */
/*                                                                            */

/* USER CODE BEGIN (0) */
/* USER CODE END */
--retain="*(.intc_text)"
--retain="*(.irqStack)"
--retain="*(.fiqStack)"
--retain="*(.abortStack)"
--retain="*(.undStack)"
--retain="*(.svcStack)"
-stack  0x2000                              /* SOFTWARE STACK SIZE           */
-heap   0x1000                              /* HEAP AREA SIZE                */

-e_vectors  /* This is the entry of the application, _vector MUST be plabed starting address 0x0 */

/* Stack Sizes for various modes */
__IRQ_STACK_SIZE = 256;
__FIQ_STACK_SIZE = 256;
__ABORT_STACK_SIZE = 256;
__UNDEFINED_STACK_SIZE = 256;
__SVC_STACK_SIZE = 4096;


/*----------------------------------------------------------------------------*/
/* Linker Settings                                                            */


/*----------------------------------------------------------------------------*/
/* Memory Map                                                                 */
MEMORY{
PAGE 0:
    RESET_VECTORS  (X)  : origin=0x00000000 length=0x00000100
    TCMA_RAM (RX) : origin=0x00000100 length=0x0000FF00
    TCMB_RAM (RW) : origin=0x00080000 length=0x00010000

    /* when using multi-core application's i.e more than one R5F active, make sure
     * this memory does not overlap with other R5F's
     */
    SBL_RESERVED_L2_RAM (RW)   : origin=0x10200000 length=0x00020000

    MSS_L2 (RW)   : origin=0x10220000 length=0x000C8000

    /* Total of 2.5 MB of DSS L3 is available. Make sure R5F core does not use this memory in demo applications */
    DSS_L3 (RW)   : origin=0x88000000 length=0x00280000
    HWA_RAM (RW)  : origin=0x82000000 length=0x00020000

    /* 1st 512 B of DSS mailbox memory and MSS mailbox memory is used for IPC with R4 and should not be used by application */
    /* MSS mailbox memory is used as shared memory, we dont use bottom 32*6 bytes, since its used as SW queue by ipc_notify */
    RTOS_NORTOS_IPC_SHM_MEM : ORIGIN = 0xC5000200, LENGTH = 0x1D40
}

/*----------------------------------------------------------------------------*/
/* Section Configuration                                                      */
SECTIONS{
    /* This has the R5F entry point and vector table, this MUST be at 0x0 */
    .vectors:{} palign(8) > RESET_VECTORS

    /* This has the R5F boot code until MPU is enabled,  this MUST be at a address < 0x80000000
     * i.e this cannot be placed in DDR
     */
    GROUP {
        .text.hwi: palign(8)
        .text.cache: palign(8)
        .text.mpu: palign(8)
        .text.boot: palign(8)
        .text:abort: palign(8) /* this helps in loading symbols when using XIP mode */
    } > MSS_L2

    /* This is rest of code. This can be placed in DDR if DDR is available and needed */
    GROUP {
        .text:   {} palign(8)   /* This is where code resides */
        .rodata: {} palign(8)   /* This is where const's go */
    } > MSS_L2

    /* This is rest of initialized data. This can be placed in DDR if DDR is available and needed */
    GROUP {
        .data:   {} palign(8)   /* This is where initialized globals and static go */
    } > MSS_L2

    /* This is rest of uninitialized data. This can be placed in DDR if DDR is available and needed */
    GROUP {
        .bss:    {} palign(8)   /* This is where uninitialized globals go */
        RUN_START(__BSS_START)
        RUN_END(__BSS_END)
        .sysmem: {} palign(8)   /* This is where the malloc heap goes */
        .stack:  {} palign(8)   /* This is where the main() stack goes */
    } > MSS_L2

    /* This is where the stacks for different R5F modes go */
    GROUP {
        .irqstack: {. = . + __IRQ_STACK_SIZE;} align(8)
        RUN_START(__IRQ_STACK_START)
        RUN_END(__IRQ_STACK_END)
        .fiqstack: {. = . + __FIQ_STACK_SIZE;} align(8)
        RUN_START(__FIQ_STACK_START)
        RUN_END(__FIQ_STACK_END)
        .svcstack: {. = . + __SVC_STACK_SIZE;} align(8)
        RUN_START(__SVC_STACK_START)
        RUN_END(__SVC_STACK_END)
        .abortstack: {. = . + __ABORT_STACK_SIZE;} align(8)
        RUN_START(__ABORT_STACK_START)
        RUN_END(__ABORT_STACK_END)
        .undefinedstack: {. = . + __UNDEFINED_STACK_SIZE;} align(8)
        RUN_START(__UNDEFINED_STACK_START)
        RUN_END(__UNDEFINED_STACK_END)
    } > MSS_L2

    /* any data buffer needed to be put in L3 can be assigned this section name */
    .bss.dss_l3 {} > DSS_L3

    /* this is used only when IPC RPMessage is enabled, else this is not used */
    .bss.ipc_vring_mem   (NOLOAD) : {} > RTOS_NORTOS_IPC_SHM_MEM
}
/*----------------------------------------------------------------------------*/

