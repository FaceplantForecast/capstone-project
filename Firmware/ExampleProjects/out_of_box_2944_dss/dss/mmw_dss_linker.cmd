/*----------------------------------------------------------------------------*/
/* Linker Settings                                                            */
--retain="*(.intvecs)"

/*----------------------------------------------------------------------------*/
/* Section Configuration                                                      */
SECTIONS
{
    systemHeap : {} palign(8) > DSS_L2
    .hwaBufs: load = HWA_RAM, type = NOINIT
    .l3ram: {} palign(8) > DSS_L3
    .dpc_l2Heap: { } palign(8) > DSS_L2
    .demoSharedMem: { } palign(8) > DSS_L3
    .preProcBuf: {} palign(8) > DSS_L2
}
/*----------------------------------------------------------------------------*/
/*.demoSharedMem: { } palign(8) > HSRAM*/
