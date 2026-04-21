#ifndef ENUMS_H //makes sure it doesn't get repeatedly defined by multiple files
#define ENUMS_H

//enum for endpoints
enum ipc_end_pt
{
    /*COMMAND ENDPOINTS*/
    gMainSendEndPt = 3U, //R5F_0
    gMainRecEndPt = 4U, //R5F_0
    gSubSendEndPt = 5U, //R5F_1
    gSubRecEndPt = 6U, //R5F_0
    gDSPSendEndPt = 7U, //DSP
    gDSPRecEndPt = 8U //DSP
};

#endif
