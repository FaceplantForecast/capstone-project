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

#include <stdio.h>
#include <kernel/dpl/DebugP.h>
#include "ti_drivers_config.h"
#include "ti_drivers_open_close.h"
#include "ti_board_open_close.h"
#include <drivers/ipc_rpmsg.h> //needed for shared memory
#include <stdlib.h> //needed for sscanf
#include <string.h> //needed for string operations
#include <C:\Users\there\Documents\Capstone\RadarFirmware\enums.h> //my custom universal values

//RPMessage objects
static RPMessage_Object gMsgObj;
static RPMessage_Object gRecvObj;

/* This function is the UART callback function
 */
static void uart_callback()
{
    //insert code here
}

/* This function sends commands to the correct cores to offload tasks
 */
static void send_to_core(uint16_t RemoteCoreID, uint16_t RemoteEndPt, char buf[64])
{
    uint16_t size = strlen(buf) + 1; //add 1 to account for terminating character
    RPMessage_send( buf, size,
                    RemoteCoreID, RemoteEndPt,
                    gSubSendEndPt, SystemP_WAIT_FOREVER);
}

/*
 * This does the subtraction operation on the data sent by the main core.
 */
void r5f1_main(void *args)
{
    /* Open drivers for the board and such */
    Drivers_open();
    Board_driversOpen();

    char buf[64];
    uint16_t buf_size;

    //setup
    RPMessage_CreateParams createParams;
    RPMessage_CreateParams_init(&createParams);
    createParams.localEndPt = gSubRecEndPt;
    RPMessage_construct(&gRecvObj, &createParams);

    //sending object
    RPMessage_CreateParams createParams2;
    RPMessage_CreateParams_init(&createParams2);
    createParams2.localEndPt = gSubSendEndPt;
    RPMessage_construct(&gMsgObj, &createParams2);

    while(1)
    {
        buf_size = sizeof(buf);
        uint16_t SrcCore = CSL_CORE_ID_R5FSS0_0;
        uint16_t SrcEndPt = gMainSendEndPt;
        int32_t status = RPMessage_recv(&gRecvObj, buf, &buf_size, &SrcCore, &SrcEndPt, SystemP_WAIT_FOREVER);

        if(status == 0) //if a message is actually received
        {
            DebugP_log("R5F1 got message: '%s' len=%u from core=%u ep=%u\r\n", buf, buf_size, SrcCore, SrcEndPt);
            int x, y;
            sscanf(buf, "SUB %d %d", &x, &y); //get numbers
            int result = x - y; //calculate
            snprintf(buf, sizeof(buf)-1, "%d", result);

            DebugP_log("R5F1 sending reply: %s\r\n", buf);

            //send result
            send_to_core(SrcCore, gMainRecEndPt, buf);
        }
    }

    Board_driversClose();
    Drivers_close();
};
