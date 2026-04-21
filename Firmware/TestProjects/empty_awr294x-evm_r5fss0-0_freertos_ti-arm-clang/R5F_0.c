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

//DEFAULT INCLUSIONS
#include <stdio.h>
#include <kernel/dpl/DebugP.h>
#include "ti_drivers_config.h"
#include "ti_drivers_open_close.h"
#include "ti_board_open_close.h"
//END DEFAULT INCLUSIONS

#include <stdlib.h> //needed for atoi
#include <string.h> //needed for parsing strings
#include <drivers/ipc_rpmsg.h> //needed for shared memory
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/utils/cli/cli.h> //needed for CLI wrapper
#include "FreeRTOS.h" //needed for task management
#include "task.h" //needed for task management
#include <C:\Users\there\Documents\Capstone\RadarFirmware\enums.h> //my custom universal values

//RPMessage objects
static RPMessage_Object gMsgObj;
static RPMessage_Object gRecvObj;

//command structure
typedef struct {
    char op[4]; //opcode
    int x; //first number
    int y; //second number
} MathCmd;

/* This function sends commands to the correct cores to offload tasks
 */
static void send_to_core(uint16_t RemoteCoreID, uint16_t RemoteEndPt, char buf[64])
{
    uint16_t size = strlen(buf) + 1; //add 1 to account for terminating character
    RPMessage_send( buf, size,
                    RemoteCoreID, RemoteEndPt,
                    gMainSendEndPt, SystemP_WAIT_FOREVER);
}

/* ======================= Command Handlers ======================= */

/* This function handles addition.
 * It is run locally
 */
static int32_t cmd_add(int32_t argc, char* argv[])
{
    if(argc == 3) //make sure there are 3 parts of the argument
    {
        int x = atoi(argv[1]); //set second part (first number) as x
        int y = atoi(argv[2]); //set third part (second number) as y
        int result = x + y; //add numbers

        //give result to the CLI
        DebugP_log("ADD result = %d\r\n", result);
    }
    else
    {
        DebugP_log("Usage: ADD X Y\r\n");
        return -1;
    }
    return 0;
}

/* This function handles subtraction.
 * It is run on the R5F1 core.
 */
static int32_t cmd_sub(int32_t argc, char* argv[])
{
    if(argc == 3) //make sure there are 3 parts of the argument
    {
        char buf[64];
        uint16_t buf_size = sizeof(buf);
        snprintf(buf, buf_size-1, "SUB %s %s", argv[1], argv[2]);
        send_to_core(CSL_CORE_ID_R5FSS0_1, gSubRecEndPt, buf);

        //create variables for core id and endpoint
        uint16_t SrcCore = CSL_CORE_ID_R5FSS0_1;
        uint16_t SrcEndPt = gSubSendEndPt;

        //wait for response
        char recv_buf[64];
        uint16_t recv_buf_size = sizeof(recv_buf);
        int32_t status = RPMessage_recv(&gRecvObj, recv_buf, &recv_buf_size, &SrcCore, &SrcEndPt, SystemP_WAIT_FOREVER);
        if(status == 0)
        {
            DebugP_log("SUB result = %s\r\n", recv_buf);
        }
    }
    else
    {
        DebugP_log("Usage: SUB X Y\r\n");
        return -1;
    }
    return 0;
}

/* This function handles multiplication.
 * It is run on the DSP core.
 */
static int32_t cmd_mul(int32_t argc, char* argv[])
{
    if(argc == 3) //make sure there are 3 parts of the argument
    {
        char buf[64];
        uint16_t buf_size = sizeof(buf);
        snprintf(buf, buf_size-1, "MUL %s %s", argv[1], argv[2]);
        send_to_core(CSL_CORE_ID_C66SS0, gDSPRecEndPt, buf);

        //create variables for core id and endpoint
        uint16_t SrcCore = CSL_CORE_ID_C66SS0;
        uint16_t SrcEndPt = gDSPSendEndPt;

        //wait for response
        char recv_buf[64];
        uint16_t recv_buf_size = sizeof(recv_buf);
        int32_t status = RPMessage_recv(&gRecvObj, recv_buf, &recv_buf_size, &SrcCore, &SrcEndPt, SystemP_WAIT_FOREVER);
        if(status == 0)
        {
            DebugP_log("MUL result = %s\r\n", recv_buf);
        }
    }
    else
    {
        DebugP_log("Usage: MUL X Y\r\n");
        return -1;
    }
    return 0;
}

/* 
 * This function handles the setting up the CLI commands
 */
static int32_t cli_setup(CLI_Cfg cliCfg)
{
    /*-----BASIC TEST COMMANDS-----*/
    //addition
    cliCfg.tableEntry[0].cmd = "ADD";
    cliCfg.tableEntry[0].helpString = "Add two integers";
    cliCfg.tableEntry[0].cmdHandlerFxn = cmd_add;

    //subtraction
    cliCfg.tableEntry[1].cmd = "SUB";
    cliCfg.tableEntry[1].helpString = "Subtract two integers";
    cliCfg.tableEntry[1].cmdHandlerFxn = cmd_sub;

    //multiplication
    cliCfg.tableEntry[2].cmd = "MUL";
    cliCfg.tableEntry[2].helpString = "Multiply two integers";
    cliCfg.tableEntry[2].cmdHandlerFxn = cmd_mul;

    return 0;
}

/*
 * This is adapted from the empty project provided in the ti sdk.
 * This handles the CLI interface and routing tasks.
 */
void r5f0_main(void *args)
{
    /* Open drivers to open the UART driver for console */
    Drivers_open();
    Board_driversOpen();

    //RPMessage setup
    RPMessage_CreateParams createParams;
    RPMessage_CreateParams_init(&createParams);
    createParams.localEndPt = gMainRecEndPt;
    RPMessage_construct(&gRecvObj, &createParams);
    DebugP_log("R5F0 RPMessage local endpoint = %u\r\n", gMainRecEndPt);

    //sending object
    RPMessage_CreateParams createParams2;
    RPMessage_CreateParams_init(&createParams2);
    createParams2.localEndPt = gMainSendEndPt;
    RPMessage_construct(&gMsgObj, &createParams2);

    //initiate CLI interface
    CLI_Cfg cliCfg = {0};
    cliCfg.cliUartHandle = gUartHandle[CONFIG_UART0]; //UART handle from sysconfig
    cliCfg.cliPrompt = "R5F0> ";
    cliCfg.taskPriority = 3;

    //set up CLI commands
    cli_setup(cliCfg);

    /*-----OPEN CLI-----*/
    CLI_open(&cliCfg);
    while(1)
    {
        vTaskDelay(500); //keep task alive
    }
    Board_driversClose();
    Drivers_close();
};
