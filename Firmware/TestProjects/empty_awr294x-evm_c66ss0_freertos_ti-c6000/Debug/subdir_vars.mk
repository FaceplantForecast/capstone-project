################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Add inputs and outputs from these tool invocations to the build variables 
CMD_SRCS += \
../linker.cmd \
../mmw_dss_linker.cmd 

SYSCFG_SRCS += \
../example.syscfg 

C_SRCS += \
../DSP.c \
../data_path.c \
./syscfg/ti_dpl_config.c \
./syscfg/ti_drivers_config.c \
./syscfg/ti_drivers_open_close.c \
./syscfg/ti_pinmux_config.c \
./syscfg/ti_power_clock_config.c \
./syscfg/ti_board_config.c \
./syscfg/ti_board_open_close.c \
./syscfg/ti_enet_config.c \
./syscfg/ti_enet_open_close.c \
./syscfg/ti_enet_soc.c \
./syscfg/ti_enet_lwipif.c \
../main.c \
../objectdetection.c 

GEN_FILES += \
./syscfg/ti_dpl_config.c \
./syscfg/ti_drivers_config.c \
./syscfg/ti_drivers_open_close.c \
./syscfg/ti_pinmux_config.c \
./syscfg/ti_power_clock_config.c \
./syscfg/ti_board_config.c \
./syscfg/ti_board_open_close.c \
./syscfg/ti_enet_config.c \
./syscfg/ti_enet_open_close.c \
./syscfg/ti_enet_soc.c \
./syscfg/ti_enet_lwipif.c 

GEN_MISC_DIRS += \
./syscfg 

C_DEPS += \
./DSP.d \
./data_path.d \
./syscfg/ti_dpl_config.d \
./syscfg/ti_drivers_config.d \
./syscfg/ti_drivers_open_close.d \
./syscfg/ti_pinmux_config.d \
./syscfg/ti_power_clock_config.d \
./syscfg/ti_board_config.d \
./syscfg/ti_board_open_close.d \
./syscfg/ti_enet_config.d \
./syscfg/ti_enet_open_close.d \
./syscfg/ti_enet_soc.d \
./syscfg/ti_enet_lwipif.d \
./main.d \
./objectdetection.d 

OBJS += \
./DSP.obj \
./data_path.obj \
./syscfg/ti_dpl_config.obj \
./syscfg/ti_drivers_config.obj \
./syscfg/ti_drivers_open_close.obj \
./syscfg/ti_pinmux_config.obj \
./syscfg/ti_power_clock_config.obj \
./syscfg/ti_board_config.obj \
./syscfg/ti_board_open_close.obj \
./syscfg/ti_enet_config.obj \
./syscfg/ti_enet_open_close.obj \
./syscfg/ti_enet_soc.obj \
./syscfg/ti_enet_lwipif.obj \
./main.obj \
./objectdetection.obj 

GEN_MISC_FILES += \
./syscfg/ti_dpl_config.h \
./syscfg/ti_drivers_config.h \
./syscfg/ti_drivers_open_close.h \
./syscfg/ti_board_config.h \
./syscfg/ti_board_open_close.h \
./syscfg/ti_enet_config.h \
./syscfg/ti_enet_open_close.h \
./syscfg/ti_enet_lwipif.h 

GEN_MISC_DIRS__QUOTED += \
"syscfg" 

OBJS__QUOTED += \
"DSP.obj" \
"data_path.obj" \
"syscfg\ti_dpl_config.obj" \
"syscfg\ti_drivers_config.obj" \
"syscfg\ti_drivers_open_close.obj" \
"syscfg\ti_pinmux_config.obj" \
"syscfg\ti_power_clock_config.obj" \
"syscfg\ti_board_config.obj" \
"syscfg\ti_board_open_close.obj" \
"syscfg\ti_enet_config.obj" \
"syscfg\ti_enet_open_close.obj" \
"syscfg\ti_enet_soc.obj" \
"syscfg\ti_enet_lwipif.obj" \
"main.obj" \
"objectdetection.obj" 

GEN_MISC_FILES__QUOTED += \
"syscfg\ti_dpl_config.h" \
"syscfg\ti_drivers_config.h" \
"syscfg\ti_drivers_open_close.h" \
"syscfg\ti_board_config.h" \
"syscfg\ti_board_open_close.h" \
"syscfg\ti_enet_config.h" \
"syscfg\ti_enet_open_close.h" \
"syscfg\ti_enet_lwipif.h" 

C_DEPS__QUOTED += \
"DSP.d" \
"data_path.d" \
"syscfg\ti_dpl_config.d" \
"syscfg\ti_drivers_config.d" \
"syscfg\ti_drivers_open_close.d" \
"syscfg\ti_pinmux_config.d" \
"syscfg\ti_power_clock_config.d" \
"syscfg\ti_board_config.d" \
"syscfg\ti_board_open_close.d" \
"syscfg\ti_enet_config.d" \
"syscfg\ti_enet_open_close.d" \
"syscfg\ti_enet_soc.d" \
"syscfg\ti_enet_lwipif.d" \
"main.d" \
"objectdetection.d" 

GEN_FILES__QUOTED += \
"syscfg\ti_dpl_config.c" \
"syscfg\ti_drivers_config.c" \
"syscfg\ti_drivers_open_close.c" \
"syscfg\ti_pinmux_config.c" \
"syscfg\ti_power_clock_config.c" \
"syscfg\ti_board_config.c" \
"syscfg\ti_board_open_close.c" \
"syscfg\ti_enet_config.c" \
"syscfg\ti_enet_open_close.c" \
"syscfg\ti_enet_soc.c" \
"syscfg\ti_enet_lwipif.c" 

C_SRCS__QUOTED += \
"../DSP.c" \
"../data_path.c" \
"./syscfg/ti_dpl_config.c" \
"./syscfg/ti_drivers_config.c" \
"./syscfg/ti_drivers_open_close.c" \
"./syscfg/ti_pinmux_config.c" \
"./syscfg/ti_power_clock_config.c" \
"./syscfg/ti_board_config.c" \
"./syscfg/ti_board_open_close.c" \
"./syscfg/ti_enet_config.c" \
"./syscfg/ti_enet_open_close.c" \
"./syscfg/ti_enet_soc.c" \
"./syscfg/ti_enet_lwipif.c" \
"../main.c" \
"../objectdetection.c" 

SYSCFG_SRCS__QUOTED += \
"../example.syscfg" 


