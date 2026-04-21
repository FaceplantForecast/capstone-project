################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Add inputs and outputs from these tool invocations to the build variables 
CMD_SRCS += \
../linker.cmd 

SYSCFG_SRCS += \
../example.syscfg 

C_SRCS += \
../R5F_0.c \
../cli.c \
../cli_mmwave.c \
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
../mmwave.c \
../mmwave_fullcfg.c \
../mmwave_link_common.c \
../mmwave_link_mailbox.c \
../mmwave_listlib.c \
../mmwave_osal.c \
../rl_controller.c \
../rl_device.c \
../rl_driver.c \
../rl_monitoring.c \
../rl_sensor.c 

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
./R5F_0.d \
./cli.d \
./cli_mmwave.d \
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
./mmwave.d \
./mmwave_fullcfg.d \
./mmwave_link_common.d \
./mmwave_link_mailbox.d \
./mmwave_listlib.d \
./mmwave_osal.d \
./rl_controller.d \
./rl_device.d \
./rl_driver.d \
./rl_monitoring.d \
./rl_sensor.d 

OBJS += \
./R5F_0.o \
./cli.o \
./cli_mmwave.o \
./syscfg/ti_dpl_config.o \
./syscfg/ti_drivers_config.o \
./syscfg/ti_drivers_open_close.o \
./syscfg/ti_pinmux_config.o \
./syscfg/ti_power_clock_config.o \
./syscfg/ti_board_config.o \
./syscfg/ti_board_open_close.o \
./syscfg/ti_enet_config.o \
./syscfg/ti_enet_open_close.o \
./syscfg/ti_enet_soc.o \
./syscfg/ti_enet_lwipif.o \
./main.o \
./mmwave.o \
./mmwave_fullcfg.o \
./mmwave_link_common.o \
./mmwave_link_mailbox.o \
./mmwave_listlib.o \
./mmwave_osal.o \
./rl_controller.o \
./rl_device.o \
./rl_driver.o \
./rl_monitoring.o \
./rl_sensor.o 

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
"R5F_0.o" \
"cli.o" \
"cli_mmwave.o" \
"syscfg\ti_dpl_config.o" \
"syscfg\ti_drivers_config.o" \
"syscfg\ti_drivers_open_close.o" \
"syscfg\ti_pinmux_config.o" \
"syscfg\ti_power_clock_config.o" \
"syscfg\ti_board_config.o" \
"syscfg\ti_board_open_close.o" \
"syscfg\ti_enet_config.o" \
"syscfg\ti_enet_open_close.o" \
"syscfg\ti_enet_soc.o" \
"syscfg\ti_enet_lwipif.o" \
"main.o" \
"mmwave.o" \
"mmwave_fullcfg.o" \
"mmwave_link_common.o" \
"mmwave_link_mailbox.o" \
"mmwave_listlib.o" \
"mmwave_osal.o" \
"rl_controller.o" \
"rl_device.o" \
"rl_driver.o" \
"rl_monitoring.o" \
"rl_sensor.o" 

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
"R5F_0.d" \
"cli.d" \
"cli_mmwave.d" \
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
"mmwave.d" \
"mmwave_fullcfg.d" \
"mmwave_link_common.d" \
"mmwave_link_mailbox.d" \
"mmwave_listlib.d" \
"mmwave_osal.d" \
"rl_controller.d" \
"rl_device.d" \
"rl_driver.d" \
"rl_monitoring.d" \
"rl_sensor.d" 

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
"../R5F_0.c" \
"../cli.c" \
"../cli_mmwave.c" \
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
"../mmwave.c" \
"../mmwave_fullcfg.c" \
"../mmwave_link_common.c" \
"../mmwave_link_mailbox.c" \
"../mmwave_listlib.c" \
"../mmwave_osal.c" \
"../rl_controller.c" \
"../rl_device.c" \
"../rl_driver.c" \
"../rl_monitoring.c" \
"../rl_sensor.c" 

SYSCFG_SRCS__QUOTED += \
"../example.syscfg" 


