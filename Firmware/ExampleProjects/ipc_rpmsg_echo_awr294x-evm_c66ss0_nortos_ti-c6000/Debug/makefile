################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

CG_TOOL_ROOT := C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS

GEN_OPTS__FLAG := 
GEN_CMDS__FLAG := 

ORDERED_OBJS += \
"./syscfg/ti_dpl_config.obj" \
"./syscfg/ti_drivers_config.obj" \
"./syscfg/ti_drivers_open_close.obj" \
"./syscfg/ti_pinmux_config.obj" \
"./syscfg/ti_power_clock_config.obj" \
"./syscfg/ti_board_config.obj" \
"./syscfg/ti_board_open_close.obj" \
"./syscfg/ti_enet_config.obj" \
"./syscfg/ti_enet_open_close.obj" \
"./syscfg/ti_enet_soc.obj" \
"./syscfg/ti_enet_lwipif.obj" \
"./ipc_rpmsg_echo.obj" \
"./main.obj" \
"../linker.cmd" \
$(GEN_CMDS__FLAG) \
-lnortos.awr294x.c66.ti-c6000.debug.lib \
-ldrivers.awr294x.c66.ti-c6000.debug.lib \
-lboard.awr294x.c66.ti-c6000.debug.lib \
-llibc.a \

-include ../makefile.init

RM := DEL /F
RMDIR := RMDIR /S/Q

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir_vars.mk
-include subdir_rules.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C55_DEPS)),)
-include $(C55_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(S67_DEPS)),)
-include $(S67_DEPS)
endif
ifneq ($(strip $(S62_DEPS)),)
-include $(S62_DEPS)
endif
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(OPT_DEPS)),)
-include $(OPT_DEPS)
endif
ifneq ($(strip $(C??_DEPS)),)
-include $(C??_DEPS)
endif
ifneq ($(strip $(ASM_UPPER_DEPS)),)
-include $(ASM_UPPER_DEPS)
endif
ifneq ($(strip $(S??_DEPS)),)
-include $(S??_DEPS)
endif
ifneq ($(strip $(C64_DEPS)),)
-include $(C64_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(S64_DEPS)),)
-include $(S64_DEPS)
endif
ifneq ($(strip $(INO_DEPS)),)
-include $(INO_DEPS)
endif
ifneq ($(strip $(CLA_DEPS)),)
-include $(CLA_DEPS)
endif
ifneq ($(strip $(S55_DEPS)),)
-include $(S55_DEPS)
endif
ifneq ($(strip $(SV7A_DEPS)),)
-include $(SV7A_DEPS)
endif
ifneq ($(strip $(C62_DEPS)),)
-include $(C62_DEPS)
endif
ifneq ($(strip $(C67_DEPS)),)
-include $(C67_DEPS)
endif
ifneq ($(strip $(PDE_DEPS)),)
-include $(PDE_DEPS)
endif
ifneq ($(strip $(K_DEPS)),)
-include $(K_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C43_DEPS)),)
-include $(C43_DEPS)
endif
ifneq ($(strip $(S43_DEPS)),)
-include $(S43_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(SA_DEPS)),)
-include $(SA_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
EXE_OUTPUTS += \
ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000.out 

EXE_OUTPUTS__QUOTED += \
"ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000.out" 


# All Target
all: $(OBJS) $(CMD_SRCS) $(GEN_CMDS)
	@$(MAKE) --no-print-directory -Onone "ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000.out"

# Tool invocations
ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000.out: $(OBJS) $(CMD_SRCS) $(GEN_CMDS)
	@echo 'Building target: "$@"'
	@echo 'Invoking: C6000 Linker'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/bin/cl6x" -mv6600 --define=SOC_AWR294X --define=_DEBUG_=1 -g --c99 --diag_warning=225 --diag_wrap=off --display_error_number --emit_warnings_as_errors --quiet --gen_func_subsections=on --assume_control_regs_read --mem_model:const=data --mem_model:data=far_aggregates --disable_push_pop --fp_mode=relaxed --remove_hooks_when_inlining --gen_opt_info=2 -z -m"ipc_rpmsg_echo.Debug.map" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/kernel/nortos/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/drivers/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/board/lib" -i"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/lib" --reread_libs --diag_wrap=off --display_error_number --emit_warnings_as_errors --warn_sections --xml_link_info="ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000_linkInfo.xml" --ram_model -o "ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000.out" $(ORDERED_OBJS)
	@echo 'Finished building target: "$@"'
	@echo ' '
	@$(MAKE) --no-print-directory post-build

# Other Targets
clean:
	-$(RM) $(GEN_MISC_FILES__QUOTED)$(GEN_FILES__QUOTED)$(EXE_OUTPUTS__QUOTED)
	-$(RM) "syscfg\ti_dpl_config.obj" "syscfg\ti_drivers_config.obj" "syscfg\ti_drivers_open_close.obj" "syscfg\ti_pinmux_config.obj" "syscfg\ti_power_clock_config.obj" "syscfg\ti_board_config.obj" "syscfg\ti_board_open_close.obj" "syscfg\ti_enet_config.obj" "syscfg\ti_enet_open_close.obj" "syscfg\ti_enet_soc.obj" "syscfg\ti_enet_lwipif.obj" "ipc_rpmsg_echo.obj" "main.obj" 
	-$(RM) "syscfg\ti_dpl_config.d" "syscfg\ti_drivers_config.d" "syscfg\ti_drivers_open_close.d" "syscfg\ti_pinmux_config.d" "syscfg\ti_power_clock_config.d" "syscfg\ti_board_config.d" "syscfg\ti_board_open_close.d" "syscfg\ti_enet_config.d" "syscfg\ti_enet_open_close.d" "syscfg\ti_enet_soc.d" "syscfg\ti_enet_lwipif.d" "ipc_rpmsg_echo.d" "main.d" 
	-$(RMDIR) $(GEN_MISC_DIRS__QUOTED)
	-@echo 'Finished clean'
	-@echo ' '

post-build:
	-$(MAKE) -C C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000 -f makefile_ccs_bootimage_gen OUTNAME=ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000 PROFILE=Debug MCU_PLUS_SDK_PATH=C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04 CG_TOOL_ROOT=C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS CCS_INSTALL_DIR=C:\ti\ccs2030\ccs\ CCS_IDE_MODE=desktop DEVICE=awr294x
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets

