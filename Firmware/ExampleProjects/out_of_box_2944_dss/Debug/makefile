################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

CG_TOOL_ROOT := C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS

GEN_OPTS__FLAG := 
GEN_CMDS__FLAG := 

ORDERED_OBJS += \
"./objectdetection.obj" \
"./dss/data_path.obj" \
"./dss/dssgenerated/ti_dpl_config.obj" \
"./dss/dssgenerated/ti_drivers_config.obj" \
"./dss/dssgenerated/ti_drivers_open_close.obj" \
"./dss/dssgenerated/ti_pinmux_config.obj" \
"./dss/dssgenerated/ti_power_clock_config.obj" \
"./dss/dssgenerated/ti_board_config.obj" \
"./dss/dssgenerated/ti_board_open_close.obj" \
"./dss/dssgenerated/ti_enet_config.obj" \
"./dss/dssgenerated/ti_enet_open_close.obj" \
"./dss/dssgenerated/ti_enet_soc.obj" \
"./dss/dssgenerated/ti_enet_lwipif.obj" \
"./dss/dss_main.obj" \
"../c66x_linker.cmd" \
"../dss/mmw_dss_linker.cmd" \
$(GEN_CMDS__FLAG) \
-llibdpm_awr2944.ae66 \
-llibmathutils.ae66 \
-llibrangeproc_hwa_ddma_awr2944.ae66 \
-llibdopplerproc_hwa_ddma_awr2944.ae66 \
-llibdpedma_hwa_awr2944.ae66 \
-lmathlib.ae66 \
-ldsplib.ae66 \
-llibrangecfarproc_hwa_ddma_awr2944.ae66 \
-lfreertos.awr294x.c66.ti-c6000.release.lib \
-ldrivers.awr294x.c66.ti-c6000.release.lib \
-lboard.awr294x.c66.ti-c6000.release.lib \
-llibc.a \

-include ../makefile.init

RM := DEL /F
RMDIR := RMDIR /S/Q

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir_vars.mk
-include dss/subdir_vars.mk
-include subdir_rules.mk
-include dss/subdir_rules.mk
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
out_of_box_2944_dss.xe66 

EXE_OUTPUTS__QUOTED += \
"out_of_box_2944_dss.xe66" 


# All Target
all:
	@$(MAKE) --no-print-directory pre-build
	@$(MAKE) --no-print-directory main-build

# Main-build Target
main-build: $(OBJS) $(CMD_SRCS) $(GEN_CMDS)
	@$(MAKE) --no-print-directory -Onone "out_of_box_2944_dss.xe66"

# Tool invocations
out_of_box_2944_dss.xe66: $(OBJS) $(CMD_SRCS) $(GEN_CMDS)
	@echo 'Building target: "$@"'
	@echo 'Invoking: C6000 Linker'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/bin/cl6x" -mv6600 --abi=eabi -O3 --opt_for_speed=3 --define=SUBSYS_DSS --define=_LITTLE_ENDIAN --define=DebugP_ASSERT_ENABLED --define=INCLUDE_DPM --define=DRIVERS_RADAR_HWA_V2 --define=APP_RESOURCE_FILE='<'ti/demo/awr294x/mmw/mmw_resDDM.h'>' --define=SOC_AWR2944 --define=SOC_AWR294X --define=_DEBUG_=1 -g --gcc --diag_suppress=238 --diag_suppress=880 --diag_suppress=1110 --diag_warning=225 --diag_wrap=off --display_error_number --emit_warnings_as_errors --quiet --gen_func_subsections=on --interrupt_threshold=10 -z -m"out_of_box_2944_dss.Debug.map" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/board/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/kernel/freertos/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/drivers/lib" -i"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/dpm/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/utils/mathutils/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mathlib_c66x_3_1_2_1/packages/ti/mathlib/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/dsplib_c66x_3_4_0_0/packages/ti/dsplib/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/datapath/dpu/rangecfarprocDDMA/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/datapath/dpu/rangeprocDDMA/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/datapath/dpu/dopplerprocDDMA/lib" -i"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/datapath/dpedma/lib" -i"C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/out_of_box_2944_dss" -i"C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/out_of_box_2944_dss/dss/dssgenerated" --reread_libs --diag_error=10015 --diag_warning=225 --diag_wrap=off --display_error_number --emit_warnings_as_errors --warn_sections --xml_link_info="out_of_box_2944_dss_linkInfo.xml" --rom_model -o "out_of_box_2944_dss.xe66" $(ORDERED_OBJS)
	@echo 'Finished building target: "$@"'
	@echo ' '
	@$(MAKE) --no-print-directory post-build

# Other Targets
clean:
	-$(RM) $(GEN_MISC_FILES__QUOTED)$(GEN_FILES__QUOTED)$(EXE_OUTPUTS__QUOTED)
	-$(RM) "objectdetection.obj" "dss\data_path.obj" "dss\dssgenerated\ti_dpl_config.obj" "dss\dssgenerated\ti_drivers_config.obj" "dss\dssgenerated\ti_drivers_open_close.obj" "dss\dssgenerated\ti_pinmux_config.obj" "dss\dssgenerated\ti_power_clock_config.obj" "dss\dssgenerated\ti_board_config.obj" "dss\dssgenerated\ti_board_open_close.obj" "dss\dssgenerated\ti_enet_config.obj" "dss\dssgenerated\ti_enet_open_close.obj" "dss\dssgenerated\ti_enet_soc.obj" "dss\dssgenerated\ti_enet_lwipif.obj" "dss\dss_main.obj" 
	-$(RM) "objectdetection.d" "dss\data_path.d" "dss\dssgenerated\ti_dpl_config.d" "dss\dssgenerated\ti_drivers_config.d" "dss\dssgenerated\ti_drivers_open_close.d" "dss\dssgenerated\ti_pinmux_config.d" "dss\dssgenerated\ti_power_clock_config.d" "dss\dssgenerated\ti_board_config.d" "dss\dssgenerated\ti_board_open_close.d" "dss\dssgenerated\ti_enet_config.d" "dss\dssgenerated\ti_enet_open_close.d" "dss\dssgenerated\ti_enet_soc.d" "dss\dssgenerated\ti_enet_lwipif.d" "dss\dss_main.d" 
	-@echo 'Finished clean'
	-@echo ' '

pre-build:
	-C:/ti/ccs2030/ccs//utils/cygwin/rm -f C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss.rprc
	-C:/ti/ccs2030/ccs/utils/sysconfig_1.25.0/nodejs/node C:/ti/ccs2030/ccs/utils/sysconfig_1.25.0/dist/cli.js --product C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/.metadata/product.json --script ../../out_of_box_2944_mss/mss/mss.syscfg --context r5fss0-0 --output ../../out_of_box_2944_mss/mss/mssgenerated/ --script ../dss/dss.syscfg --context c66ss0 --output ../dss/dssgenerated/
	-@echo ' '

post-build:
	-C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/bin/strip6x -p C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss.xe66 -o=C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss_strip.xe66
	-C:/ti/ccs2030/ccs/utils/sysconfig_1.25.0/nodejs/node.exe  C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/tools/boot/out2rprc/elf2rprc.js C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss_strip.xe66
	-C:/ti/ccs2030/ccs//utils/cygwin/mv C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss_strip.rprc C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss.rprc
	-C:/ti/ccs2030/ccs//utils/cygwin/rm -f C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss_strip.rprc
	-C:/ti/ccs2030/ccs//utils/cygwin/rm -f C:\Users\there\Documents\Capstone\RadarFirmware\ExampleProjects\out_of_box_2944_dss/Debug/out_of_box_2944_dss_strip.xe66
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets

