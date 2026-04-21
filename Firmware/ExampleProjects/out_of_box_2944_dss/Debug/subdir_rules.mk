################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
%.obj: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: C6000 Compiler'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/bin/cl6x" -mv6600 --abi=eabi -O3 --opt_for_speed=3 --include_path="C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/include" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mathlib_c66x_3_1_2_1/packages" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/dsplib_c66x_3_4_0_0/packages" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/kernel/freertos/FreeRTOS-Kernel/include" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/kernel/freertos/portable/TI_CGT/DSP_C66" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source/kernel/freertos/config/awr294x/c66" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/datapath/dpc/objectdetection/objdethwaDDMA/src/" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_dfp_02_04_17_00" --include_path="C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/out_of_box_2944_dss/dss/dssgenerated" --include_path="C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/out_of_box_2944_dss" --define=SUBSYS_DSS --define=_LITTLE_ENDIAN --define=DebugP_ASSERT_ENABLED --define=INCLUDE_DPM --define=DRIVERS_RADAR_HWA_V2 --define=APP_RESOURCE_FILE='<'ti/demo/awr294x/mmw/mmw_resDDM.h'>' --define=SOC_AWR2944 --define=SOC_AWR294X --define=_DEBUG_=1 -g --gcc --diag_suppress=238 --diag_suppress=880 --diag_suppress=1110 --diag_warning=225 --diag_wrap=off --display_error_number --emit_warnings_as_errors --quiet --gen_func_subsections=on --interrupt_threshold=10 --preproc_with_compile --preproc_dependency="$(basename $(<F)).d_raw" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


