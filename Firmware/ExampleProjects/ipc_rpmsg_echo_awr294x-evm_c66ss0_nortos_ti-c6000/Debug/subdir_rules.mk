################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
build-1622401777: ../example.syscfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: SysConfig'
	"C:/ti/ccs2030/ccs/utils/sysconfig_1.25.0/sysconfig_cli.bat" --script "C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/ipc_rpmsg_echo_awr294x-evm_r5fss0-0_freertos_ti-arm-clang/example.syscfg" --context "r5fss0-0" --script "C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/ipc_rpmsg_echo_awr294x-evm_r5fss0-1_nortos_ti-arm-clang/example.syscfg" --context "r5fss0-1" --script "C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000/example.syscfg" -o "syscfg" -s "C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/.metadata/product.json" -p "ETS" -r "Default" --context "c66ss0" --compiler ccs
	@echo 'Finished building: "$<"'
	@echo ' '

syscfg/ti_dpl_config.c: build-1622401777 ../example.syscfg
syscfg/ti_dpl_config.h: build-1622401777
syscfg/ti_drivers_config.c: build-1622401777
syscfg/ti_drivers_config.h: build-1622401777
syscfg/ti_drivers_open_close.c: build-1622401777
syscfg/ti_drivers_open_close.h: build-1622401777
syscfg/ti_pinmux_config.c: build-1622401777
syscfg/ti_power_clock_config.c: build-1622401777
syscfg/ti_board_config.c: build-1622401777
syscfg/ti_board_config.h: build-1622401777
syscfg/ti_board_open_close.c: build-1622401777
syscfg/ti_board_open_close.h: build-1622401777
syscfg/ti_enet_config.c: build-1622401777
syscfg/ti_enet_config.h: build-1622401777
syscfg/ti_enet_open_close.c: build-1622401777
syscfg/ti_enet_open_close.h: build-1622401777
syscfg/ti_enet_soc.c: build-1622401777
syscfg/ti_enet_lwipif.c: build-1622401777
syscfg/ti_enet_lwipif.h: build-1622401777
syscfg: build-1622401777

syscfg/%.obj: ./syscfg/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: C6000 Compiler'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/bin/cl6x" -mv6600 --include_path="C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/include" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source" --define=SOC_AWR294X --define=_DEBUG_=1 -g --c99 --diag_warning=225 --diag_wrap=off --display_error_number --emit_warnings_as_errors --quiet --gen_func_subsections=on --assume_control_regs_read --mem_model:const=data --mem_model:data=far_aggregates --disable_push_pop --fp_mode=relaxed --remove_hooks_when_inlining --gen_opt_info=2 --preproc_with_compile --preproc_dependency="syscfg/$(basename $(<F)).d_raw" --include_path="C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000/Debug/syscfg" --obj_directory="syscfg" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

%.obj: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: C6000 Compiler'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/bin/cl6x" -mv6600 --include_path="C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS/include" --include_path="C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source" --define=SOC_AWR294X --define=_DEBUG_=1 -g --c99 --diag_warning=225 --diag_wrap=off --display_error_number --emit_warnings_as_errors --quiet --gen_func_subsections=on --assume_control_regs_read --mem_model:const=data --mem_model:data=far_aggregates --disable_push_pop --fp_mode=relaxed --remove_hooks_when_inlining --gen_opt_info=2 --preproc_with_compile --preproc_dependency="$(basename $(<F)).d_raw" --include_path="C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/ipc_rpmsg_echo_awr294x-evm_c66ss0_nortos_ti-c6000/Debug/syscfg" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


