################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
build-1301752511: ../example.syscfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: SysConfig'
	"C:/ti/ccs2030/ccs/utils/sysconfig_1.25.0/sysconfig_cli.bat" --script "C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/sbl_null_awr294x-evm_r5fss0-0_nortos_ti-arm-clang/example.syscfg" -o "syscfg" -s "C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/.metadata/product.json" -p "ETS" -r "Default" --context "r5fss0-0" --compiler ticlang
	@echo 'Finished building: "$<"'
	@echo ' '

syscfg/ti_dpl_config.c: build-1301752511 ../example.syscfg
syscfg/ti_dpl_config.h: build-1301752511
syscfg/ti_drivers_config.c: build-1301752511
syscfg/ti_drivers_config.h: build-1301752511
syscfg/ti_drivers_open_close.c: build-1301752511
syscfg/ti_drivers_open_close.h: build-1301752511
syscfg/ti_pinmux_config.c: build-1301752511
syscfg/ti_power_clock_config.c: build-1301752511
syscfg/ti_board_config.c: build-1301752511
syscfg/ti_board_config.h: build-1301752511
syscfg/ti_board_open_close.c: build-1301752511
syscfg/ti_board_open_close.h: build-1301752511
syscfg/ti_enet_config.c: build-1301752511
syscfg/ti_enet_config.h: build-1301752511
syscfg/ti_enet_open_close.c: build-1301752511
syscfg/ti_enet_open_close.h: build-1301752511
syscfg/ti_enet_soc.c: build-1301752511
syscfg/ti_enet_lwipif.c: build-1301752511
syscfg/ti_enet_lwipif.h: build-1301752511
syscfg: build-1301752511

syscfg/%.o: ./syscfg/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-armllvm_4.0.3.LTS/bin/tiarmclang.exe" -c -mcpu=cortex-r5 -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -mthumb -I"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-armllvm_4.0.3.LTS/include/c" -I"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source" -DSOC_AWR294X -D_DEBUG_=1 -g -Wall -Wno-gnu-variable-sized-type-not-at-end -Wno-unused-function -MMD -MP -MF"syscfg/$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/sbl_null_awr294x-evm_r5fss0-0_nortos_ti-arm-clang/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

%.o: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-armllvm_4.0.3.LTS/bin/tiarmclang.exe" -c -mcpu=cortex-r5 -mfloat-abi=hard -mfpu=vfpv3-d16 -mlittle-endian -mthumb -I"C:/ti/ccs2030/ccs/tools/compiler/ti-cgt-armllvm_4.0.3.LTS/include/c" -I"C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mcu_plus_sdk_awr294x_10_01_00_04/source" -DSOC_AWR294X -D_DEBUG_=1 -g -Wall -Wno-gnu-variable-sized-type-not-at-end -Wno-unused-function -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/there/Documents/Capstone/RadarFirmware/ExampleProjects/sbl_null_awr294x-evm_r5fss0-0_nortos_ti-arm-clang/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


