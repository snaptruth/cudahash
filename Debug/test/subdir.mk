################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../test/hashagenttest.cpp \
../test/testmain.cpp 

OBJS += \
./test/hashagenttest.o \
./test/testmain.o 

CPP_DEPS += \
./test/hashagenttest.d \
./test/testmain.d 


# Each subdirectory must supply rules for building sources it contributes
test/%.o: ../test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__CUDA_ARCH__ -I/usr/include -I/usr/include/mysql -I/usr/local/cuda/include -I"/data/likejun/workspace/cudahash" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


