################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cpuhashagent.cpp \
../src/cudaadapter.cpp \
../src/cudahashagent.cpp \
../src/hashgent.cpp 

OBJS += \
./src/cpuhashagent.o \
./src/cudaadapter.o \
./src/cudahashagent.o \
./src/hashgent.o 

CPP_DEPS += \
./src/cpuhashagent.d \
./src/cudaadapter.d \
./src/cudahashagent.d \
./src/hashgent.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__CUDA_ARCH__ -I/usr/include -I/usr/include/mysql -I/usr/local/cuda/include -I"/data/likejun/workspace/cudahash" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


