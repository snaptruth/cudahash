/*
 * cudaadapter.cpp
 *
 *  Created on: 2017年7月26日
 */

#include <cuda_runtime.h>
#include "cudasha256.h"

void shutdownCuda(int thr_id){
    cudaDeviceSynchronize();
    cudaDeviceReset();
    cudaThreadExit();
}

int getCudaDeviceNumber(){
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        deviceCount = 0;
    }
    return deviceCount;
}

void getCudaConfigure(int deviceId, int& iBlockNumPerGrid, int& iThreadNumPerBlock){
    cudaDeviceProp tCudaProp;
    CUDA_CHECK(cudaGetDeviceProperties(&tCudaProp, deviceId));

//    int maxGridSizeX = tCudaProp.maxGridSize[0];
//    int multiProcCount = tCudaProp.multiProcessorCount;
//    int sm_code_verion = tCudaProp.major * 100;
//    sm_code_verion += (tCudaProp.minor * 10);
    int iSmCount = tCudaProp.multiProcessorCount;

    // TODO: 计算合适的block和thread数量,现在用比较简单的方式
    iThreadNumPerBlock = 128;
    iBlockNumPerGrid = iSmCount * 8;
}

void createStream(void *pStream){
    if(NULL == pStream){
        return;
    }
    cudaStream_t *pCudaStream = (cudaStream_t *) pStream;

    // TODO: CUDA ERROR handle
    cudaStreamCreate(pCudaStream);
}

void deleteStream(void *pStream){
    if(NULL == pStream){
        return;
    }
    cudaStream_t *pCudaStream = (cudaStream_t *) pStream;

    // TODO: CUDA ERROR handle
    cudaStreamDestroy(*pCudaStream);
}



