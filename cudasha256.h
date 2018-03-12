/*
 * cudasha256.h
 *
 *  Created on: 2017年7月26日
 */

#ifndef CUDASHA256_H_
#define CUDASHA256_H_

#include <string.h>
#include <assert.h>

#define CUDA_CHECK(cmd) {\
    cudaError_t error = cmd; \
    if(error!=cudaSuccess){\
        assert(true);\
    }\
}

// CUDA简单数组相加测试函数
void vecAddTest(int *piArray1, int *piArray2, int *piArrayDest, int iLength);

// CUDA配置相关函数
int getCudaDeviceNumber();
void getCudaConfigure(int deviceId, int& iBlockNum, int& iThreadNum);
void createStream(void *pStream);
void deleteStream(void *pStream);

// CUDA SHA256D函数实现
void initSha256d(int thr_id, unsigned int uiBlockNum, unsigned int uiThreadNum);
void freeSha256d(int thr_id);
void hashFirstBlock(unsigned int *pdata, unsigned int *ptarget);
void hashLastBlock(int deviceId, unsigned int threadNumber, unsigned int startNonce, unsigned int endNonce, unsigned int *resNonces);

// CUDA SHA256单独函数
void hashSingleBlock(unsigned int threads, unsigned int* in, unsigned int* hashValue);


#endif /* CUDASHA256_H_ */
