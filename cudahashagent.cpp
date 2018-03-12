/*
 * cudahashagent.cpp
 *
 *  Created on: 2017年7月25日
 */

#include <memory.h>
#include <malloc.h>
#include "cpuhashagent.h"
#include "cudahashagent.h"
#include "cudasha256.h"

CudaHashAgent::CudaHashAgent():
    m_iBlockPerGrid(0),
    m_iThreadPerBlock(0),
    m_iDeviceId(0),
    m_tStreamId(NULL){
    // TODO: Should set device first when in multi-GPU environment
    m_iDeviceNumber = getCudaDeviceNumber();

    if(0 == m_iDeviceNumber){
        return;
    }

    FindBestConfiguration();

    initSha256d(m_iDeviceId, m_iBlockPerGrid, m_iThreadPerBlock);     // 目前只支持一块GPU
    m_iThreadsNumber = m_iBlockPerGrid * m_iThreadPerBlock;
}

CudaHashAgent::~CudaHashAgent() {
    if(NULL != m_tStreamId){
//        cudaStreamDestroy(m_tStreamId);
    }

    freeSha256d(m_iDeviceId);
}

bool CudaHashAgent::isCudaValid(){
    bool bIsValid = true;
    if(0 == m_iDeviceNumber){
        BN_LOG_ERROR("Can't found CUDA device on host, device number:%d, Please check GPU's run state",
                m_iDeviceNumber);
        bIsValid = false;
    }
    return bIsValid;
}

ERRCODE CudaHashAgent::addArray(int *piSrcArray1, int *piSrcArray2, int *piDestArray, int iLength) {
    BN_CHECK_POINTER(piSrcArray1);
    BN_CHECK_POINTER(piSrcArray2);
    BN_CHECK_POINTER(piDestArray);

    vecAddTest(piSrcArray1, piSrcArray2, piDestArray, iLength);

    return SUCCESS;
}

ERRCODE CudaHashAgent::getSHA256(const unsigned char *pucSrcData,
        size_t length,
        unsigned char *pucHashValue) {
    BN_CHECK_CONDITION_RET((!isCudaValid()), ERROR_GPUNOTEXIST);
    BN_CHECK_POINTER(pucSrcData);
    BN_CHECK_POINTER(pucHashValue);
    BN_CHECK_CONDITION_RET((length % 4 != 0), ERROR_PARAM);
    BN_CHECK_CONDITION_RET((length == 0), ERROR_PARAM);
    int iRealLength = length;
    unsigned char *pucSrcBlockData = NULL;

    const unsigned int *puiSrcData = (const unsigned int *)pucSrcData;
    BN_LOG_INTERFACE("puiSrcData[1]:0x%08x, puiSrcData[9]:0x%08x, puiSrcData[17]:0x%08x",\
            puiSrcData[1], puiSrcData[9], puiSrcData[17]);

    // 跟hash算法相关，hash的block数据，如果不足512长度的，需要补齐512
    int iBufferLength = 64 + ((length + 8) / 64) * 64;
    pucSrcBlockData = (unsigned char *)malloc(iBufferLength * sizeof(unsigned char));
    BN_CHECK_POINTER(pucSrcBlockData);
    memset(pucSrcBlockData, 0, iBufferLength * sizeof(unsigned char));
    memcpy(pucSrcBlockData, pucSrcData, iRealLength * sizeof(unsigned char));
    int iBlockNum = FormatHashBlocks(pucSrcBlockData, length);

    unsigned int *puiSrcBlockData = (unsigned int *)pucSrcBlockData;
    for(int i = 0; i < length / 4; i++){
        puiSrcBlockData[i] = reverseEndian(puiSrcBlockData[i]);
    }

    memcpy(pucHashValue, sha256BeginHash, sizeof(sha256BeginHash));
    for(int index = 0; index < iBlockNum; index++){
        unsigned int *puiSrcData = (unsigned int *)(pucSrcBlockData + index * 64);
        hashSingleBlock(1, puiSrcData, (unsigned int *)pucHashValue);
        if((iBlockNum > 1) && (index != iBlockNum - 1)){
            unsigned int *puiHashValue = (unsigned int *) pucHashValue;
            // 多区块的hash时，中间的hash结果需要转换回小端方式传入CUDA
            for(int hashIndex = 0; hashIndex < 8; hashIndex++){
                puiHashValue[hashIndex] = reverseEndian(puiHashValue[hashIndex]);
            }
        }
    }
    free(pucSrcBlockData);
    return SUCCESS;
}

ERRCODE CudaHashAgent::getSHA256D(const unsigned char *pucSrcData,
        size_t length,
        unsigned char *pucHashValue){
    BN_CHECK_CONDITION_RET((!isCudaValid()), ERROR_GPUNOTEXIST);
    BN_CHECK_POINTER(pucSrcData);
    BN_CHECK_POINTER(pucHashValue);

    const unsigned int *puiSrcData = (const unsigned int *)pucSrcData;
    BN_LOG_INTERFACE("puiSrcData[1]:0x%08x, puiSrcData[9]:0x%08x, puiSrcData[17]:0x%08x",\
            puiSrcData[1], puiSrcData[9], puiSrcData[17]);

    unsigned char aucRound1HashValue[32] = {0};
    getSHA256(pucSrcData, length, aucRound1HashValue);
    return getSHA256(aucRound1HashValue, 32, pucHashValue);
}

ERRCODE CudaHashAgent::findNonce(const unsigned char *pucSrcData,
            size_t length,
            unsigned int uiNonceBegin,
            unsigned int uiNonceEnd,
            unsigned int& uiNonce,
            unsigned char *pucHashValue){
    BN_CHECK_CONDITION_RET((!isCudaValid()), ERROR_GPUNOTEXIST);
    BN_CHECK_POINTER(pucSrcData);
    BN_CHECK_CONDITION_RET((length == 0), ERROR_PARAM);
    BN_CHECK_CONDITION_RET((length % 4 != 0), ERROR_PARAM);
    BN_CHECK_CONDITION_RET((uiNonceEnd < uiNonceBegin), ERROR_PARAM);

    unsigned int *puiSrcData = (unsigned int *)pucSrcData;
    BN_LOG_INTERFACE("prehash[0]:0x%08x, merkle[0]:0x%08x, time:0x%08x, bits:0x%08x, \
uiNonceBegin:0x%x, uiNonceEnd:0x%x",\
            puiSrcData[1], puiSrcData[9], puiSrcData[17], puiSrcData[18], uiNonceBegin, uiNonceEnd);

    ERRCODE retValue = ERROR_NOTFOUND;

    // 获取hash目标值
    const unsigned int *puiBits = (const unsigned int *) pucSrcData;
    unsigned char aucHashTargetValue[32] = {0};
    getTargetHashValue(puiBits[18], aucHashTargetValue);

    unsigned int *puiTarget = (unsigned int *)aucHashTargetValue;
    // Hash first block, in CPU.
    hashFirstBlock(puiSrcData, puiTarget);

    // Hash the last block, in GPU parallel
    unsigned int auiNonceResult[10] = {0};

    int iTotalLoopTime = ((uiNonceEnd - uiNonceBegin) / m_iThreadsNumber) + 1;
    unsigned int iThreadNumbers = m_iThreadsNumber;

    if(uiNonceBegin + m_iThreadsNumber - 1 < uiNonceBegin){
        iThreadNumbers = uiNonceEnd - uiNonceBegin + 1;
    }else if(uiNonceBegin + m_iThreadsNumber - 1 > uiNonceEnd){
        iThreadNumbers = uiNonceEnd - uiNonceBegin + 1;
    }

    memset(auiNonceResult, 0, sizeof(auiNonceResult));
    hashLastBlock(m_iDeviceId, iThreadNumbers, uiNonceBegin, uiNonceEnd, auiNonceResult);

    if(auiNonceResult[0] != BN_UINT32_MAX){
        uiNonce = (auiNonceResult[0]);
        CpuHashAgent *pCpuHashAgent = new CpuHashAgent();
        BN_CHECK_POINTER(pCpuHashAgent);

        puiSrcData[19] = uiNonce;
        pCpuHashAgent->getSHA256D(pucSrcData, length, pucHashValue);
        retValue = SUCCESS;
        BN_LOG_INTERFACE("Found Nonce:0x%08x", uiNonce);
        delete pCpuHashAgent;
    }else{
        BN_LOG_INTERFACE("Nonce Not Found for 0x%08x - 0x%08x!", uiNonceBegin, uiNonceEnd);
    }

    return retValue;
}

void CudaHashAgent::FindBestConfiguration(){
    getCudaConfigure(m_iDeviceId, m_iBlockPerGrid, m_iThreadPerBlock);
    m_iThreadsNumber = m_iBlockPerGrid * m_iThreadPerBlock;
}
