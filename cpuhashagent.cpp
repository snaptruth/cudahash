/*
 * cpuhashagent.cpp
 *
 *  Created on: 2017年7月26日
 */

#include <openssl/sha.h>
#include <memory.h>
#include <malloc.h>
#include "cpuhashagent.h"

CpuHashAgent::CpuHashAgent() {
    // TODO Auto-generated constructor stub

}

CpuHashAgent::~CpuHashAgent() {
    // TODO Auto-generated destructor stub
}

ERRCODE CpuHashAgent::getSHA256(const unsigned char *pucSrcData,
        size_t length,
        unsigned char *pucHashValue){
    BN_CHECK_POINTER(pucSrcData);
    BN_CHECK_POINTER(pucHashValue);

    const unsigned int *puiSrcData = (const unsigned int *)pucSrcData;
    BN_LOG_INTERFACE("puiSrcData[1]:0x%08x, puiSrcData[9]:0x%08x, puiSrcData[17]:0x%08x",\
            puiSrcData[1], puiSrcData[9], puiSrcData[17]);

    unsigned char * pucRetValue = SHA256(pucSrcData, length, pucHashValue);
    BN_CHECK_POINTER(pucRetValue);
    return SUCCESS;
}

ERRCODE CpuHashAgent::getSHA256D(const unsigned char *pucSrcData,
        size_t length,
        unsigned char *pucHashValue){
    BN_CHECK_POINTER(pucSrcData);
    BN_CHECK_POINTER(pucHashValue);

    const unsigned int *puiSrcData = (const unsigned int *)pucSrcData;
    BN_LOG_INTERFACE("puiSrcData[1]:0x%08x, puiSrcData[9]:0x%08x, puiSrcData[17]:0x%08x",\
            puiSrcData[1], puiSrcData[9], puiSrcData[17]);

    unsigned char aucHashTempValue[32] = {0};

    unsigned char * pucRetValue = SHA256(pucSrcData, length, aucHashTempValue);
    BN_CHECK_POINTER(pucRetValue);

    pucRetValue = SHA256(aucHashTempValue, 32, pucHashValue);
    BN_CHECK_POINTER(pucRetValue);

    return SUCCESS;
}

ERRCODE CpuHashAgent::findNonce(const unsigned char *pucSrcData,
        size_t length,
        unsigned int uiNonceBegin,
        unsigned int uiNonceEnd,
        unsigned int& uiNonce,
        unsigned char *pucHashValue){
    BN_CHECK_POINTER(pucSrcData);
    BN_CHECK_POINTER(pucHashValue);
    BN_CHECK_CONDITION_RET((length == 0), ERROR_PARAM);

    ERRCODE returnValue = ERROR_NOTFOUND;
    unsigned char aucTempHash[32] = {0};

    const unsigned int *puiSrcData = (const unsigned int *)pucSrcData;
    BN_LOG_INTERFACE("puiSrcData[1]:0x%08x, puiSrcData[9]:0x%08x, puiSrcData[17]:0x%08x, \
uiNonceBegin:0x%x, uiNonceEnd:0x%x",\
            puiSrcData[1], puiSrcData[9], puiSrcData[17], uiNonceBegin, uiNonceEnd);

    // 获取hash目标值
    const unsigned int *puiBits = (const unsigned int *) pucSrcData;
    unsigned char aucHashTargetValue[32] = {0};
    getTargetHashValue(puiBits[18], aucHashTargetValue);

    uiNonce = uiNonceBegin;

    // Search, Performance TO BE improving
    for(uiNonce = uiNonceBegin; uiNonce <= uiNonceEnd; uiNonce++){
        // Change the nonce value at beginning
        unsigned int *puiSrcBlockData = (unsigned int *)pucSrcData;
        puiSrcBlockData[19]= uiNonce;

        // Hash Round 1
        unsigned char *pucRetValue = SHA256(pucSrcData, length, aucTempHash);
        BN_CHECK_POINTER_BREAK(pucRetValue);

        // Hash Round 2
        pucRetValue = SHA256(aucTempHash, 32, pucHashValue);
        BN_CHECK_POINTER_BREAK(pucRetValue);

        if (compareHashSmaller(pucHashValue, aucHashTargetValue)){
            returnValue = SUCCESS;
            break;
        }
    }

    return returnValue;
}


