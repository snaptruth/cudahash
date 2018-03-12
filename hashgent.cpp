/*
 * hashgent.cpp
 *
 *  Created on: 2017年7月26日
 */

#include <openssl/bn.h>
#include <memory.h>
#include "bnlogif.h"
#include "hashagent.h"

//INIT_MEMBER_LOGGER(HashAgent, "HashAgent");
// 字节序转换
unsigned int reverseEndian(unsigned int value){
    return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
        (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
}

/*
 * 两个作用，一个是补齐hash块到512的整数，第二个是计算512的块数
 * 必须保证pucBuffer长度是512的整数倍！！但uiDataLength是实际数据长度
 */
int HashAgent::FormatHashBlocks(const unsigned char* pucBuffer, unsigned int uiDataLength){
    BN_CHECK_POINTER_NEWCODE(pucBuffer, 0);
    unsigned char* pucData = (unsigned char*)pucBuffer;
    unsigned int uiBlockNumber = 1 + ((uiDataLength + 8) / 64);
    unsigned int* puiPend = (unsigned int *)pucData;
    memset(pucData + uiDataLength, 0, 64 * uiBlockNumber - uiDataLength);
    puiPend[uiDataLength/4] = 0x80000000;

    puiPend[uiBlockNumber * 16 - 1] =  uiDataLength * 8;

    return uiBlockNumber;
}

// 调用者需要保证pucTargetHashValue包含有256bit的空间
int HashAgent::getTargetHashValue(unsigned int uiBits, unsigned char * pucTargetHashValue)
{
    BN_CHECK_POINTER_NEWCODE(pucTargetHashValue, 0);

    unsigned int uiSize = uiBits >> 24;

    memset(pucTargetHashValue, 0, (32) * sizeof(unsigned char));
    if (uiSize >= 1) {
        pucTargetHashValue[uiSize - 1] = (uiBits >> 16) & 0xff;
    }
    if (uiSize >= 2) {
        pucTargetHashValue[uiSize - 2] = (uiBits >> 8) & 0xff;
    }
    if (uiSize >= 3) {
        pucTargetHashValue[uiSize - 3] = (uiBits >> 0) & 0xff;
    }
    return uiSize;
}

// Hash值比较，新hash值小于等于目标值，返回成功
bool HashAgent::compareHashSmaller(unsigned char *pucHashValue, unsigned char *pucTargetHashValue){
    BN_CHECK_POINTER_NEWCODE(pucHashValue, false);
    BN_CHECK_POINTER_NEWCODE(pucTargetHashValue, true);

    for(unsigned int index = 0; index < 32; index++){
        unsigned int iRevertIndex = 31 - index;
        if(pucHashValue[iRevertIndex] < pucTargetHashValue[iRevertIndex]){
            return true;
        }
        if(pucHashValue[iRevertIndex] > pucTargetHashValue[iRevertIndex]){
            return false;
        }
    }

    return true;
}



