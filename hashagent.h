/*
 * hashagent.h
 *
 *  Created on: 2017年7月26日
 */

#ifndef HASHAGENT_H_
#define HASHAGENT_H_

#include "include/bnconst.h"
//#include "include/logging4.h"

const unsigned int  sha256BeginHash[8] = {
    0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
    0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};

unsigned int reverseEndian(unsigned int value);

class HashAgent {
public:
    HashAgent() {};
    virtual ~HashAgent() {};

    virtual ERRCODE getSHA256(const unsigned char *pcSrcData, size_t n, unsigned char *pcHashValue) = 0;
    virtual ERRCODE getSHA256D(const unsigned char *pcSrcData, size_t n, unsigned char *pcHashValue) = 0;
    virtual ERRCODE findNonce(const unsigned char *pcSrcData,
            size_t n,
            unsigned int uiNonceBegin,
            unsigned int uiNonceEnd,
            unsigned int& uiNonce,
            unsigned char *pucHashValue) = 0;

    int FormatHashBlocks(const unsigned char* pucBuffer, unsigned int uiBufferLength);
    int getTargetHashValue(unsigned int uiBits, unsigned char * pucTargetHashValue);
    bool compareHashSmaller(unsigned char *pucHashValue, unsigned char *pucTargetHashValue);

public:
//    MEMBER_LOGGER;

};

#endif /* HASHAGENT_H_ */
