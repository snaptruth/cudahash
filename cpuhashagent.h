/*
 * cpuhashagent.h
 *
 *  Created on: 2017年7月26日
 */

#ifndef CPUHASHAGENT_H_
#define CPUHASHAGENT_H_

#include <stddef.h>
#include "bnlogif.h"
#include "hashagent.h"

class CpuHashAgent: public HashAgent {
public:
    CpuHashAgent();
    virtual ~CpuHashAgent();

    virtual ERRCODE getSHA256(const unsigned char *pucSrcData, size_t n, unsigned char *pucHashValue);
    virtual ERRCODE getSHA256D(const unsigned char *pucSrcData, size_t n, unsigned char *pucHashValue);
    virtual ERRCODE findNonce(const unsigned char *pucSrcData,
            size_t n,
            unsigned int uiNonceBegin,
            unsigned int uiNonceEnd,
            unsigned int& uiNonce,
            unsigned char *pucHashValue);
};

#endif /* CPUHASHAGENT_H_ */
