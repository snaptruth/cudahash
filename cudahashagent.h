/*
 * cudahashagent.h
 *
 *  Created on: 2017年7月25日
 */

#ifndef CUDAHASHAGENT_H_
#define CUDAHASHAGENT_H_

#include "cuda.h"
#include "driver_types.h"
#include "hashagent.h"

class CudaHashAgent : public HashAgent{
public:
    ERRCODE addArray(int *piSrcArray1, int *piSrcArray2, int *piDestArray, int iLength);
    virtual ERRCODE getSHA256(const unsigned char *pcSrcData, size_t n, unsigned char *pcHashValue);
    virtual ERRCODE getSHA256D(const unsigned char *pcSrcData, size_t n, unsigned char *pcHashValue);
    virtual ERRCODE findNonce(const unsigned char *pcSrcData,
                size_t n,
                unsigned int uiNonceBegin,
                unsigned int uiNonceEnd,
                unsigned int& uiNonce,
                unsigned char *pucHashValue);

    bool isCudaValid();
    CudaHashAgent();
    virtual ~CudaHashAgent();
private:
    void FindBestConfiguration();

private:
    int m_iBlockPerGrid;
    int m_iThreadPerBlock;
    int m_iDeviceId;        // 多GPU时的GPUID
    int m_iDeviceNumber;    // GPU数量
    int m_iThreadsNumber;   // GPU并行thread数量

    cudaStream_t m_tStreamId;
};

#endif /* CUDAHASHAGENT_H_ */
