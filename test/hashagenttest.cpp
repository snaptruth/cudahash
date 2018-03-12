/*
 * cudahashagent_test.cpp
 *
 *  Created on: 2017年7月25日
 */

#include <assert.h>
#include <cpuhashagent.h>
#include <cudahashagent.h>
#include <memory.h>
#include <string.h>
#include <test/hashagenttest.h>
#include "cppunit/TestAssert.h"
#include "sys/time.h"


CPPUNIT_TEST_SUITE_REGISTRATION(HashTest);
void HashTest::setUp(){
    if(NULL != pCudaHashAgent){
        delete pCudaHashAgent;
        pCudaHashAgent = NULL;
    }
    if(NULL != pCpuHashAgent){
        delete pCpuHashAgent;
        pCpuHashAgent = NULL;
    }

    pCudaHashAgent = new CudaHashAgent();
    pCpuHashAgent = new CpuHashAgent();

    if(NULL == pCudaHashAgent){
        assert(true);
    }
    if(NULL == pCpuHashAgent){
        assert(true);
    }
}

void HashTest::tearDown(){
    if(NULL != pCudaHashAgent){
        delete pCudaHashAgent;
        pCudaHashAgent = NULL;
    }
    if(NULL != pCpuHashAgent){
        delete pCpuHashAgent;
        pCpuHashAgent = NULL;
    }
}

void HashTest::testAdd(){
    int *piSrcArray1 = (int *)malloc(10 * sizeof(int));
    int *piSrcArray2 = (int *)malloc(10 * sizeof(int));
    int *piDestArray = (int *)malloc(10 * sizeof(int));

    for(unsigned int index = 0; index < 10; index++){
        piSrcArray1[index] = index;
        piSrcArray2[index] = 10 - index;
    }
    memset(piDestArray, 0, 10 * sizeof(int));

    pCudaHashAgent->addArray(piSrcArray1, piSrcArray2, piDestArray, 10);

    for(unsigned int index = 0; index < 10; index++){
        CPPUNIT_ASSERT_EQUAL(10, piDestArray[index]);
    }

    free(piSrcArray1);
    free(piSrcArray2);
    free(piDestArray);
}

void HashTest::createGenesisBlock(unsigned int *puiBlock){
    // Genesis Block:
    // GetHash()      = 0x000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
    // hashMerkleRoot = 0x4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b
    // txNew.vin[0].scriptSig     = 486604799 4 0x736B6E616220726F662074756F6C69616220646E6F63657320666F206B6E697262206E6F20726F6C6C65636E61684320393030322F6E614A2F33302073656D695420656854
    // txNew.vout[0].nValue       = 5000000000
    // txNew.vout[0].scriptPubKey = 0x5F1DF16B2B704C8A578D0BBAF74D385CDE12C11EE50455F3C438EF4C3FBCF649B6DE611FEAE06279A60939E028A8D65C10B73071A6F16719274855FEB0FD8A6704 OP_CHECKSIG
    // block.nVersion = 1
    // block.nTime    = 1231006505
    // block.nBits    = 0x1d00ffff
    // block.nNonce   = 2083236893
    // CBlock(hash=000000000019d6, ver=1, hashPrevBlock=00000000000000, hashMerkleRoot=4a5e1e, nTime=1231006505, nBits=1d00ffff, nNonce=2083236893, vtx=1)
    //   CTransaction(hash=4a5e1e, ver=1, vin.size=1, vout.size=1, nLockTime=0)
    //     CTxIn(COutPoint(000000, -1), coinbase 04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73)
    //     CTxOut(nValue=50.00000000, scriptPubKey=0x5F1DF16B2B704C8A578D0B)
    //   vMerkleTree: 4a5e1e
    unsigned char arrMerkleRoot[32] = { 0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2, \
                                        0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61, \
                                        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32, \
                                        0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a};
    if(NULL == puiBlock){
        return;
    }
    puiBlock[0] = 1;    // version
                        // previous block hash
    memcpy(puiBlock + 9, arrMerkleRoot, 32 * sizeof(unsigned char)); // merkle root
    puiBlock[17] = 1231006505; // time
    puiBlock[18] = 0x1d00ffff; // bits
    puiBlock[19] = 2083236893; // nonce

}

// 测试abcdefghijklmnopqrstuvwxyz的hash值
void HashTest::testCpuSha256abc(){
    unsigned char *pucSrcData = (unsigned char *)malloc(26 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x71, 0xC4, 0x80, 0xDF, 0x93, 0xD6, 0xAE, 0x2F, \
                                        0x1E, 0xFA, 0xD1, 0x44, 0x7C, 0x66, 0xC9, 0x52, \
                                        0x5E, 0x31, 0x62, 0x18, 0xCF, 0x51, 0xFC, 0x8D, \
                                        0x9E, 0xD8, 0x32, 0xF2, 0xDA, 0xF1, 0x8B, 0x73};

    for(unsigned int index = 0; index < 26; index++){
        pucSrcData[index] = 0x61 + index;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCpuHashAgent->getSHA256(pucSrcData, 26, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试abcdefghijklmnopqrstuvwxyz的hash值
void HashTest::testCudaSha256abc(){
    unsigned char *pucSrcData = (unsigned char *)malloc(24 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x93, 0xb0, 0xca, 0xbf, 0x86, 0x68, 0xe0, 0xc5, \
                                        0x34, 0xc5, 0x2a, 0x56, 0x89, 0x57, 0x49, 0x9e, \
                                        0x12, 0xa2, 0x84, 0xf5, 0x9d, 0x97, 0xdc, 0x9b, \
                                        0x27, 0x25, 0xef, 0x83, 0x68, 0x04, 0x87, 0x5b};

    memset(pucSrcData, 0, 24 * sizeof(unsigned char));
    memset(pucDestData, 0, 32 * sizeof(unsigned char));
    for(unsigned int index = 0; index < 24; index++){
        pucSrcData[index] = 0x61 + index;
    }

    pCudaHashAgent->getSHA256(pucSrcData, 24, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试ABCDEFGHIJKLMNOPQRSTUVWXYZ的hash值
void HashTest::testCpuSha256ABC(){
    unsigned char *pucSrcData = (unsigned char *)malloc(26 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0xD6, 0xEC, 0x68, 0x98, 0xDE, 0x87, 0xDD, 0xAC, \
                                        0x6E, 0x5B, 0x36, 0x11, 0x70, 0x8A, 0x7A, 0xA1, \
                                        0xC2, 0xD2, 0x98, 0x29, 0x33, 0x49, 0xCC, 0x1A, \
                                        0x6C, 0x29, 0x9A, 0x1D, 0xB7, 0x14, 0x9D, 0x38};

    for(unsigned int index = 0; index < 26; index++){
        pucSrcData[index] = 0x41 + index;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCpuHashAgent->getSHA256(pucSrcData, 26, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试ABCDEFGHIJKLMNOPQRSTUVWX的hash值
void HashTest::testCudaSha256ABC(){
    unsigned char *pucSrcData = (unsigned char *)malloc(24 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0xbf, 0xa1, 0xd0, 0xdf, 0xce, 0xda, 0xe3, 0x14, \
                                        0xea, 0x4f, 0x7b, 0x7b, 0x31, 0x53, 0x21, 0x6e, \
                                        0x43, 0x53, 0x72, 0xc9, 0xaa, 0x0c, 0x33, 0x1a, \
                                        0xec, 0x04, 0xe3, 0xf0, 0xfb, 0x8b, 0xae, 0x12};


    for(unsigned int index = 0; index < 24; index++){
        pucSrcData[index] = 0x41 + index;
    }

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCudaHashAgent->getSHA256(pucSrcData, 24, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试0123456789的hash值
void HashTest::testCpuSha256123(){
    unsigned char *pucSrcData = (unsigned char *)malloc(10 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x84, 0xD8, 0x98, 0x77, 0xF0, 0xD4, 0x04, 0x1E, \
                                        0xFB, 0x6B, 0xF9, 0x1A, 0x16, 0xF0, 0x24, 0x8F, \
                                        0x2F, 0xD5, 0x73, 0xE6, 0xAF, 0x05, 0xC1, 0x9F, \
                                        0x96, 0xBE, 0xDB, 0x9F, 0x88, 0x2F, 0x78, 0x82};

    for(unsigned int index = 0; index < 10; index++){
        pucSrcData[index] = 0x30 + index;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCpuHashAgent->getSHA256(pucSrcData, 10, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试1234567890的hash值
void HashTest::testCudaSha256123(){
    unsigned char *pucSrcData = (unsigned char *)malloc(12 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x5c, 0xe9, 0xab, 0x10, 0xea, 0x74, 0x27, 0x0d, \
                                        0x12, 0x62, 0x0d, 0xfa, 0xcd, 0x74, 0xd2, 0x62, \
                                        0xc6, 0x41, 0x1e, 0x20, 0x76, 0x1e, 0x45, 0x9b, \
                                        0xb1, 0xb2, 0x65, 0xde, 0x88, 0x34, 0x22, 0xac};

    for(unsigned int index = 0; index < 10; index++){
        pucSrcData[index] = 0x30 + index;
    }
    pucSrcData[10] = 0x30;
    pucSrcData[11] = 0x31;

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCudaHashAgent->getSHA256(pucSrcData, 12, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试双BLOCK的hash值
void HashTest::testCpuSha256DoubleBlock(){
    unsigned char *pucSrcData = (unsigned char *)malloc(80 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0xd9, 0xb1, 0xf3, 0xe2, 0xc6, 0xd5, 0x28, 0x66, \
                                        0x8a, 0x73, 0xf2, 0x25, 0x75, 0xc4, 0x4e, 0xd9, \
                                        0xf9, 0x8d, 0x9c, 0x68, 0x49, 0x64, 0x76, 0x1b, \
                                        0x62, 0x14, 0x17, 0xef, 0xd8, 0x0d, 0x7a, 0x60};

    for(unsigned int index = 0; index < 80; index++){
        pucSrcData[index] = 0x41;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCpuHashAgent->getSHA256(pucSrcData, 80, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试双BLOCK的hash值
void HashTest::testCudaSha256DoubleBlock(){
    unsigned char *pucSrcData = (unsigned char *)malloc(80 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0xd9, 0xb1, 0xf3, 0xe2, 0xc6, 0xd5, 0x28, 0x66, \
                                        0x8a, 0x73, 0xf2, 0x25, 0x75, 0xc4, 0x4e, 0xd9, \
                                        0xf9, 0x8d, 0x9c, 0x68, 0x49, 0x64, 0x76, 0x1b, \
                                        0x62, 0x14, 0x17, 0xef, 0xd8, 0x0d, 0x7a, 0x60};

    for(unsigned int index = 0; index < 80; index++){
        pucSrcData[index] = 0x41;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCudaHashAgent->getSHA256(pucSrcData, 80, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}


// 测试随机字符串的hash值
void HashTest::testCpuSha256RandomLong(){
    unsigned char acSrcData[] = "Bitcoin is a collection of concepts and technologies that form the basis of a digital money ecosystem. Units of currency called bitcoin are used to store and transmit value among participants in the bitcoin network. Bitcoin users communicate with each other using the bitcoin protocol primarily via the internet, although other transport networks can also be used. The bitcoin protocol stack, available as open source software, can be run on a wide range of computing devices, including laptops and smartphones, making the technology easily accessible.";
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(acSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0xB2, 0xF7, 0xE2, 0x2B, 0xF3, 0xB5, 0x2E, 0x76, \
                                        0x7D, 0xB9, 0x77, 0x15, 0x66, 0xE8, 0xEE, 0xDB, \
                                        0x82, 0xFE, 0xF4, 0x81, 0x39, 0x4A, 0x4A, 0x92, \
                                        0x82, 0xB4, 0x8D, 0x22, 0x2A, 0xD8, 0xC7, 0xC8};

    // 需要去掉字符串结束的\0，不然不知道hash值是多少……
    for(unsigned int index = 0; index < (sizeof(acSrcData) - 1); index++){
        pucSrcData[index] = acSrcData[index];
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCpuHashAgent->getSHA256(pucSrcData, (sizeof(acSrcData) - 1), pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试随机字符串的hash值
void HashTest::testCudaSha256RandomLong(){
    unsigned char acSrcData[] = "Bitcoin is a collection of concepts and technologies that form the basis of a digital money ecosystem. Units of currency called bitcoin are used to store and transmit value among participants in the bitcoin network. Bitcoin users communicate with each other using the bitcoin protocol primarily via the internet, although other transport networks can also be used. The bitcoin protocol stack, available as open source software, can be run on a wide range of computing devices, including laptops and smartphones, making the technology easily accessible.";
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(acSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0xB2, 0xF7, 0xE2, 0x2B, 0xF3, 0xB5, 0x2E, 0x76, \
                                        0x7D, 0xB9, 0x77, 0x15, 0x66, 0xE8, 0xEE, 0xDB, \
                                        0x82, 0xFE, 0xF4, 0x81, 0x39, 0x4A, 0x4A, 0x92, \
                                        0x82, 0xB4, 0x8D, 0x22, 0x2A, 0xD8, 0xC7, 0xC8};

    // 需要去掉字符串结束的\0，不然不知道正确hash值是多少……
    for(unsigned int index = 0; index < (sizeof(acSrcData) - 1); index++){
        pucSrcData[index] = acSrcData[index];
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCudaHashAgent->getSHA256(pucSrcData, (sizeof(acSrcData) - 1), pucDestData);


    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试比特币创世区块的hash值
void HashTest::testCpuSha256GenesisBlock(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    // 初始化创世区块
    createGenesisBlock(auiSrcData);

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    pCpuHashAgent->getSHA256D(pucSrcData, 80, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[31 - index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试比特币创世区块的hash值
void HashTest::testCudaSha256GenesisBlock(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    // 初始化创世区块
    createGenesisBlock(auiSrcData);

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    pCudaHashAgent->getSHA256D(pucSrcData, 80, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[31 - index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值
void HashTest::testCpuSha256FoundNonce(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};
    unsigned int uiFinalNonce = 0;
    unsigned int uiRightNonce = 2083236893;

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    // 初始化创世区块
    createGenesisBlock(auiSrcData);

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    pCpuHashAgent->findNonce(pucSrcData, 80, 2083236800, 2083237900, uiFinalNonce, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[31 - index], arrRightAnswer[index]);
    }

    CPPUNIT_ASSERT_EQUAL(uiFinalNonce, uiRightNonce);

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值
void HashTest::testCudaSha256FoundNonce(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};
    unsigned int uiFinalNonce = 0;
    unsigned int uiRightNonce = 2083236893;

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    // 初始化创世区块
    createGenesisBlock(auiSrcData);

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    pCudaHashAgent->findNonce(pucSrcData, 80, 2083236800, 2083237900, uiFinalNonce, pucDestData);

    // 不建议做内文测试，因为为了效率，findnonce算出来的可能不是精准值（目前应圣贤的要求，做成了精准值）
    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[31 - index], arrRightAnswer[index]);
    }

    CPPUNIT_ASSERT_EQUAL(uiFinalNonce, uiRightNonce);

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值
void HashTest::testCpuSha256D80(){
    unsigned char *pucSrcData = (unsigned char *)malloc(80 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x77, 0x67, 0x33, 0x11, 0x4f, 0xd0, 0x5e, 0x8c, \
                                        0x7a, 0x67, 0x7c, 0x7f, 0xb4, 0xf5, 0x36, 0x6a, \
                                        0xf5, 0x3a, 0xc8, 0x92, 0x87, 0x53, 0xf7, 0x17, \
                                        0xef, 0xdd, 0x76, 0x1a, 0xbb, 0xb4, 0x84, 0xfa};

    for(unsigned int index = 0; index < 80; index++){
        pucSrcData[index] = 0x44;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCpuHashAgent->getSHA256(pucSrcData, 80, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    unsigned char pucDHashValue[32] = {0};
    pCpuHashAgent->getSHA256(pucDestData, 80, pucDHashValue);

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值
void HashTest::testCudaSha256D80(){
    unsigned char *pucSrcData = (unsigned char *)malloc(80 * sizeof(unsigned char));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x77, 0x67, 0x33, 0x11, 0x4f, 0xd0, 0x5e, 0x8c, \
                                        0x7a, 0x67, 0x7c, 0x7f, 0xb4, 0xf5, 0x36, 0x6a, \
                                        0xf5, 0x3a, 0xc8, 0x92, 0x87, 0x53, 0xf7, 0x17, \
                                        0xef, 0xdd, 0x76, 0x1a, 0xbb, 0xb4, 0x84, 0xfa};

    for(unsigned int index = 0; index < 80; index++){
        pucSrcData[index] = 0x44;
    }
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    pCudaHashAgent->getSHA256(pucSrcData, 80, pucDestData);

    for(unsigned int index = 0; index < 32; index++){
        CPPUNIT_ASSERT_EQUAL(pucDestData[index], arrRightAnswer[index]);
    }

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值,提前结束
void HashTest::testCudaSha256EndEarlier(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};
    unsigned int uiFinalNonce = 0;
    unsigned int uiRightNonce = 2083236893;
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    struct timeval tTimeBegin;
    struct timeval tTimeEnd;

    // 初始化创世区块
    createGenesisBlock(auiSrcData);

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    gettimeofday(&tTimeBegin, NULL);
    pCudaHashAgent->findNonce(pucSrcData, 80,  2083236890, 4083237900, uiFinalNonce, pucDestData);
    gettimeofday(&tTimeEnd, NULL);
    unsigned long ulCudaTimeConsume = (tTimeEnd.tv_sec * 1000 + tTimeEnd.tv_usec / 1000) -
            (tTimeBegin.tv_sec * 1000 + tTimeBegin.tv_usec / 1000);

    CPPUNIT_ASSERT_EQUAL(uiRightNonce, uiFinalNonce);
    CPPUNIT_ASSERT((ulCudaTimeConsume < 1000));

    free(pucSrcData);
    free(pucDestData);
}

// 性能测试
void HashTest::testCudaSha256Perfmance(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};
    unsigned int uiFinalNonce = 0;
    unsigned int uiRightNonce = 2083236893;
    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    struct timeval tTimeBegin;
    struct timeval tTimeEnd;

    // 初始化创世区块
    createGenesisBlock(auiSrcData);

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    gettimeofday(&tTimeBegin, NULL);
    pCudaHashAgent->findNonce(pucSrcData, 80,  1, 2083237900, uiFinalNonce, pucDestData);
    gettimeofday(&tTimeEnd, NULL);
    unsigned long ulCudaTimeConsume = (tTimeEnd.tv_sec * 1000 + tTimeEnd.tv_usec / 1000) -
            (tTimeBegin.tv_sec * 1000 + tTimeBegin.tv_usec / 1000);

    gettimeofday(&tTimeBegin, NULL);
    pCpuHashAgent->findNonce(pucSrcData, 80, 2000000000, 2083237900, uiFinalNonce, pucDestData);
    gettimeofday(&tTimeEnd, NULL);
    unsigned long ulCpuTimeConsume = (tTimeEnd.tv_sec * 1000 + tTimeEnd.tv_usec / 1000) -
            (tTimeBegin.tv_sec * 1000 + tTimeBegin.tv_usec / 1000);

    printf("\ntestCudaSha256Perfmance time consume : %ld : %ld\n", ulCudaTimeConsume, ulCpuTimeConsume);


    CPPUNIT_ASSERT_EQUAL(uiRightNonce, uiFinalNonce);

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值，找不到的情况
void HashTest::testCpuSha256Unfounded(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};
    unsigned int uiFinalNonce = 0;
    unsigned int uiRightNonce = 2083236893;

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    // 初始化创世区块
    createGenesisBlock(auiSrcData);
    auiSrcData[18] = 0x40ffffff;

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    ERRCODE retValue = pCpuHashAgent->findNonce(pucSrcData, 80, 2083236800, 2083237900,
            uiFinalNonce, pucDestData);
    CPPUNIT_ASSERT(ERROR_NOTFOUND == retValue);

    free(pucSrcData);
    free(pucDestData);
}

// 测试遍历查找比特币创世区块的hash值，找不到的情况
void HashTest::testCudaSha256Unfounded(){
    unsigned int auiSrcData[20] = {0};
    unsigned char *pucSrcData = (unsigned char *)malloc(sizeof(auiSrcData));
    unsigned char *pucDestData = (unsigned char *)malloc(32 * sizeof(unsigned char));
    unsigned char arrRightAnswer[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, \
                                        0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93, \
                                        0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, \
                                        0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f};
    unsigned int uiFinalNonce = 0;
    unsigned int uiRightNonce = 2083236893;
    struct timeval tTimeBegin;
    struct timeval tTimeEnd;

    memset(pucDestData, 0, 32 * sizeof(unsigned char));

    // 初始化创世区块
    createGenesisBlock(auiSrcData);
    auiSrcData[18] = 0x40ffffff;

    memcpy(pucSrcData, auiSrcData, sizeof(auiSrcData));
    gettimeofday(&tTimeBegin, NULL);
    ERRCODE retValue = pCudaHashAgent->findNonce(pucSrcData, 80, 0xEFFFFFFF, 0xFFFFFFFE,
            uiFinalNonce, pucDestData);
    gettimeofday(&tTimeEnd, NULL);
    unsigned long ulCudaTimeConsume = (tTimeEnd.tv_sec * 1000 + tTimeEnd.tv_usec / 1000) -
            (tTimeBegin.tv_sec * 1000 + tTimeBegin.tv_usec / 1000);

    CPPUNIT_ASSERT(ERROR_NOTFOUND == retValue);
    CPPUNIT_ASSERT_EQUAL(uiRightNonce, uiFinalNonce);

    free(pucSrcData);
    free(pucDestData);
}
