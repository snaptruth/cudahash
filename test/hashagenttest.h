/*
 * cudahashagenttest.h
 *
 *  Created on: 2017年7月25日
 */

#ifndef TEST_HASHAGENTTEST_H_
#define TEST_HASHAGENTTEST_H_

#include <stddef.h>
#include "cppunit/extensions/HelperMacros.h"

class CudaHashAgent;
class CpuHashAgent;
class HashTest : public CppUnit::TestFixture {
    // 声明一个TestSuite
    CPPUNIT_TEST_SUITE(HashTest);
    // 添加测试用例到TestSuite, 新的测试用例都需要在这儿声明
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testCpuSha256abc);
    CPPUNIT_TEST(testCudaSha256abc);
    CPPUNIT_TEST(testCpuSha256ABC);
    CPPUNIT_TEST(testCudaSha256ABC);
    CPPUNIT_TEST(testCpuSha256123);
    CPPUNIT_TEST(testCudaSha256123);
    CPPUNIT_TEST(testCpuSha256DoubleBlock);
    CPPUNIT_TEST(testCudaSha256DoubleBlock);
    CPPUNIT_TEST(testCpuSha256RandomLong);
    CPPUNIT_TEST(testCudaSha256RandomLong);
    CPPUNIT_TEST(testCpuSha256GenesisBlock);
    CPPUNIT_TEST(testCudaSha256GenesisBlock);
    CPPUNIT_TEST(testCpuSha256FoundNonce);
    CPPUNIT_TEST(testCudaSha256FoundNonce);
    CPPUNIT_TEST(testCpuSha256D80);
    CPPUNIT_TEST(testCudaSha256D80);
    CPPUNIT_TEST(testCudaSha256EndEarlier);


    // 性能测试，需要时再打开
//    CPPUNIT_TEST(testCudaSha256Perfmance);
//    CPPUNIT_TEST(testCpuSha256Unfounded);
//    CPPUNIT_TEST(testCudaSha256Unfounded);
    // TestSuite声明完成
    CPPUNIT_TEST_SUITE_END();
protected:
    CudaHashAgent *pCudaHashAgent;
    CpuHashAgent *pCpuHashAgent;
public:
    HashTest(){pCudaHashAgent = NULL; pCpuHashAgent = NULL;};

    // 初始化函数
    void setUp();
    // 清理函数
    void tearDown();

    // 添加新的测试函数
    void testAdd();

    void testCpuSha256abc();
    void testCudaSha256abc();
    void testCpuSha256ABC();
    void testCudaSha256ABC();
    void testCpuSha256123();
    void testCudaSha256123();
    void testCpuSha256DoubleBlock();
    void testCudaSha256DoubleBlock();
    void testCpuSha256RandomLong();
    void testCudaSha256RandomLong();
    void testCpuSha256GenesisBlock();
    void testCudaSha256GenesisBlock();
    void testCpuSha256FoundNonce();
    void testCudaSha256FoundNonce();
    void testCpuSha256D80();
    void testCudaSha256D80();
    void testCudaSha256EndEarlier();
    void testCudaSha256Perfmance();
    void testCpuSha256Unfounded();
    void testCudaSha256Unfounded();
private:
    void createGenesisBlock(unsigned int *puiBlock);

};


#endif /* TEST_HASHAGENTTEST_H_ */
