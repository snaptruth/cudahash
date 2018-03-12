/*
 * testmain.cpp
 *
 *  Created on: 2017年7月25日
 */
#include <stdio.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include "cppunit/ui/text/TestRunner.h"
//#include <log4cxx/propertyconfigurator.h>
#include "bnlogif.h"
// Only for linux
//void printMemoryUsage(){
//    FILE *fStatm = fopen("/proc/self/statm","r");
//    int iRss = 0;
//    int iResident = 0;
//    int iShare = 0;
//    int iText = 0;
//    int iLib = 0;
//    int iData = 0;
//    int iDt = 0;
//
//    if(fStatm){
//        fscanf(fStatm,"%u%u%u%u%u%u%u",
//                &iRss, &iResident, &iShare, &iText, &iLib, &iData, &iDt);
//
//        fclose(fStatm);
//
//        printf("Rss: %d, Resident: %d, Share: %d, Text: %d, Lib:%d, Data:%d, Dt:%d\n",
//                iRss, iResident, iShare, iText, iLib, iData, iDt);
//    }
//}

int main()
{
    system("rm -rf /tmp/log/bn/*");
    InitBnLog("CUDAHASH", 2);

//    log4cxx::PropertyConfigurator::configure("log4cxx.properties");

    CppUnit::TextUi::TestRunner runner;

    // 从注册的TestSuite中获取特定的TestSuite, 没有参数获取未命名的TestSuite.
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    // 添加这个TestSuite到TestRunner中
    runner.addTest( registry.makeTest() );    // 指定运行TestSuite

    // 开始运行, 自动显示测试进度和测试结果
//    printMemoryUsage();
    runner.run( "", true );    // Run all tests and wait
//    printMemoryUsage();
}
