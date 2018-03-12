/*
 * bnlogthread.cpp
 *
 *  Created on: 2017年8月22日
 */

#include "bnlogthread.h"
#include <bnosutils/bntime.h>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include "include/bnconst.h"
#include "bnlogparams.h"
#include "bnlogif.h"

#ifdef WIN32
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

#ifdef WIN32
#define ACCESS(fileName,accessMode) _access(fileName,accessMode)
#define MKDIR(path) _mkdir(path)
#else
#define ACCESS(fileName,accessMode) access(fileName,accessMode)
#define MKDIR(path) mkdir(path,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

void createDirectory(const std::string &directoryPath){
    int dirPathLen = directoryPath.length();

    char tmpDirPath[256] = { 0 };
    for (int i = 0; i < dirPathLen; ++i){
        tmpDirPath[i] = directoryPath[i];
        if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/'){
            if (ACCESS(tmpDirPath, 0) != 0){
                int iRetValue = MKDIR(tmpDirPath);
                if(0 != iRetValue){
                    std::cout<< "mkdir " << tmpDirPath << " return " << iRetValue \
                            <<", Make sure you got Authorization "<< std::endl;
                }
            }
        }
    }
}

BnLogThread * BnLogThread::m_pInstance = new BnLogThread("BnLogThread", BN_THREAD_PRIORITY_10);

void InitBnLog(const char *pcLogParamFile){
    BnLogParams *pLogParams = BnLogParams::GetInstance();
    pLogParams->InitialParams(pcLogParamFile);

    BnLogThread *pLogThread = BnLogThread::GetInstance();
    pLogThread->InitialLogVar();
    pLogThread->CreateThread();
}

void ClearCurrentLog(){
    BnLogThread *pLogThread = BnLogThread::GetInstance();
    pLogThread->ClearCurLogFile();
}

void ClearAllLogFile(){
    BnLogThread *pLogThread = BnLogThread::GetInstance();
    pLogThread->ClearCurLogFile();

    system("rm -rf /tmp/log/bn/*_*");
}

void BnLogDebug(const std::string &strInfo){
    if(BnLogParams::GetInstance()->getLogLevel() & BN_LOGLEVEL_DEBUG){
        BnLogInfo("DEBUG", strInfo);
    }
}

void BnLogInterface(const std::string &strInfo){
    if(BnLogParams::GetInstance()->getLogLevel() & BN_LOGLEVEL_INTERFACE){
        BnLogInfo("IFACE", strInfo);
    }
}

void BnLogError(const std::string &strInfo){
    if(BnLogParams::GetInstance()->getLogLevel() & BN_LOGLEVEL_ERROR){
        BnLogInfo("ERROR", strInfo);
    }
}

void BnLogFatal(const std::string &strInfo){
    if(BnLogParams::GetInstance()->getLogLevel() & BN_LOGLEVEL_FATAL){
        BnLogInfo("FATAL", strInfo);
    }
}

void BnLogInfo(const std::string &strLevel, const std::string &strLogInfo){
    struct timeval tTimeOfDay;
    gettimeofday(&tTimeOfDay, NULL);
    struct tm* pTime;
    pTime = localtime(&tTimeOfDay.tv_sec);

    char arrcTime[24] = {0};
    snprintf(arrcTime, sizeof(arrcTime), "%04d-%02d-%02d %02d:%02d:%02d-%03d",
            pTime->tm_year+1900,  pTime->tm_mon+1, pTime->tm_mday,
            pTime->tm_hour, pTime->tm_min, pTime->tm_sec,
            tTimeOfDay.tv_usec/1000);

    std::string strAllInfo = "[" + strLevel + "] " + "[" + std::string(arrcTime) + "] " + strLogInfo;
    BnLogThread::GetInstance()->PostMsg(BN_MSG_POST_LOGINFO, strAllInfo.c_str(), strAllInfo.length());

    // Print to console according to user's chooses
    if(BnLogParams::GetInstance()->isPrintToConsole()){
        std::cout<< strAllInfo << std::endl;
    }
    strAllInfo.clear();
    // need c++11 for this
    strAllInfo.shrink_to_fit();
}

BnLogThread::BnLogThread(const char *pcThreadName, int iPriority) : BnThread(pcThreadName, iPriority){

}

BnLogThread::~BnLogThread() {
}

BnLogThread *BnLogThread::GetInstance() {
    return m_pInstance;
}

void BnLogThread::OnMessage(int iMsgId, int iLength, void *pvData){
    switch(iMsgId){
    case BN_MSG_POST_LOGINFO:
        try{
            m_iLogFileSize += iLength;
            if(!m_ofLogFile.is_open()){
                // Something may not happen
                m_ofLogFile.open(m_strLogFile);
            }
            char *pcData = static_cast<char *>(pvData);
            std::string strInfo(pcData, iLength);
            m_ofLogFile << strInfo << std::endl;
            if(m_iLogFileSize >= BnLogParams::GetInstance()->getMaxFileSize()){
                BackupCurLogFile();
                // After current log file backuped, we should reopen current log file
                m_ofLogFile.open(m_strLogFile);
            }
        }catch(...){
        }
        break;
    default:
        break;
    }
}

void BnLogThread::InitialLogVar(){
    std::fstream fLogDirection;
    m_strIndexFile = BnLogParams::GetInstance()->getLogDirection() + "/index";
    m_strLogFile = BnLogParams::GetInstance()->getLogDirection() + "/" +
            BnLogParams::GetInstance()->getLogFilePrefix() + ".log";

    createDirectory(BnLogParams::GetInstance()->getLogDirection());

    m_iLogFileBackNo = 0;
    std::ifstream ifIndexFile;
    ifIndexFile.open(m_strIndexFile);
    if(ifIndexFile.is_open()){
        ifIndexFile >> m_iLogFileBackNo;
        ifIndexFile.close();
        m_iLogFileBackNo ++;    // Next NO will plus one
    }

    std::ifstream ifCurLogFile;
    ifCurLogFile.open(m_strLogFile);
    if(ifCurLogFile.is_open()){
        // If last log file exist, then we should rename to backup file
        ifCurLogFile.close();
        BackupCurLogFile();
    }

    // Open or reopen current log file
    m_ofLogFile.open(m_strLogFile);
}

void BnLogThread::BackupCurLogFile(){
    std::string strCurBackFile = BnLogParams::GetInstance()->getLogDirection() + "/" +
            BnLogParams::GetInstance()->getLogFilePrefix() + "_" +
            std::to_string(m_iLogFileBackNo) + ".log";
    rename(m_strLogFile.c_str(), strCurBackFile.c_str());

    std::ofstream ofIndexFile;
    ofIndexFile.open(m_strIndexFile);
    ofIndexFile << m_iLogFileBackNo << std::endl;
    ofIndexFile.close();
    m_iLogFileBackNo++;
    if(m_iLogFileBackNo >= BnLogParams::GetInstance()->getMaxFileBackup()){
        // Then we write back
        m_iLogFileBackNo = 0;
    }

    m_ofLogFile.open(m_strLogFile);
    m_ofLogFile << std::endl;
    m_ofLogFile.close();
    m_iLogFileSize = 0;
}

void BnLogThread::ClearCurLogFile(){
    m_ofLogFile.close();
    m_ofLogFile.open(m_strLogFile);
    m_ofLogFile << std::endl;
    m_iLogFileSize = 0;
    m_ofLogFile.close();
}
