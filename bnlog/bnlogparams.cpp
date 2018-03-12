/*
 * bnlogparams.cpp
 *
 *  Created on: 2017年8月22日
 */
#include <assert.h>
#include <iostream>
#include "bnlogparams.h"
#include "tinyxml/tinyxml.h"

BnLogParams * BnLogParams::m_pInstance = new BnLogParams;

BnLogParams::BnLogParams() {
    m_iLogLevel = 0;
    m_strLogFilePath = "/tmp/log/bn/";
    m_strLogFilePrefix = "default";
    m_iMaxFileSize = 10 * 1024 * 1024;
    m_iMaxFileBackup = 10;
    m_iLogLevel = BN_LOGLEVEL_INTERFACE | BN_LOGLEVEL_ERROR | BN_LOGLEVEL_FATAL;
    m_bPrintToConsole = true;
    m_bInitialed = false;
}

BnLogParams::~BnLogParams() {
}

BnLogParams *BnLogParams::GetInstance() {
    return m_pInstance;
}

void BnLogParams::InitialParams(const char *pcParamFiles){
    BN_LOG_CHECK_POINTER_VOID(pcParamFiles);

    if(m_bInitialed){
        return;
    }

    TiXmlDocument xmlDocument;
    bool bResult = xmlDocument.LoadFile(pcParamFiles);
    if(bResult != true)
    {
        std::cout<<"ERROR! load xml file failed, ErrCode:" << bResult <<std::endl;
        return;
    }

    TiXmlElement *pRootElement = xmlDocument.RootElement();
    BN_LOG_CHECK_POINTER_VOID(pRootElement);

    pRootElement->Attribute("FILEMAXSIZE", &m_iMaxFileSize);
    pRootElement->Attribute("BACKUPNUM", &m_iMaxFileBackup);
    pRootElement->QueryBoolAttribute("CONSOLE", &m_bPrintToConsole);
    m_iMaxFileSize *= 1024 * 1024;

    TiXmlElement *pParamElement = pRootElement->FirstChildElement("FILEPATH");
    BN_LOG_CHECK_POINTER_VOID(pParamElement);
    m_strLogFilePath = pParamElement->GetText();

    pParamElement = pRootElement->FirstChildElement("FILEPREFIX");
    BN_LOG_CHECK_POINTER_VOID(pParamElement);
    m_strLogFilePrefix = pParamElement->GetText();

    pParamElement = pRootElement->FirstChildElement("LOGLEVEL");
    BN_LOG_CHECK_POINTER_VOID(pParamElement);
    SetLogLevel(pParamElement->GetText());

    m_bInitialed = true;
}

void BnLogParams::SetLogLevel(std::string strLogLevel){
    if(strLogLevel.find("DEBUG")){
        m_iLogLevel |= BN_LOGLEVEL_DEBUG;
    }
}

