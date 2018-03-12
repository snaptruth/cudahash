/*
 * bnlogparams.h
 *
 *  Created on: 2017年8月22日
 */
#ifndef BNLOGPARAMS_H_
#define BNLOGPARAMS_H_

#include <string>

#define BN_LOGLEVEL_DEBUG       (1<<0)
#define BN_LOGLEVEL_INTERFACE   (1<<10)
#define BN_LOGLEVEL_ERROR       (1<<20)
#define BN_LOGLEVEL_FATAL       (1<<30)

// 检查指针值是否为空，为空则返回，DEBUG版本主动挂起
#define BN_LOG_CHECK_POINTER_VOID(ptr) {\
                if(NULL == ptr){\
                    std::cout<<__FILE__ << __FUNCTION__ << __LINE__ << "Pointer is NULL" << std::endl;\
                    assert(true);\
                    return;\
                }\
            }

class BnLogParams {
public:
    static BnLogParams *m_pInstance;
    static BnLogParams *GetInstance();
    virtual ~BnLogParams();

    void InitialParams(const char *pcParamFiles);

    std::string getLogDirection(){return m_strLogFilePath;}
    std::string getLogFilePrefix(){return m_strLogFilePrefix;}
    int getMaxFileSize(){return m_iMaxFileSize;}
    int getMaxFileBackup(){return m_iMaxFileBackup;}
    int getLogLevel(){return m_iLogLevel;}
    bool isPrintToConsole(){return m_bPrintToConsole;}

private:
    BnLogParams();
    void SetLogLevel(std::string strLogLevel);
private:
    std::string m_strLogFilePath;
    std::string m_strLogFilePrefix;
    int m_iMaxFileSize;
    int m_iMaxFileBackup;
    int m_iLogLevel;
    bool m_bPrintToConsole;
    bool m_bInitialed;
};

#endif /* BNLOGPARAMS_H_ */
