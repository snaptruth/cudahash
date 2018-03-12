/*
 * bnlogthread.h
 *
 *  Created on: 2017年8月22日
 */

#ifndef BNLOGTHREAD_H_
#define BNLOGTHREAD_H_
#include <bnthread/bnthread.h>
#include <string.h>
#include <fstream>

void createDirectory(const std::string &directoryPath);
void BnLogInfo(const std::string &strLevel, const std::string &strLogInfo);

class BnLogThread: public BnThread {
public:
    static BnLogThread *m_pInstance;
    static BnLogThread *GetInstance();
    virtual ~BnLogThread();

    virtual void OnMessage(int iMsgId, int iLength, void *pvData);

    void InitialLogVar();

    void ClearCurLogFile();
private:
    BnLogThread(const char *pcThreadName, int iPriority = BN_THREAD_PRIORITY_DEFAULT);

    void BackupCurLogFile();

private:
    int m_iLogFileBackNo;
    std::string m_strIndexFile;
    std::string m_strLogFile;

    std::ofstream m_ofLogFile;
    int m_iLogFileSize;
};

#endif /* BNLOGTHREAD_H_ */
