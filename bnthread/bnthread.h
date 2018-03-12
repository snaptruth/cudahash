/*
 * bnthread.h
 *
 *  Created on: 2017年8月22日
 */

#ifndef BNTHREAD_BNTHREAD_H_
#define BNTHREAD_BNTHREAD_H_

#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "include/bnconst.h"

struct ThreadMsg
{
    ThreadMsg(int i, int iLength, void* pvMsg) { m_id = i; m_iLength = iLength; m_pvMsg = pvMsg; }
    int m_id;
    int m_iLength;
    void* m_pvMsg;
};

class BnThread {
public:
    BnThread(const char* pcThreadName, int iPriority = BN_THREAD_PRIORITY_DEFAULT);

    virtual ~BnThread();

    void PostMsg(int iMsgId, const void *pvData, int iLength);

    virtual void OnMessage(int iMsgId, int iLength, void *pvData);

    bool CreateThread();

    void ExitThread();

    void SetPriority(int iPriority);

private:
    std::thread::id GetThreadId();

    void Process();

    void ClearMessageQueue();

    void RenameThread(const char* pcThreadName);

    void SetThreadPriority(int iPriority);

    // DO NOT ALLOW
//    BnThread(const BnThread&);
//    BnThread& operator=(const BnThread&);

private:
    std::thread* m_pThread;
    std::queue<ThreadMsg *> m_queueThreadMsg;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    const char* m_pcThreadName;
    int m_iThreadPriority;
};

#endif /* BNTHREAD_BNTHREAD_H_ */
