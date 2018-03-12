/*
 * bnthread.cpp
 *
 *  Created on: 2017年8月22日
 */
#include <memory.h>
#include <malloc.h>
#include <iostream>
#include <assert.h>
#include <bnthread/bnthread.h>
#include "include/bnconst.h"

// For set thread's name
// TODO: use automake tool to find out if HAVE_SYS_PRCTL_H is defined
#ifdef WIN32
#else
#include <sys/prctl.h>
#endif
#if (defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__DragonFly__))
#include <pthread.h>
#include <pthread_np.h>
#endif
// End for set thread's name

// For set thread's priority
#ifdef WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#endif
// End for set thread's priority


BnThread::BnThread(const char* pcThreadName, int iPriority) :
    m_pThread(NULL),
    m_pcThreadName(pcThreadName),
    m_iThreadPriority(iPriority){

}

BnThread::~BnThread() {
    ExitThread();
}

bool BnThread::CreateThread()
{
    if (NULL == m_pThread)
        m_pThread = new std::thread(&BnThread::Process, this);
    return true;
}

std::thread::id BnThread::GetThreadId()
{
    std::thread::id tId;
    if(NULL == m_pThread){
        assert(0);
    }else{
        tId = m_pThread->get_id();
    }
    return tId;
}

void BnThread::ExitThread()
{
    if (NULL == m_pThread){
        return;
    }

    // Create a new ThreadMsg
    ThreadMsg* threadMsg = new ThreadMsg(BN_MSG_EXIT_THREAD, 0, NULL);

    // Put exit thread message into the queue
    {
        std::lock_guard<std::mutex> lockGuard(m_mutex);
        m_queueThreadMsg.push(threadMsg);
        m_cv.notify_one();
    }

    m_pThread->join();
    delete m_pThread;
    m_pThread = NULL;
}

void BnThread::SetPriority(int iPriority){
    if(BN_THREAD_PRIORITY_MIN > iPriority || BN_THREAD_PRIORITY_MAX < iPriority){
        return;
    }

    if(m_iThreadPriority != iPriority){
        if (NULL == m_pThread){
            return;
        }
        m_iThreadPriority = iPriority;
        // Create a new ThreadMsg
        ThreadMsg* threadMsg = new ThreadMsg(BN_MSG_CHANGE_PRIORITY, 0, NULL);

        // Put exit thread message into the queue
        {
            std::lock_guard<std::mutex> lockGuard(m_mutex);
            m_queueThreadMsg.push(threadMsg);
            m_cv.notify_one();
        }
    }

}

void BnThread::PostMsg(int iMsgId, const void* pvData, int iLength)
{
    if (NULL == m_pThread){
        return;
    }

    void *pvNewData = NULL;
    if(NULL != pvData && iLength > 0){
        pvNewData = malloc(iLength + 1);
        if(NULL == pvNewData){
            return;
        }
        memset(pvNewData, 0, iLength + 1);
        memcpy(pvNewData, pvData, iLength);
    }

    ThreadMsg* threadMsg = new ThreadMsg(iMsgId, iLength, pvNewData);

    // Add user data msg to queue and notify worker thread
    std::unique_lock<std::mutex> uniqueLock(m_mutex);
    m_queueThreadMsg.push(threadMsg);
    m_cv.notify_one();
}

void BnThread::ClearMessageQueue(){
    std::unique_lock<std::mutex> uniqueLock(m_mutex);
    while (!m_queueThreadMsg.empty()){
        ThreadMsg *pThreadMsg = m_queueThreadMsg.front();
        m_queueThreadMsg.pop();
        if(NULL != pThreadMsg->m_pvMsg){
            free(pThreadMsg->m_pvMsg);
            pThreadMsg->m_pvMsg = NULL;
        }
        delete pThreadMsg;
    }
}

void BnThread::Process()
{
    RenameThread(m_pcThreadName);
    SetThreadPriority(m_iThreadPriority);
    while (true){
        ThreadMsg *pThreadMsg = NULL;
        {
            // Wait for a message to be added to the queue
            std::unique_lock<std::mutex> uniqueLock(m_mutex);
            while (m_queueThreadMsg.empty())
                m_cv.wait(uniqueLock);

            if (m_queueThreadMsg.empty())
                continue;

            pThreadMsg = m_queueThreadMsg.front();
            m_queueThreadMsg.pop();
        }

        switch (pThreadMsg->m_id){
            case BN_MSG_EXIT_THREAD:
                ClearMessageQueue();
                std::cout << "Exit thread on " << m_pcThreadName << std::endl;
                break;

            case BN_MSG_CHANGE_PRIORITY:
                SetThreadPriority(m_iThreadPriority);
                break;

            default:
                OnMessage(pThreadMsg->m_id, pThreadMsg->m_iLength, pThreadMsg->m_pvMsg);

                break;
        }

        // free memory
        if(NULL != pThreadMsg->m_pvMsg){
            free(pThreadMsg->m_pvMsg);
            pThreadMsg->m_pvMsg = NULL;
        }
        delete pThreadMsg;
    }
}

void BnThread::OnMessage(int iMsgId, int iLength, void *pvData){
    return;
}


void BnThread::RenameThread(const char* pcThreadName)
{
#if defined(PR_SET_NAME)
    // Only the first 15 characters are used (16 - NUL terminator)
    ::prctl(PR_SET_NAME, pcThreadName, 0, 0, 0);
#elif (defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__DragonFly__))
    pthread_set_name_np(pthread_self(), pcThreadName);

#elif defined(MAC_OSX)
    pthread_setname_np(pcThreadName);
#else
    // Prevent warnings for unused parameters...
    (void)pcThreadName;
#endif
}


void BnThread::SetThreadPriority(int iPriority)
{
#ifdef WIN32
    SetThreadPriority(GetCurrentThread(), iPriority);
#else
#ifdef PRIO_THREAD
    setpriority(PRIO_THREAD, 0, iPriority);
#else
    setpriority(PRIO_PROCESS, 0, iPriority);
#endif
#endif
}




