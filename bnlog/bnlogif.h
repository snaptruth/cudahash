/*
 * bnlogif.h
 *
 *  Created on: 2017年8月23日
 */

#ifndef BNLOGIF_H_
#define BNLOGIF_H_
#include <string>
#include <assert.h>
#include <stdio.h>

/*
 *  log interface expose to users
 */
void InitBnLog(const char *pcLogParamFile);
void ClearCurrentLog();
void ClearAllLogFile();

void BnLogDebug(const std::string &strInfo);
void BnLogInterface(const std::string &strInfo);
void BnLogError(const std::string &strInfo);
void BnLogFatal(const std::string &strInfo);

/*
 *  Micro for internal use
 */
#define BN_LOG_DEBUG(file, line, function, fm, ...) \
    {\
        char cStrBuffer[1024] = {0};\
        int iStrLength = 0;\
        iStrLength = snprintf(cStrBuffer, 1023, " %s:%d %s ", file, line, function);\
        if(iStrLength > 0 && iStrLength < 1023 ){\
            snprintf(cStrBuffer + iStrLength, 1023 - iStrLength, fm, __VA_ARGS__);\
        }\
        BnLogDebug(cStrBuffer);\
    }

#define BN_LOG_INTERFACE(fm, ...) \
    {\
        char cStrBuffer[1024] = {0};\
        int iStrLength = 0;\
        iStrLength = snprintf(cStrBuffer, 1023, " %s:%d %s ", \
                __FILE__, __LINE__, __FUNCTION__);\
        if(iStrLength > 0 && iStrLength < 1023 ){\
            snprintf(cStrBuffer + iStrLength, 1023 - iStrLength, fm, __VA_ARGS__);\
        }\
        BnLogInterface(cStrBuffer);\
    }

#define BN_LOG_ERROR(fm, ...) \
    {\
        char cStrBuffer[1024] = {0};\
        int iStrLength = 0;\
        iStrLength = snprintf(cStrBuffer, 1023, " %s:%d %s ", \
                __FILE__, __LINE__, __FUNCTION__);\
        if(iStrLength > 0 && iStrLength < 1023 ){\
            snprintf(cStrBuffer + iStrLength, 1023 - iStrLength, fm, __VA_ARGS__);\
        }\
        BnLogError(cStrBuffer);\
    }

#define BN_LOG_FATAL(fm, ...) \
    {\
        char cStrBuffer[1024] = {0};\
        int iStrLength = 0;\
        iStrLength = snprintf(cStrBuffer, 1023, " %s:%d %s ", \
                __FILE__, __LINE__, __FUNCTION__);\
        if(iStrLength > 0 && iStrLength < 1023 ){\
            snprintf(cStrBuffer + iStrLength, 1023 - iStrLength, fm, __VA_ARGS__);\
        }\
        BnLogFatal(cStrBuffer);\
    }

/*
 * Micro for easy use
 */
// 检查函数返回值，不成功即返回错误值，DEBUG版本主动挂起
#define BN_CHECK_RETURN(expression) {\
                ERRCODE returnValue = expression;\
                if(returnValue != SUCCESS){\
                    BN_LOG_ERROR("Return ERRCODE: %d", returnValue);\
                    assert(true);\
                    return returnValue;\
                }\
            }

// 检查函数返回值，不成功即返回错误值，不挂起
#define BN_CHECK_RETURN_NOASSERT(expression) {\
                ERRCODE returnValue = expression;\
                if(returnValue != SUCCESS){\
                    BN_LOG_ERROR("Return ERRCODE: %d", returnValue);\
                    return returnValue;\
                }\
            }

// 检查函数返回值，不成功即返回指定错误码，DEBUG版本主动挂起
#define BN_CHECK_RETURN_NEWCODE(expression,errCode) {\
                ERRCODE returnValue = expression;\
                if(returnValue != SUCCESS){\
                    BN_LOG_ERROR("Return ERRCODE: %d", returnValue);\
                    assert(true);\
                    return errCode;\
                }\
            }

// 检查函数返回值，不成功即返回指定错误码，不挂起
#define BN_CHECK_RETURN_NOASSERT_NEWCODE(expression,errCode) {\
                ERRCODE returnValue = expression;\
                if(returnValue != SUCCESS){\
                    BN_LOG_ERROR("Return ERRCODE: %d", returnValue);\
                    return errCode;\
                }\
            }

// 检查函数返回值，不成功即返回，DEBUG版本主动挂起
#define BN_CHECK_RETURN_RETNULL(expression) {\
                ERRCODE returnValue = expression;\
                if(returnValue != SUCCESS){\
                    BN_LOG_ERROR("Return ERRCODE: %d", returnValue);\
                    assert(true);\
                    return ;\
                }\
            }

// 检查函数返回值，不成功即返回，不挂起
#define BN_CHECK_RETURN_NOASSERT_RETNULL(expression) {\
                ERRCODE returnValue = expression;\
                if(returnValue != SUCCESS){\
                    BN_LOG_ERROR("Return ERRCODE: %d", returnValue);\
                    return ;\
                }\
            }

// 检查指针值是否为空，为空即返回指针为空，DEBUG版本主动挂起
#define BN_CHECK_POINTER(ptr) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    assert(true);\
                    return ERROR_POINTERISNULL;\
                }\
            }

// 检查指针值是否为空，为空即返回指针为空，DEBUG版本主动挂起, Break跳出循环
#define BN_CHECK_POINTER_BREAK(ptr) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    assert(true);\
                    break;\
                }\
            }

// 检查指针值是否为空，为空即返回指针为空，不挂起
#define BN_CHECK_POINTER_NOASSERT(ptr) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    return ERROR_POINTERISNULL;\
                }\
            }

// 检查指针值是否为空，为空即返回指定错误码，DEBUG版本主动挂起
#define BN_CHECK_POINTER_NEWCODE(ptr,errCode) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    assert(true);\
                    return errCode;\
                }\
            }

// 检查指针值是否为空，为空即返回指针为空，不挂起
#define BN_CHECK_POINTER_NOASSERT(ptr) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    return ERROR_POINTERISNULL;\
                }\
            }

// 检查指针值是否为空，为空即返回指定错误码，不挂起
#define BN_CHECK_POINTER_NOASSERT_NEWCODE(ptr,errCode) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    return errCode;\
                }\
            }

// 检查指针值是否为空，为空则返回，DEBUG版本主动挂起
#define BN_CHECK_POINTER_VOID(ptr) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    assert(true);\
                    return;\
                }\
            }

// 检查指针值是否为空，为空则返回，不挂起
#define BN_CHECK_POINTER_NOASSERT_VOID(ptr) {\
                if(NULL == ptr){\
                    BN_LOG_ERROR("Pointer is 0x%p!", ptr);\
                    return;\
                }\
            }

// 判断表达式是否错误，错误则返回
#define BN_CHECK_CONDITION_RET(express,err) {\
                if(express){\
                    BN_LOG_ERROR("Express return %d", false);\
                    return err;\
                }\
            }


#endif /* BNLOGIF_H_ */
