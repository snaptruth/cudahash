/*
 * bntime.h
 *
 *  Created on: 2017年8月23日
 */

#ifndef BNOSUTILS_BNTIME_H_
#define BNOSUTILS_BNTIME_H_

#include <time.h>
#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Add gettimeofday function for windows system
#ifdef WIN32
int gettimeofday(struct timeval *tp, void *tzp);
#endif



#endif /* BNOSUTILS_BNTIME_H_ */
