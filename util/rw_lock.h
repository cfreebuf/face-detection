// File   rw_lock.h
// Author lidongming1@360.cn
// Date   2019-08-31 02:29:00
// Brief

#ifndef _UTIL_RW_LOCK_H_
#define _UTIL_RW_LOCK_H_

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

typedef boost::shared_mutex SharedMutex;
typedef boost::shared_lock<SharedMutex> ReadLock;
typedef boost::unique_lock<SharedMutex> WriteLock;

#endif
