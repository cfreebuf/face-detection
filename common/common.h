// File   common.h
// Author lidongming1@360.cn
// Date   2019-09-23 18:39:45
// Brief

#ifndef _COMMON_COMMON_H_
#define _COMMON_COMMON_H_

#ifdef __cpp_lib_make_unique
using std::make_unique;
#else
template<typename T, typename... TArgs>
std::unique_ptr<T> make_unique(TArgs&&... args) {
  return std::unique_ptr<T>(new T(std::forward<TArgs>(args)...));
}
#endif

#endif
