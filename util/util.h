#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <vector>
#include <iostream>
#include <string>

std::vector<unsigned char> IntToBytes(int value);
void PrintDims(const std::vector<double>& dims);

template <typename T>
class alignas(8) Padded {
 public:
  explicit Padded() = default;
  explicit Padded(T d) {
    memset(this, 0, sizeof(*this));
    _data = d;
  }

  T& get() {return _data;}
  const T& const_get()      const {return _data;}
  const T& operator*() const {return _data;}

  T _data;
};

// static_assert(!is_valid_pod<char>::value, "");
// static_assert(is_valid_pod<double>::value, "");
// static_assert(is_valid_pod<Padded<char>>::value, "");

void LowerString(std::string& str);

#endif
