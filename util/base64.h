#ifndef _UTIL_BASE64_H_
#define _UTIL_BASE64_H_

#include <string>

namespace util {

std::string decode64(const std::string &val);

std::string encode64(const std::string &val);

int Base64Decode(const char* data, int data_byte, std::string* res);

int Base64Encode(const unsigned char* data, int data_byte, std::string* res);

}

#endif
