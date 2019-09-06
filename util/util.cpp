#include "util/util.h"

std::vector<unsigned char> IntToBytes(int value) {
  std::vector<unsigned char> bytes(4);
  for (int i = 0; i < 4; i++) {
    bytes[3 - i] = (value >> (i * 8));
  }
  return bytes;
}

void PrintDims(const std::vector<double>& dims) {
  for (double v : dims) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}
