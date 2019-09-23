// File   file_utils.h
// Author lidongming
// Date   2018-08-28 15:56:16
// Brief

#ifndef COMMONLIB_FILE_UTILS_H_
#define COMMONLIB_FILE_UTILS_H_

#include <libgen.h>
#include <string>
#include <vector>

class FileUtils {
 public:
  static std::vector<std::string> ListDir(const std::string& dir_name);
  static std::string BaseName(const std::string& file_name);
  static bool FileExists(const std::string& file_name);
  static int SaveFile(const std::string& file_name, const char* buf, int len);
  static int CopyFile(const std::string& srcfile, const std::string& newfile);
};

#endif  // COMMONLIB_FILE_UTILS_H_
