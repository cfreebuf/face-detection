// File   file_utils.cpp
// Author lidongming
// Date   2018-08-28 15:56:29
// Brief

#include "util/file_utils.h"
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <fstream>

std::vector<std::string> FileUtils::ListDir(const std::string& dir_name) {
    char buf[512] = { '\0' };
    memset(buf, 0, sizeof(buf));
    std::vector<std::string> res;
    DIR* dir;
    struct dirent* ptr;
    dir = opendir(dir_name.c_str());
    if (dir != NULL) {
        while ((ptr = readdir(dir)) != NULL) {
            if (std::string(".") == ptr->d_name
                    || std::string("..") == ptr->d_name) {
                continue;
            }
            // FIXME(lidm):check!
            // std::string path = dir_name + "/" + std::string(ptr->d_name);
            snprintf(buf, sizeof(buf), "%s/%s", dir_name.c_str(), ptr->d_name);
            // std::string path = buf;
            // res.push_back(path);
            res.push_back(buf);
        }
        closedir(dir);
    }

    return res;
}

std::string FileUtils::BaseName(const std::string& file_name) {
    std::string f = basename((char*)file_name.c_str());
    auto it = f.find('.');
    if (it != std::string::npos) {
        return f.substr(0, it);
    } else {
        return f;
    }
}

bool FileUtils::FileExists(const std::string& file_name) {
    if (access(file_name.c_str(), R_OK|W_OK|F_OK) != -1) {
        return true;
    }
    return false;
}

int FileUtils::SaveFile(const std::string& file_name, const char* buf, int len) {
    if (file_name.empty() || !buf || len <= 0) return -1;

    FILE* f = fopen(file_name.c_str(), "w+");
    if (f == NULL) { return -1; }
    fwrite(buf, len, 1, f);
    fclose(f);
    return 0;
}

int FileUtils::CopyFile(const std::string& srcfile, const std::string& newfile) {
    std::ifstream in;
    std::ofstream out;
    in.open(srcfile.c_str());
    if (in.fail()) {
        // LOG_ERROR("open src file error:%s", srcfile.c_str());
        in.close();
        out.close();
        return -1;
    }
    out.open(newfile.c_str(), std::ios::trunc);
    if (out.fail()) {
        // LOG_ERROR("open dst file error:%s", newfile.c_str());
        out.close();
        in.close();
        return -1;
    } else {
        out << in.rdbuf();
        out.close();
        in.close();
        return 0;
    }
}
