// File   face_index.h
// Author lidongming1@360.cn
// Date   2019-08-31 02:20:52
// Brief

#ifndef _FACE_INDEX_H_
#define _FACE_INDEX_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "third_party/annoy/annoylib.h"
#include "third_party/annoy/kissrandom.h"

typedef AnnoyIndex<uint64_t, double, Angular, Kiss64Random> FaceAnnoyIndex;

struct FaceInfo {
  int age;
  int gender;  // 0:make 1:female
  uint64_t face_id;
  std::string name;
  float dist;
  bool valid;
};

using FaceInfoMap = std::unordered_map<uint64_t, FaceInfo>;
class FaceIndex {
 public:
  static const int kDims;

 public:
  FaceIndex();
  // ~FaceIndex();
  void BuildIndexFromFaceDB();

  bool GetNearest(int n, const std::vector<double>& query_dims,
                  std::vector<uint64_t>* index, std::vector<double>* dist);

  bool GetFaceInfo(uint64_t face_id, FaceInfo* face_info);

  bool LoadFaceInfos();
  std::shared_ptr<FaceInfoMap> LoadFaceInfosFromJsonFile();

  std::string FaceInfoString(FaceInfo& face_info) {
    std::string info;
    if (face_info.valid) {
      info = "age:" + std::to_string(face_info.age) + " name:" + face_info.name;
      + " dist:" + std::to_string(face_info.dist);
    } else {
      info = " NO RECORD dist:" + std::to_string(face_info.dist);
    }
    return info;
  }

  int Size() {
    if (face_annoy_index_) {
      return face_annoy_index_->get_n_items();
    }
    return 0;
  }

 private:
  std::unique_ptr<FaceAnnoyIndex> face_annoy_index_;
  std::vector<std::shared_ptr<FaceInfoMap>> face_infos_buf_;
  int buf_index_;
};

#endif
