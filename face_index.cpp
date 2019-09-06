// File   face_index.cpp
// Author lidongming1@360.cn
// Date   2019-08-31 02:20:18
// Brief

#include <fstream>
#include <thread>
#include <chrono>
#include <glog/logging.h>

#include "third_party/rapidjson/rapidjson.h"
#include "third_party/rapidjson/document.h"
#include "third_party/rapidjson/writer.h"
#include "third_party/rapidjson/istreamwrapper.h"
#include "third_party/rapidjson/error/en.h"

#include "face_index.h"
#include "face_db.h"
#include "util/util.h"
#include "util/rw_lock.h"
#include "util/string_utils.h"
#include "common/common_gflags.h"

FaceIndex::FaceIndex() : buf_index_(1) {
  for (int i = 0; i < 2; i++) {
    face_infos_buf_.push_back(std::make_shared<FaceInfoMap>());
  }
  face_annoy_index_ = std::make_unique<FaceAnnoyIndex>(FaceIndex::kDims);
  std::thread(&FaceIndex::LoadFaceInfos, this).detach();
}

void FaceIndex::BuildIndexFromFaceDB() {
  uint64_t face_id;
  std::vector<double> dims;

  FaceDB& face_db = FaceDB::Instance();

  try {
    lmdb::txn txn = face_db.Txn();
    lmdb::cursor cursor = face_db.Cursor(txn);

    while (face_db.GetNext(cursor, &face_id, &dims)) {
      // PrintDims(dims);
      if (dims.size() == kDims) {
        face_annoy_index_->add_item(face_id, dims.data());
        LOG(INFO) << "Add dim in to annoy index face_id:" << face_id;
      } else {
        LOG(WARNING) << "Invalid dims size:" << dims.size() << " for face_id:" << face_id;
      }
      dims.clear();
    }
  } catch(const std::exception& ex) {
    LOG(WARNING) << "Build index from face db exception:" << ex.what();
  }
  face_annoy_index_->build(2 * FaceIndex::kDims);
  face_annoy_index_->save(FLAGS_face_index_file.c_str());
  LOG(INFO) << "Finish build index size:" << Size();
}

void FaceIndex::GetNearst(int n, const std::vector<double>& query_dims,
                          std::vector<uint64_t>* index,
                          std::vector<double>* dist) {
  if (Size() > 0) {
    face_annoy_index_->get_nns_by_vector(query_dims.data(), n, -1, index, dist);
  }
}

bool FaceIndex::GetFaceInfo(uint64_t face_id, FaceInfo* face_info) {
  if (buf_index_ == -1) {
    return false;
  }
  const std::shared_ptr<FaceInfoMap>& face_info_map = face_infos_buf_[buf_index_];
  auto it = face_info_map->find(face_id);
  if (it != face_info_map->end()) {
    *face_info = it->second;
    return true;
  }
  return false;
}

bool FaceIndex::LoadFaceInfos() {
  LOG(INFO) << "Start load face infos from json";
  std::shared_ptr<FaceInfoMap> init_face_info_map = LoadFaceInfosFromJsonFile();
  if (init_face_info_map == nullptr) {
    LOG(WARNING) << "Load face infos from json error";
    return false;
  } else {
    face_infos_buf_[0] = init_face_info_map;
    buf_index_ = 0;
    LOG(INFO) << "Finish load face infos from json";
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  while (true) {
    std::shared_ptr<FaceInfoMap> face_info_map = LoadFaceInfosFromJsonFile();
    if (face_info_map != nullptr) {
      int buf_index = 1 - buf_index_;
      face_infos_buf_[buf_index] = face_info_map;
      buf_index_ = buf_index;
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }
  return true;
}

std::shared_ptr<FaceInfoMap> FaceIndex::LoadFaceInfosFromJsonFile() {
  LOG(INFO) << "Start load face infos from:" << FLAGS_face_infos_file;
  std::ifstream ifs(FLAGS_face_infos_file);
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document d;
  rapidjson::ParseResult ok = d.ParseStream(isw);
  if (!ok) {
    LOG(WARNING) << "Parse face infos json error file:" << FLAGS_face_infos_file;
    std::cout << "Config parse error:" << rapidjson::GetParseError_En(ok.Code()) << " " << ok.Offset();
    return nullptr;
  }

  if (!d.HasMember("face_infos")) {
    LOG(WARNING) << "Invalid face infos json error file:" << FLAGS_face_infos_file;
    return nullptr;
  }

  rapidjson::Value& face_infos_array = d["face_infos"];
  if (!face_infos_array.IsArray()) {
    LOG(WARNING) << "Face infos in json is not array file:" << FLAGS_face_infos_file;
    return nullptr;
  }
  LOG(INFO) << "size:" << face_infos_array.Size();

  std::shared_ptr<FaceInfoMap> face_info_map = std::make_shared<FaceInfoMap>();
  for (int i = 0; i < face_infos_array.Size(); i++) {
    const rapidjson::Value& face_info_v = face_infos_array[i];
    if (!face_info_v.HasMember("age")
        || !face_info_v.HasMember("gender")
        || !face_info_v.HasMember("face_id")
        || !face_info_v.HasMember("name")) {
      LOG(WARNING) << "Invalid Face info";
      continue;
    }
    int age = face_info_v["age"].GetInt();
    int gender = face_info_v["gender"].GetInt();
    uint64_t face_id = face_info_v["face_id"].GetUint64();
    const std::string& name = face_info_v["name"].GetString();
    struct FaceInfo face_info={age, gender, face_id, name};
    (*face_info_map)[face_id] = face_info;
  }

  return face_info_map;
}
