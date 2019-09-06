#include <iostream>
#include <iomanip>
#include "third_party/annoy/annoylib.h"
#include "third_party/annoy/kissrandom.h"
#include <chrono>
#include <algorithm>
#include <map>
#include <random>
#include <fstream>
#include "util/string_utils.h"

const int kDims = 512;
typedef AnnoyIndex<int, float, Angular, Kiss64Random> FaceAnnoyIndex;

int main() {
  std::string matrix_file = "matrix.dat";
  std::string target_file = "target.dat";

  std::ifstream ifs(matrix_file);
  if (!ifs.is_open()) {
    std::cerr << "cannot open matrix file:" << matrix_file << std::endl; 
    return -1;
  }

  FaceAnnoyIndex face_annoy_index(kDims);

  std::string line;
  std::vector<std::string> dims_str;
  std::vector<float> dims;
  int id = 0;
  while (std::getline(ifs, line)) {
    StringUtils::Split(line, ",", dims_str);
    if (dims_str.size() != kDims) {
      std::cerr << "invalid dims size:" << dims_str.size() << std::endl; 
      continue;
    }
    for (const std::string& dim : dims_str) {
      dims.push_back(std::stof(dim));
    }
    ++id;
    face_annoy_index.add_item(id, dims.data());
    std::vector<std::string>().swap(dims_str);
    std::vector<float>().swap(dims);
  }
  face_annoy_index.build(2 * kDims);
  face_annoy_index.save("face.index");
  std::cout << "finish build index" << std::endl; 
  ifs.close();

  std::ifstream target_ifs(target_file);
  if (!target_ifs.is_open()) {
    std::cerr << "cannot open target file:" << target_file << std::endl; 
    return -1;
  }

  std::vector<int> neighbor_index;
  std::vector<float> neighbor_dist;

  while (std::getline(target_ifs, line)) {
    StringUtils::Split(line, ",", dims_str);
    if (dims_str.size() != kDims) {
      std::cerr << "invalid dims size:" << dims_str.size() << std::endl; 
      continue;
    }
    for (const std::string& dim : dims_str) {
      dims.push_back(std::stof(dim));
    }

    const int K = 10;
    face_annoy_index.get_nns_by_vector(dims.data(), K, -1, &neighbor_index, &neighbor_dist);
    for (int i = 0; i < neighbor_index.size(); i++) {
      std::cout << "neighbor index:" << neighbor_index[i]
                << " dist:" << neighbor_dist[i] << std::endl;
    }

    std::vector<std::string>().swap(dims_str);
    std::vector<float>().swap(dims);
    neighbor_index.clear();
    neighbor_dist.clear();
  }

  return 0;
}
