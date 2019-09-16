// File   lmdb_wrapper.h
// Author lidongming1@360.cn
// Date   2019-09-01 13:11:34
// Brief

#ifndef _LMDB_WRAPPER_H_
#define _LMDB_WRAPPER_H_

#include <string>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <iostream>
#include "third_party/lmdbxx/lmdb++.h"

namespace lmdb_wrapper {

template <class T>
struct is_valid_pod {
  static constexpr bool value = std::is_pod<T>::value && !(sizeof(T) % 8);
};

template <typename T>
class LMDBValue {
 public:
  template<typename Arg>
  static bool is_aligned(const void* ptr) {
   return !(reinterpret_cast<std::uintptr_t>(ptr) % alignof(Arg));
  }

  size_t   size()  const {return size_;}
  const T* begin() const {return data_;}
  const T* end()   const {return data_ + size_;}

  explicit LMDBValue(const T* ptr, size_t count) : data_(ptr), size_(count) {}

  explicit LMDBValue(const T* b, const T* e)
    : data_(b), size_(static_cast<size_t>(e - b)) {
    assert (e - b >= 0);
  }

  explicit LMDBValue(MDB_val& v) : data_(reinterpret_cast<T*>(v.mv_data)),
           size_(v.mv_size / sizeof(T)) {
    assert(!(v.mv_size % sizeof(T)));
    assert(is_aligned<T>(data_));
  }

  // static LMDBValue<T> makeNull() {return LMDBValue<T>{nullptr, nullptr};}
  LMDBValue<T>() : data_(nullptr), size_(0) {}

  bool IsNull() { return data_ == nullptr; }

  const T& operator[](size_t n) const {
    assert(n < size_);
    return data_[n];
  }

  template <typename K>
  const K& TransType() const {
    assert(sizeof(K) == sizeof(T) * size_);
    assert(is_aligned<K>(data_));
    return *reinterpret_cast<const K*>(begin());
  }

  template <typename K>
  LMDBValue<K> TransValue() {
    assert(!((sizeof(T)*size_) % sizeof(K)));
    assert(is_aligned<K>(data_));
    return LMDBValue<K>((K*)data_, size_ / sizeof(K));
  }

  const T* data_;
  size_t size_;
};

class LMDBEnv {
 static constexpr size_t kDefaultMaxLMDBSize = 1UL * 1024UL * 1024UL * 1024UL; // 1GB
 static constexpr size_t kDefaultMaxLMDBDbs = 10;

 lmdb::env env_;

 public:
  explicit LMDBEnv(const std::string& db_path,
                   size_t max_size = kDefaultMaxLMDBSize,
                   size_t max_dbs = kDefaultMaxLMDBDbs)
    : env_{lmdb::env::create()} {
    env_.set_mapsize(
        (max_size % 4096 == 0) ? max_size
          : (max_size + 4096 - (max_size % 4096)));
    env_.set_max_dbs(max_dbs);
    env_.open(db_path.c_str(), MDB_NOSUBDIR, 0664);
  }

  lmdb::txn OpenWriteTxn() { return lmdb::txn::begin(env_, nullptr); }
  lmdb::txn OpenReadTxn()  { return lmdb::txn::begin(env_, nullptr, MDB_RDONLY);}
};

class LMDBDbi {
 public:
  const std::string db_name_;
  const unsigned int dbi_flags_ = MDB_CREATE;

  lmdb::dbi dbi(lmdb::txn& txn) {
    return lmdb::dbi::open(txn, db_name_.c_str(), dbi_flags_);
  }

  explicit LMDBDbi(const std::string& db_name,
                   const unsigned int dbi_flags = MDB_CREATE)
    :db_name_{db_name}, dbi_flags_(dbi_flags) {}

  template <typename K, typename V>
  bool put(lmdb::txn& txn, const K& key, const V& val) {
    MDB_val mdb_key{sizeof(key), (void*)(&key)};
    MDB_val mdb_val{sizeof(val), (void*)(&val)};
    int rc = mdb_put(txn, dbi(txn), &mdb_key, &mdb_val, 0);
    return rc == MDB_SUCCESS;
  }

  template <typename K, typename V>
  bool put(lmdb::txn& txn, const K& key, const V* dat, size_t count) {
    MDB_val mdb_key{sizeof(key), (void*)&key};
    MDB_val mdb_val{sizeof(dat[0])*count, (void*)dat};
    int rc = mdb_put(txn, dbi(txn), &mdb_key, &mdb_val, 0);
    return rc == MDB_SUCCESS;
  }

  template <typename K>
  LMDBValue<unsigned char> get(lmdb::txn& txn, const K& key) {
    MDB_val mdb_val;
    MDB_val mdb_key{sizeof(key), (void*)&key};
    int rc = mdb_get(txn, dbi(txn), &mdb_key, &mdb_val);
    if (rc != MDB_SUCCESS) {
      return LMDBValue<unsigned char>();
    }
    return LMDBValue<unsigned char>{mdb_val};
  }

  template <typename K>
  bool exists(lmdb::txn& txn, const K& key) {
    MDB_val mdb_key {sizeof(key), (void*)&key};
    MDB_val mdb_val;
    int rc = mdb_get(txn, dbi(txn), &mdb_key, &mdb_val);
    return rc == MDB_SUCCESS;
  }

  template <typename K>
  bool del(lmdb::txn& txn, const K& key) {
    MDB_val mdb_key{sizeof(key), (void*)(&key)};
    int rc = mdb_del(txn, dbi(txn), &mdb_key, nullptr);
    return (rc == MDB_SUCCESS);
  }

  lmdb::cursor cursor(lmdb::txn& txn) {
     return lmdb::cursor::open(txn, dbi(txn));
  }

  LMDBValue<unsigned char> get_next(lmdb::cursor& cursor,
                                    const MDB_cursor_op op,
                                    LMDBValue<unsigned char>* key) {
    MDB_val mdb_key;
    MDB_val mdb_val;

    int rc = mdb_cursor_get(cursor, &mdb_key, &mdb_val, op);
    if (rc !=MDB_SUCCESS) {
      return LMDBValue<unsigned char>();
    }
    *key = LMDBValue<unsigned char>(mdb_key);
    return LMDBValue<unsigned char>(mdb_val);
  }
};

template <typename KeyType, typename ValueType>
class LMDBDatabase {
 public:
  static_assert(is_valid_pod<KeyType>::value, "");
  static_assert(is_valid_pod<ValueType>::value, "");

  LMDBDbi lmdb_dbi_;

  explicit LMDBDatabase(const std::string &db_name) :lmdb_dbi_{db_name} {}
  explicit LMDBDatabase(const std::string &db_name,
                        const unsigned int dbi_flags)
    : lmdb_dbi_{db_name, dbi_flags} {}

   bool get(lmdb::txn& txn, const KeyType& key, LMDBValue<ValueType>* value) {
    LMDBValue<unsigned char> v = lmdb_dbi_.get(txn, key);
    if (v.IsNull()) {
      return false;
    } else {
      *value = std::move(v.TransValue<ValueType>());
      return true;
    }
  }

  bool put(lmdb::txn& txn, const KeyType& key, const ValueType* data,
           size_t count) {
    return lmdb_dbi_.put(txn, key, data, count);
  }

  bool exists(lmdb::txn& txn, const KeyType& key) {
    return lmdb_dbi_.exists(txn, key);
  }

  bool del(lmdb::txn& txn, const KeyType& key) {
    return lmdb_dbi_.del(txn, key);
  }

  lmdb::cursor cursor(lmdb::txn& txn) {
    return lmdb_dbi_.cursor(txn);
  }

  bool get_next(lmdb::cursor& cursor, LMDBValue<KeyType>* key,
                LMDBValue<ValueType>* value) {
      LMDBValue<unsigned char> k;
      LMDBValue<unsigned char> v = lmdb_dbi_.get_next(cursor, MDB_NEXT, &k);
      if (v.IsNull()) {
        return false;
      } else {
        *key = k.TransValue<KeyType>();
        *value = std::move(v.TransValue<ValueType>());
        return true;
      }
  }
};

}  // namespace lmdbcols

#endif
