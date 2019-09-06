// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: facenet.proto

#ifndef PROTOBUF_INCLUDED_facenet_2eproto
#define PROTOBUF_INCLUDED_facenet_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3007000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3007001 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_facenet_2eproto

// Internal implementation detail -- do not use these members.
struct TableStruct_facenet_2eproto {
  static const ::google::protobuf::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors_facenet_2eproto();
namespace facenet_server {
class EmbeddingReply;
class EmbeddingReplyDefaultTypeInternal;
extern EmbeddingReplyDefaultTypeInternal _EmbeddingReply_default_instance_;
class EmbeddingRequest;
class EmbeddingRequestDefaultTypeInternal;
extern EmbeddingRequestDefaultTypeInternal _EmbeddingRequest_default_instance_;
}  // namespace facenet_server
namespace google {
namespace protobuf {
template<> ::facenet_server::EmbeddingReply* Arena::CreateMaybeMessage<::facenet_server::EmbeddingReply>(Arena*);
template<> ::facenet_server::EmbeddingRequest* Arena::CreateMaybeMessage<::facenet_server::EmbeddingRequest>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace facenet_server {

// ===================================================================

class EmbeddingRequest :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:facenet_server.EmbeddingRequest) */ {
 public:
  EmbeddingRequest();
  virtual ~EmbeddingRequest();

  EmbeddingRequest(const EmbeddingRequest& from);

  inline EmbeddingRequest& operator=(const EmbeddingRequest& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  EmbeddingRequest(EmbeddingRequest&& from) noexcept
    : EmbeddingRequest() {
    *this = ::std::move(from);
  }

  inline EmbeddingRequest& operator=(EmbeddingRequest&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const EmbeddingRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const EmbeddingRequest* internal_default_instance() {
    return reinterpret_cast<const EmbeddingRequest*>(
               &_EmbeddingRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(EmbeddingRequest* other);
  friend void swap(EmbeddingRequest& a, EmbeddingRequest& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline EmbeddingRequest* New() const final {
    return CreateMaybeMessage<EmbeddingRequest>(nullptr);
  }

  EmbeddingRequest* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<EmbeddingRequest>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const EmbeddingRequest& from);
  void MergeFrom(const EmbeddingRequest& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(EmbeddingRequest* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string image_base64 = 1;
  void clear_image_base64();
  static const int kImageBase64FieldNumber = 1;
  const ::std::string& image_base64() const;
  void set_image_base64(const ::std::string& value);
  #if LANG_CXX11
  void set_image_base64(::std::string&& value);
  #endif
  void set_image_base64(const char* value);
  void set_image_base64(const char* value, size_t size);
  ::std::string* mutable_image_base64();
  ::std::string* release_image_base64();
  void set_allocated_image_base64(::std::string* image_base64);

  // @@protoc_insertion_point(class_scope:facenet_server.EmbeddingRequest)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr image_base64_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_facenet_2eproto;
};
// -------------------------------------------------------------------

class EmbeddingReply :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:facenet_server.EmbeddingReply) */ {
 public:
  EmbeddingReply();
  virtual ~EmbeddingReply();

  EmbeddingReply(const EmbeddingReply& from);

  inline EmbeddingReply& operator=(const EmbeddingReply& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  EmbeddingReply(EmbeddingReply&& from) noexcept
    : EmbeddingReply() {
    *this = ::std::move(from);
  }

  inline EmbeddingReply& operator=(EmbeddingReply&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const EmbeddingReply& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const EmbeddingReply* internal_default_instance() {
    return reinterpret_cast<const EmbeddingReply*>(
               &_EmbeddingReply_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(EmbeddingReply* other);
  friend void swap(EmbeddingReply& a, EmbeddingReply& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline EmbeddingReply* New() const final {
    return CreateMaybeMessage<EmbeddingReply>(nullptr);
  }

  EmbeddingReply* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<EmbeddingReply>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const EmbeddingReply& from);
  void MergeFrom(const EmbeddingReply& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(EmbeddingReply* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated double dim = 2;
  int dim_size() const;
  void clear_dim();
  static const int kDimFieldNumber = 2;
  double dim(int index) const;
  void set_dim(int index, double value);
  void add_dim(double value);
  const ::google::protobuf::RepeatedField< double >&
      dim() const;
  ::google::protobuf::RepeatedField< double >*
      mutable_dim();

  // int32 error = 1;
  void clear_error();
  static const int kErrorFieldNumber = 1;
  ::google::protobuf::int32 error() const;
  void set_error(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:facenet_server.EmbeddingReply)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< double > dim_;
  mutable std::atomic<int> _dim_cached_byte_size_;
  ::google::protobuf::int32 error_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_facenet_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// EmbeddingRequest

// string image_base64 = 1;
inline void EmbeddingRequest::clear_image_base64() {
  image_base64_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& EmbeddingRequest::image_base64() const {
  // @@protoc_insertion_point(field_get:facenet_server.EmbeddingRequest.image_base64)
  return image_base64_.GetNoArena();
}
inline void EmbeddingRequest::set_image_base64(const ::std::string& value) {
  
  image_base64_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:facenet_server.EmbeddingRequest.image_base64)
}
#if LANG_CXX11
inline void EmbeddingRequest::set_image_base64(::std::string&& value) {
  
  image_base64_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:facenet_server.EmbeddingRequest.image_base64)
}
#endif
inline void EmbeddingRequest::set_image_base64(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  image_base64_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:facenet_server.EmbeddingRequest.image_base64)
}
inline void EmbeddingRequest::set_image_base64(const char* value, size_t size) {
  
  image_base64_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:facenet_server.EmbeddingRequest.image_base64)
}
inline ::std::string* EmbeddingRequest::mutable_image_base64() {
  
  // @@protoc_insertion_point(field_mutable:facenet_server.EmbeddingRequest.image_base64)
  return image_base64_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* EmbeddingRequest::release_image_base64() {
  // @@protoc_insertion_point(field_release:facenet_server.EmbeddingRequest.image_base64)
  
  return image_base64_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void EmbeddingRequest::set_allocated_image_base64(::std::string* image_base64) {
  if (image_base64 != nullptr) {
    
  } else {
    
  }
  image_base64_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), image_base64);
  // @@protoc_insertion_point(field_set_allocated:facenet_server.EmbeddingRequest.image_base64)
}

// -------------------------------------------------------------------

// EmbeddingReply

// int32 error = 1;
inline void EmbeddingReply::clear_error() {
  error_ = 0;
}
inline ::google::protobuf::int32 EmbeddingReply::error() const {
  // @@protoc_insertion_point(field_get:facenet_server.EmbeddingReply.error)
  return error_;
}
inline void EmbeddingReply::set_error(::google::protobuf::int32 value) {
  
  error_ = value;
  // @@protoc_insertion_point(field_set:facenet_server.EmbeddingReply.error)
}

// repeated double dim = 2;
inline int EmbeddingReply::dim_size() const {
  return dim_.size();
}
inline void EmbeddingReply::clear_dim() {
  dim_.Clear();
}
inline double EmbeddingReply::dim(int index) const {
  // @@protoc_insertion_point(field_get:facenet_server.EmbeddingReply.dim)
  return dim_.Get(index);
}
inline void EmbeddingReply::set_dim(int index, double value) {
  dim_.Set(index, value);
  // @@protoc_insertion_point(field_set:facenet_server.EmbeddingReply.dim)
}
inline void EmbeddingReply::add_dim(double value) {
  dim_.Add(value);
  // @@protoc_insertion_point(field_add:facenet_server.EmbeddingReply.dim)
}
inline const ::google::protobuf::RepeatedField< double >&
EmbeddingReply::dim() const {
  // @@protoc_insertion_point(field_list:facenet_server.EmbeddingReply.dim)
  return dim_;
}
inline ::google::protobuf::RepeatedField< double >*
EmbeddingReply::mutable_dim() {
  // @@protoc_insertion_point(field_mutable_list:facenet_server.EmbeddingReply.dim)
  return &dim_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace facenet_server

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // PROTOBUF_INCLUDED_facenet_2eproto