// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_EXTRAINFO_ACE_H_
#define FLATBUFFERS_GENERATED_EXTRAINFO_ACE_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 2 &&
              FLATBUFFERS_VERSION_MINOR == 0 &&
              FLATBUFFERS_VERSION_REVISION == 8,
             "Non-compatible flatbuffers version included");

#include "tensor_generated.h"

namespace ace {

struct ExtraInfo;
struct ExtraInfoBuilder;
struct ExtraInfoT;

inline const flatbuffers::TypeTable *ExtraInfoTypeTable();

struct ExtraInfoT : public flatbuffers::NativeTable {
  typedef ExtraInfo TableType;
  std::vector<int8_t> buffer{};
  std::string name{};
  std::string version{};
};

struct ExtraInfo FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ExtraInfoT NativeTableType;
  typedef ExtraInfoBuilder Builder;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ExtraInfoTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BUFFER = 4,
    VT_NAME = 6,
    VT_VERSION = 8
  };
  const flatbuffers::Vector<int8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<int8_t> *>(VT_BUFFER);
  }
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::String *version() const {
    return GetPointer<const flatbuffers::String *>(VT_VERSION);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_VERSION) &&
           verifier.VerifyString(version()) &&
           verifier.EndTable();
  }
  ExtraInfoT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ExtraInfoT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<ExtraInfo> Pack(flatbuffers::FlatBufferBuilder &_fbb, const ExtraInfoT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct ExtraInfoBuilder {
  typedef ExtraInfo Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer) {
    fbb_.AddOffset(ExtraInfo::VT_BUFFER, buffer);
  }
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(ExtraInfo::VT_NAME, name);
  }
  void add_version(flatbuffers::Offset<flatbuffers::String> version) {
    fbb_.AddOffset(ExtraInfo::VT_VERSION, version);
  }
  explicit ExtraInfoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<ExtraInfo> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ExtraInfo>(end);
    return o;
  }
};

inline flatbuffers::Offset<ExtraInfo> CreateExtraInfo(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer = 0,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::String> version = 0) {
  ExtraInfoBuilder builder_(_fbb);
  builder_.add_version(version);
  builder_.add_name(name);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<ExtraInfo> CreateExtraInfoDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<int8_t> *buffer = nullptr,
    const char *name = nullptr,
    const char *version = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<int8_t>(*buffer) : 0;
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto version__ = version ? _fbb.CreateString(version) : 0;
  return ace::CreateExtraInfo(
      _fbb,
      buffer__,
      name__,
      version__);
}

flatbuffers::Offset<ExtraInfo> CreateExtraInfo(flatbuffers::FlatBufferBuilder &_fbb, const ExtraInfoT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline ExtraInfoT *ExtraInfo::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<ExtraInfoT>(new ExtraInfoT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void ExtraInfo::UnPackTo(ExtraInfoT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = buffer(); if (_e) { _o->buffer.resize(_e->size()); std::copy(_e->begin(), _e->end(), _o->buffer.begin()); } }
  { auto _e = name(); if (_e) _o->name = _e->str(); }
  { auto _e = version(); if (_e) _o->version = _e->str(); }
}

inline flatbuffers::Offset<ExtraInfo> ExtraInfo::Pack(flatbuffers::FlatBufferBuilder &_fbb, const ExtraInfoT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateExtraInfo(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<ExtraInfo> CreateExtraInfo(flatbuffers::FlatBufferBuilder &_fbb, const ExtraInfoT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const ExtraInfoT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _buffer = _o->buffer.size() ? _fbb.CreateVector(_o->buffer) : 0;
  auto _name = _o->name.empty() ? 0 : _fbb.CreateString(_o->name);
  auto _version = _o->version.empty() ? 0 : _fbb.CreateString(_o->version);
  return ace::CreateExtraInfo(
      _fbb,
      _buffer,
      _name,
      _version);
}

inline const flatbuffers::TypeTable *ExtraInfoTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 1, -1 },
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_STRING, 0, -1 }
  };
  static const char * const names[] = {
    "buffer",
    "name",
    "version"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 3, type_codes, nullptr, nullptr, nullptr, names
  };
  return &tt;
}

}  // namespace ace

#endif  // FLATBUFFERS_GENERATED_EXTRAINFO_ACE_H_