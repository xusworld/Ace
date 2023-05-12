// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_GRAPH_ACE_H_
#define FLATBUFFERS_GENERATED_GRAPH_ACE_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 2 &&
              FLATBUFFERS_VERSION_MINOR == 0 &&
              FLATBUFFERS_VERSION_REVISION == 8,
             "Non-compatible flatbuffers version included");

#include "op_generated.h"
#include "tensor_generated.h"
#include "types_generated.h"

namespace ace {

struct Version;
struct VersionBuilder;
struct VersionT;

struct SubGraphProto;
struct SubGraphProtoBuilder;
struct SubGraphProtoT;

struct GraphProto;
struct GraphProtoBuilder;
struct GraphProtoT;

inline const flatbuffers::TypeTable *VersionTypeTable();

inline const flatbuffers::TypeTable *SubGraphProtoTypeTable();

inline const flatbuffers::TypeTable *GraphProtoTypeTable();

struct VersionT : public flatbuffers::NativeTable {
  typedef Version TableType;
  int32_t major = 0;
  int32_t minor = 0;
  int32_t patch = 0;
  int32_t version = 0;
};

struct Version FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef VersionT NativeTableType;
  typedef VersionBuilder Builder;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return VersionTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_MAJOR = 4,
    VT_MINOR = 6,
    VT_PATCH = 8,
    VT_VERSION = 10
  };
  int32_t major() const {
    return GetField<int32_t>(VT_MAJOR, 0);
  }
  int32_t minor() const {
    return GetField<int32_t>(VT_MINOR, 0);
  }
  int32_t patch() const {
    return GetField<int32_t>(VT_PATCH, 0);
  }
  int32_t version() const {
    return GetField<int32_t>(VT_VERSION, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_MAJOR, 4) &&
           VerifyField<int32_t>(verifier, VT_MINOR, 4) &&
           VerifyField<int32_t>(verifier, VT_PATCH, 4) &&
           VerifyField<int32_t>(verifier, VT_VERSION, 4) &&
           verifier.EndTable();
  }
  VersionT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(VersionT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Version> Pack(flatbuffers::FlatBufferBuilder &_fbb, const VersionT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct VersionBuilder {
  typedef Version Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_major(int32_t major) {
    fbb_.AddElement<int32_t>(Version::VT_MAJOR, major, 0);
  }
  void add_minor(int32_t minor) {
    fbb_.AddElement<int32_t>(Version::VT_MINOR, minor, 0);
  }
  void add_patch(int32_t patch) {
    fbb_.AddElement<int32_t>(Version::VT_PATCH, patch, 0);
  }
  void add_version(int32_t version) {
    fbb_.AddElement<int32_t>(Version::VT_VERSION, version, 0);
  }
  explicit VersionBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<Version> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Version>(end);
    return o;
  }
};

inline flatbuffers::Offset<Version> CreateVersion(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t major = 0,
    int32_t minor = 0,
    int32_t patch = 0,
    int32_t version = 0) {
  VersionBuilder builder_(_fbb);
  builder_.add_version(version);
  builder_.add_patch(patch);
  builder_.add_minor(minor);
  builder_.add_major(major);
  return builder_.Finish();
}

flatbuffers::Offset<Version> CreateVersion(flatbuffers::FlatBufferBuilder &_fbb, const VersionT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct SubGraphProtoT : public flatbuffers::NativeTable {
  typedef SubGraphProto TableType;
  std::string name{};
  std::vector<int32_t> inputs{};
  std::vector<int32_t> outputs{};
  std::vector<std::string> tensors{};
  std::vector<std::unique_ptr<ace::OpT>> nodes{};
  SubGraphProtoT() = default;
  SubGraphProtoT(const SubGraphProtoT &o);
  SubGraphProtoT(SubGraphProtoT&&) FLATBUFFERS_NOEXCEPT = default;
  SubGraphProtoT &operator=(SubGraphProtoT o) FLATBUFFERS_NOEXCEPT;
};

struct SubGraphProto FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef SubGraphProtoT NativeTableType;
  typedef SubGraphProtoBuilder Builder;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return SubGraphProtoTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_INPUTS = 6,
    VT_OUTPUTS = 8,
    VT_TENSORS = 10,
    VT_NODES = 12
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::Vector<int32_t> *inputs() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_INPUTS);
  }
  const flatbuffers::Vector<int32_t> *outputs() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_OUTPUTS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *tensors() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_TENSORS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<ace::Op>> *nodes() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<ace::Op>> *>(VT_NODES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_INPUTS) &&
           verifier.VerifyVector(inputs()) &&
           VerifyOffset(verifier, VT_OUTPUTS) &&
           verifier.VerifyVector(outputs()) &&
           VerifyOffset(verifier, VT_TENSORS) &&
           verifier.VerifyVector(tensors()) &&
           verifier.VerifyVectorOfStrings(tensors()) &&
           VerifyOffset(verifier, VT_NODES) &&
           verifier.VerifyVector(nodes()) &&
           verifier.VerifyVectorOfTables(nodes()) &&
           verifier.EndTable();
  }
  SubGraphProtoT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(SubGraphProtoT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<SubGraphProto> Pack(flatbuffers::FlatBufferBuilder &_fbb, const SubGraphProtoT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct SubGraphProtoBuilder {
  typedef SubGraphProto Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(SubGraphProto::VT_NAME, name);
  }
  void add_inputs(flatbuffers::Offset<flatbuffers::Vector<int32_t>> inputs) {
    fbb_.AddOffset(SubGraphProto::VT_INPUTS, inputs);
  }
  void add_outputs(flatbuffers::Offset<flatbuffers::Vector<int32_t>> outputs) {
    fbb_.AddOffset(SubGraphProto::VT_OUTPUTS, outputs);
  }
  void add_tensors(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensors) {
    fbb_.AddOffset(SubGraphProto::VT_TENSORS, tensors);
  }
  void add_nodes(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<ace::Op>>> nodes) {
    fbb_.AddOffset(SubGraphProto::VT_NODES, nodes);
  }
  explicit SubGraphProtoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<SubGraphProto> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<SubGraphProto>(end);
    return o;
  }
};

inline flatbuffers::Offset<SubGraphProto> CreateSubGraphProto(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> inputs = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> outputs = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensors = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<ace::Op>>> nodes = 0) {
  SubGraphProtoBuilder builder_(_fbb);
  builder_.add_nodes(nodes);
  builder_.add_tensors(tensors);
  builder_.add_outputs(outputs);
  builder_.add_inputs(inputs);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<SubGraphProto> CreateSubGraphProtoDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const std::vector<int32_t> *inputs = nullptr,
    const std::vector<int32_t> *outputs = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *tensors = nullptr,
    const std::vector<flatbuffers::Offset<ace::Op>> *nodes = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto inputs__ = inputs ? _fbb.CreateVector<int32_t>(*inputs) : 0;
  auto outputs__ = outputs ? _fbb.CreateVector<int32_t>(*outputs) : 0;
  auto tensors__ = tensors ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*tensors) : 0;
  auto nodes__ = nodes ? _fbb.CreateVector<flatbuffers::Offset<ace::Op>>(*nodes) : 0;
  return ace::CreateSubGraphProto(
      _fbb,
      name__,
      inputs__,
      outputs__,
      tensors__,
      nodes__);
}

flatbuffers::Offset<SubGraphProto> CreateSubGraphProto(flatbuffers::FlatBufferBuilder &_fbb, const SubGraphProtoT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct GraphProtoT : public flatbuffers::NativeTable {
  typedef GraphProto TableType;
  std::string name{};
  std::vector<std::unique_ptr<ace::OpT>> ops{};
  std::vector<int32_t> inputs{};
  std::vector<int32_t> outputs{};
  std::vector<std::string> inputs_name{};
  std::vector<std::string> outputs_name{};
  ace::FrontendFramework ir = ace::FrontendFramework_ONNX;
  std::vector<std::string> tensors{};
  std::vector<std::unique_ptr<ace::SubGraphProtoT>> subgraphs{};
  std::unique_ptr<ace::VersionT> version{};
  std::string desc{};
  GraphProtoT() = default;
  GraphProtoT(const GraphProtoT &o);
  GraphProtoT(GraphProtoT&&) FLATBUFFERS_NOEXCEPT = default;
  GraphProtoT &operator=(GraphProtoT o) FLATBUFFERS_NOEXCEPT;
};

struct GraphProto FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef GraphProtoT NativeTableType;
  typedef GraphProtoBuilder Builder;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return GraphProtoTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_OPS = 6,
    VT_INPUTS = 8,
    VT_OUTPUTS = 10,
    VT_INPUTS_NAME = 12,
    VT_OUTPUTS_NAME = 14,
    VT_IR = 16,
    VT_TENSORS = 18,
    VT_SUBGRAPHS = 20,
    VT_VERSION = 22,
    VT_DESC = 24
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::Vector<flatbuffers::Offset<ace::Op>> *ops() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<ace::Op>> *>(VT_OPS);
  }
  const flatbuffers::Vector<int32_t> *inputs() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_INPUTS);
  }
  const flatbuffers::Vector<int32_t> *outputs() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_OUTPUTS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *inputs_name() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_INPUTS_NAME);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *outputs_name() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_OUTPUTS_NAME);
  }
  ace::FrontendFramework ir() const {
    return static_cast<ace::FrontendFramework>(GetField<int8_t>(VT_IR, 1));
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *tensors() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_TENSORS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<ace::SubGraphProto>> *subgraphs() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<ace::SubGraphProto>> *>(VT_SUBGRAPHS);
  }
  const ace::Version *version() const {
    return GetPointer<const ace::Version *>(VT_VERSION);
  }
  const flatbuffers::String *desc() const {
    return GetPointer<const flatbuffers::String *>(VT_DESC);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_OPS) &&
           verifier.VerifyVector(ops()) &&
           verifier.VerifyVectorOfTables(ops()) &&
           VerifyOffset(verifier, VT_INPUTS) &&
           verifier.VerifyVector(inputs()) &&
           VerifyOffset(verifier, VT_OUTPUTS) &&
           verifier.VerifyVector(outputs()) &&
           VerifyOffset(verifier, VT_INPUTS_NAME) &&
           verifier.VerifyVector(inputs_name()) &&
           verifier.VerifyVectorOfStrings(inputs_name()) &&
           VerifyOffset(verifier, VT_OUTPUTS_NAME) &&
           verifier.VerifyVector(outputs_name()) &&
           verifier.VerifyVectorOfStrings(outputs_name()) &&
           VerifyField<int8_t>(verifier, VT_IR, 1) &&
           VerifyOffset(verifier, VT_TENSORS) &&
           verifier.VerifyVector(tensors()) &&
           verifier.VerifyVectorOfStrings(tensors()) &&
           VerifyOffset(verifier, VT_SUBGRAPHS) &&
           verifier.VerifyVector(subgraphs()) &&
           verifier.VerifyVectorOfTables(subgraphs()) &&
           VerifyOffset(verifier, VT_VERSION) &&
           verifier.VerifyTable(version()) &&
           VerifyOffset(verifier, VT_DESC) &&
           verifier.VerifyString(desc()) &&
           verifier.EndTable();
  }
  GraphProtoT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(GraphProtoT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<GraphProto> Pack(flatbuffers::FlatBufferBuilder &_fbb, const GraphProtoT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct GraphProtoBuilder {
  typedef GraphProto Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(GraphProto::VT_NAME, name);
  }
  void add_ops(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<ace::Op>>> ops) {
    fbb_.AddOffset(GraphProto::VT_OPS, ops);
  }
  void add_inputs(flatbuffers::Offset<flatbuffers::Vector<int32_t>> inputs) {
    fbb_.AddOffset(GraphProto::VT_INPUTS, inputs);
  }
  void add_outputs(flatbuffers::Offset<flatbuffers::Vector<int32_t>> outputs) {
    fbb_.AddOffset(GraphProto::VT_OUTPUTS, outputs);
  }
  void add_inputs_name(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> inputs_name) {
    fbb_.AddOffset(GraphProto::VT_INPUTS_NAME, inputs_name);
  }
  void add_outputs_name(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> outputs_name) {
    fbb_.AddOffset(GraphProto::VT_OUTPUTS_NAME, outputs_name);
  }
  void add_ir(ace::FrontendFramework ir) {
    fbb_.AddElement<int8_t>(GraphProto::VT_IR, static_cast<int8_t>(ir), 1);
  }
  void add_tensors(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensors) {
    fbb_.AddOffset(GraphProto::VT_TENSORS, tensors);
  }
  void add_subgraphs(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<ace::SubGraphProto>>> subgraphs) {
    fbb_.AddOffset(GraphProto::VT_SUBGRAPHS, subgraphs);
  }
  void add_version(flatbuffers::Offset<ace::Version> version) {
    fbb_.AddOffset(GraphProto::VT_VERSION, version);
  }
  void add_desc(flatbuffers::Offset<flatbuffers::String> desc) {
    fbb_.AddOffset(GraphProto::VT_DESC, desc);
  }
  explicit GraphProtoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<GraphProto> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<GraphProto>(end);
    return o;
  }
};

inline flatbuffers::Offset<GraphProto> CreateGraphProto(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<ace::Op>>> ops = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> inputs = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> outputs = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> inputs_name = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> outputs_name = 0,
    ace::FrontendFramework ir = ace::FrontendFramework_ONNX,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensors = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<ace::SubGraphProto>>> subgraphs = 0,
    flatbuffers::Offset<ace::Version> version = 0,
    flatbuffers::Offset<flatbuffers::String> desc = 0) {
  GraphProtoBuilder builder_(_fbb);
  builder_.add_desc(desc);
  builder_.add_version(version);
  builder_.add_subgraphs(subgraphs);
  builder_.add_tensors(tensors);
  builder_.add_outputs_name(outputs_name);
  builder_.add_inputs_name(inputs_name);
  builder_.add_outputs(outputs);
  builder_.add_inputs(inputs);
  builder_.add_ops(ops);
  builder_.add_name(name);
  builder_.add_ir(ir);
  return builder_.Finish();
}

inline flatbuffers::Offset<GraphProto> CreateGraphProtoDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const std::vector<flatbuffers::Offset<ace::Op>> *ops = nullptr,
    const std::vector<int32_t> *inputs = nullptr,
    const std::vector<int32_t> *outputs = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *inputs_name = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *outputs_name = nullptr,
    ace::FrontendFramework ir = ace::FrontendFramework_ONNX,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *tensors = nullptr,
    const std::vector<flatbuffers::Offset<ace::SubGraphProto>> *subgraphs = nullptr,
    flatbuffers::Offset<ace::Version> version = 0,
    const char *desc = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto ops__ = ops ? _fbb.CreateVector<flatbuffers::Offset<ace::Op>>(*ops) : 0;
  auto inputs__ = inputs ? _fbb.CreateVector<int32_t>(*inputs) : 0;
  auto outputs__ = outputs ? _fbb.CreateVector<int32_t>(*outputs) : 0;
  auto inputs_name__ = inputs_name ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*inputs_name) : 0;
  auto outputs_name__ = outputs_name ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*outputs_name) : 0;
  auto tensors__ = tensors ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*tensors) : 0;
  auto subgraphs__ = subgraphs ? _fbb.CreateVector<flatbuffers::Offset<ace::SubGraphProto>>(*subgraphs) : 0;
  auto desc__ = desc ? _fbb.CreateString(desc) : 0;
  return ace::CreateGraphProto(
      _fbb,
      name__,
      ops__,
      inputs__,
      outputs__,
      inputs_name__,
      outputs_name__,
      ir,
      tensors__,
      subgraphs__,
      version,
      desc__);
}

flatbuffers::Offset<GraphProto> CreateGraphProto(flatbuffers::FlatBufferBuilder &_fbb, const GraphProtoT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline VersionT *Version::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<VersionT>(new VersionT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void Version::UnPackTo(VersionT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = major(); _o->major = _e; }
  { auto _e = minor(); _o->minor = _e; }
  { auto _e = patch(); _o->patch = _e; }
  { auto _e = version(); _o->version = _e; }
}

inline flatbuffers::Offset<Version> Version::Pack(flatbuffers::FlatBufferBuilder &_fbb, const VersionT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateVersion(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<Version> CreateVersion(flatbuffers::FlatBufferBuilder &_fbb, const VersionT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const VersionT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _major = _o->major;
  auto _minor = _o->minor;
  auto _patch = _o->patch;
  auto _version = _o->version;
  return ace::CreateVersion(
      _fbb,
      _major,
      _minor,
      _patch,
      _version);
}

inline SubGraphProtoT::SubGraphProtoT(const SubGraphProtoT &o)
      : name(o.name),
        inputs(o.inputs),
        outputs(o.outputs),
        tensors(o.tensors) {
  nodes.reserve(o.nodes.size());
  for (const auto &nodes_ : o.nodes) { nodes.emplace_back((nodes_) ? new ace::OpT(*nodes_) : nullptr); }
}

inline SubGraphProtoT &SubGraphProtoT::operator=(SubGraphProtoT o) FLATBUFFERS_NOEXCEPT {
  std::swap(name, o.name);
  std::swap(inputs, o.inputs);
  std::swap(outputs, o.outputs);
  std::swap(tensors, o.tensors);
  std::swap(nodes, o.nodes);
  return *this;
}

inline SubGraphProtoT *SubGraphProto::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<SubGraphProtoT>(new SubGraphProtoT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void SubGraphProto::UnPackTo(SubGraphProtoT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = name(); if (_e) _o->name = _e->str(); }
  { auto _e = inputs(); if (_e) { _o->inputs.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->inputs[_i] = _e->Get(_i); } } }
  { auto _e = outputs(); if (_e) { _o->outputs.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->outputs[_i] = _e->Get(_i); } } }
  { auto _e = tensors(); if (_e) { _o->tensors.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->tensors[_i] = _e->Get(_i)->str(); } } }
  { auto _e = nodes(); if (_e) { _o->nodes.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->nodes[_i] = std::unique_ptr<ace::OpT>(_e->Get(_i)->UnPack(_resolver)); } } }
}

inline flatbuffers::Offset<SubGraphProto> SubGraphProto::Pack(flatbuffers::FlatBufferBuilder &_fbb, const SubGraphProtoT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateSubGraphProto(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<SubGraphProto> CreateSubGraphProto(flatbuffers::FlatBufferBuilder &_fbb, const SubGraphProtoT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const SubGraphProtoT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _name = _o->name.empty() ? 0 : _fbb.CreateString(_o->name);
  auto _inputs = _o->inputs.size() ? _fbb.CreateVector(_o->inputs) : 0;
  auto _outputs = _o->outputs.size() ? _fbb.CreateVector(_o->outputs) : 0;
  auto _tensors = _o->tensors.size() ? _fbb.CreateVectorOfStrings(_o->tensors) : 0;
  auto _nodes = _o->nodes.size() ? _fbb.CreateVector<flatbuffers::Offset<ace::Op>> (_o->nodes.size(), [](size_t i, _VectorArgs *__va) { return CreateOp(*__va->__fbb, __va->__o->nodes[i].get(), __va->__rehasher); }, &_va ) : 0;
  return ace::CreateSubGraphProto(
      _fbb,
      _name,
      _inputs,
      _outputs,
      _tensors,
      _nodes);
}

inline GraphProtoT::GraphProtoT(const GraphProtoT &o)
      : name(o.name),
        inputs(o.inputs),
        outputs(o.outputs),
        inputs_name(o.inputs_name),
        outputs_name(o.outputs_name),
        ir(o.ir),
        tensors(o.tensors),
        version((o.version) ? new ace::VersionT(*o.version) : nullptr),
        desc(o.desc) {
  ops.reserve(o.ops.size());
  for (const auto &ops_ : o.ops) { ops.emplace_back((ops_) ? new ace::OpT(*ops_) : nullptr); }
  subgraphs.reserve(o.subgraphs.size());
  for (const auto &subgraphs_ : o.subgraphs) { subgraphs.emplace_back((subgraphs_) ? new ace::SubGraphProtoT(*subgraphs_) : nullptr); }
}

inline GraphProtoT &GraphProtoT::operator=(GraphProtoT o) FLATBUFFERS_NOEXCEPT {
  std::swap(name, o.name);
  std::swap(ops, o.ops);
  std::swap(inputs, o.inputs);
  std::swap(outputs, o.outputs);
  std::swap(inputs_name, o.inputs_name);
  std::swap(outputs_name, o.outputs_name);
  std::swap(ir, o.ir);
  std::swap(tensors, o.tensors);
  std::swap(subgraphs, o.subgraphs);
  std::swap(version, o.version);
  std::swap(desc, o.desc);
  return *this;
}

inline GraphProtoT *GraphProto::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<GraphProtoT>(new GraphProtoT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void GraphProto::UnPackTo(GraphProtoT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = name(); if (_e) _o->name = _e->str(); }
  { auto _e = ops(); if (_e) { _o->ops.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->ops[_i] = std::unique_ptr<ace::OpT>(_e->Get(_i)->UnPack(_resolver)); } } }
  { auto _e = inputs(); if (_e) { _o->inputs.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->inputs[_i] = _e->Get(_i); } } }
  { auto _e = outputs(); if (_e) { _o->outputs.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->outputs[_i] = _e->Get(_i); } } }
  { auto _e = inputs_name(); if (_e) { _o->inputs_name.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->inputs_name[_i] = _e->Get(_i)->str(); } } }
  { auto _e = outputs_name(); if (_e) { _o->outputs_name.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->outputs_name[_i] = _e->Get(_i)->str(); } } }
  { auto _e = ir(); _o->ir = _e; }
  { auto _e = tensors(); if (_e) { _o->tensors.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->tensors[_i] = _e->Get(_i)->str(); } } }
  { auto _e = subgraphs(); if (_e) { _o->subgraphs.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->subgraphs[_i] = std::unique_ptr<ace::SubGraphProtoT>(_e->Get(_i)->UnPack(_resolver)); } } }
  { auto _e = version(); if (_e) _o->version = std::unique_ptr<ace::VersionT>(_e->UnPack(_resolver)); }
  { auto _e = desc(); if (_e) _o->desc = _e->str(); }
}

inline flatbuffers::Offset<GraphProto> GraphProto::Pack(flatbuffers::FlatBufferBuilder &_fbb, const GraphProtoT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateGraphProto(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<GraphProto> CreateGraphProto(flatbuffers::FlatBufferBuilder &_fbb, const GraphProtoT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const GraphProtoT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _name = _o->name.empty() ? 0 : _fbb.CreateString(_o->name);
  auto _ops = _o->ops.size() ? _fbb.CreateVector<flatbuffers::Offset<ace::Op>> (_o->ops.size(), [](size_t i, _VectorArgs *__va) { return CreateOp(*__va->__fbb, __va->__o->ops[i].get(), __va->__rehasher); }, &_va ) : 0;
  auto _inputs = _o->inputs.size() ? _fbb.CreateVector(_o->inputs) : 0;
  auto _outputs = _o->outputs.size() ? _fbb.CreateVector(_o->outputs) : 0;
  auto _inputs_name = _o->inputs_name.size() ? _fbb.CreateVectorOfStrings(_o->inputs_name) : 0;
  auto _outputs_name = _o->outputs_name.size() ? _fbb.CreateVectorOfStrings(_o->outputs_name) : 0;
  auto _ir = _o->ir;
  auto _tensors = _o->tensors.size() ? _fbb.CreateVectorOfStrings(_o->tensors) : 0;
  auto _subgraphs = _o->subgraphs.size() ? _fbb.CreateVector<flatbuffers::Offset<ace::SubGraphProto>> (_o->subgraphs.size(), [](size_t i, _VectorArgs *__va) { return CreateSubGraphProto(*__va->__fbb, __va->__o->subgraphs[i].get(), __va->__rehasher); }, &_va ) : 0;
  auto _version = _o->version ? CreateVersion(_fbb, _o->version.get(), _rehasher) : 0;
  auto _desc = _o->desc.empty() ? 0 : _fbb.CreateString(_o->desc);
  return ace::CreateGraphProto(
      _fbb,
      _name,
      _ops,
      _inputs,
      _outputs,
      _inputs_name,
      _outputs_name,
      _ir,
      _tensors,
      _subgraphs,
      _version,
      _desc);
}

inline const flatbuffers::TypeTable *VersionTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_INT, 0, -1 },
    { flatbuffers::ET_INT, 0, -1 },
    { flatbuffers::ET_INT, 0, -1 },
    { flatbuffers::ET_INT, 0, -1 }
  };
  static const char * const names[] = {
    "major",
    "minor",
    "patch",
    "version"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 4, type_codes, nullptr, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *SubGraphProtoTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_STRING, 1, -1 },
    { flatbuffers::ET_SEQUENCE, 1, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::OpTypeTable
  };
  static const char * const names[] = {
    "name",
    "inputs",
    "outputs",
    "tensors",
    "nodes"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 5, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *GraphProtoTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_SEQUENCE, 1, 0 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_STRING, 1, -1 },
    { flatbuffers::ET_STRING, 1, -1 },
    { flatbuffers::ET_CHAR, 0, 1 },
    { flatbuffers::ET_STRING, 1, -1 },
    { flatbuffers::ET_SEQUENCE, 1, 2 },
    { flatbuffers::ET_SEQUENCE, 0, 3 },
    { flatbuffers::ET_STRING, 0, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::OpTypeTable,
    ace::FrontendFrameworkTypeTable,
    ace::SubGraphProtoTypeTable,
    ace::VersionTypeTable
  };
  static const char * const names[] = {
    "name",
    "ops",
    "inputs",
    "outputs",
    "inputs_name",
    "outputs_name",
    "ir",
    "tensors",
    "subgraphs",
    "version",
    "desc"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 11, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const ace::GraphProto *GetGraphProto(const void *buf) {
  return flatbuffers::GetRoot<ace::GraphProto>(buf);
}

inline const ace::GraphProto *GetSizePrefixedGraphProto(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<ace::GraphProto>(buf);
}

inline bool VerifyGraphProtoBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<ace::GraphProto>(nullptr);
}

inline bool VerifySizePrefixedGraphProtoBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<ace::GraphProto>(nullptr);
}

inline void FinishGraphProtoBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<ace::GraphProto> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedGraphProtoBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<ace::GraphProto> root) {
  fbb.FinishSizePrefixed(root);
}

inline std::unique_ptr<ace::GraphProtoT> UnPackGraphProto(
    const void *buf,
    const flatbuffers::resolver_function_t *res = nullptr) {
  return std::unique_ptr<ace::GraphProtoT>(GetGraphProto(buf)->UnPack(res));
}

inline std::unique_ptr<ace::GraphProtoT> UnPackSizePrefixedGraphProto(
    const void *buf,
    const flatbuffers::resolver_function_t *res = nullptr) {
  return std::unique_ptr<ace::GraphProtoT>(GetSizePrefixedGraphProto(buf)->UnPack(res));
}

}  // namespace ace

#endif  // FLATBUFFERS_GENERATED_GRAPH_ACE_H_