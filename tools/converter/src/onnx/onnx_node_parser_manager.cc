#include <glog/logging.h>

#include "onnx_node_parser_manager.h"

namespace ace {
namespace parser {

OnnxNodeParserManager *OnnxNodeParserManager::instance_ = nullptr;

OnnxNodeParserManager::~OnnxNodeParserManager() {
  for (auto &item : name2parser_) {
    delete item.second;
  }
  name2parser_.clear();
}

OnnxNodeParserManager *OnnxNodeParserManager::Global() {
  if (instance_ == nullptr) {
    instance_ = new OnnxNodeParserManager;
    instance_->name2parser_.insert(
        std::make_pair("default", new OnnxNodeParser));
  }
  return instance_;
}

void OnnxNodeParserManager::Set(const std::string &name,
                                OnnxNodeParser *parser) {
  if (name2parser_.find(name) != name2parser_.end()) {
    LOG(INFO) << "Parser of " << name << " already registered.";
    return;
  }
  LOG(INFO) << "Set " << name << " onnx node parser";
  name2parser_.insert(std::make_pair(name, parser));
}

OnnxNodeParser *OnnxNodeParserManager::Get(const std::string &name) {
  if (name2parser_.find(name) == name2parser_.end()) {
    LOG(INFO) << "Parser of " << name << " not found, return a nullptr.";
    return name2parser_["default"];
  }
  return name2parser_[name];
}

std::vector<std::string> OnnxNodeParserManager::Names() {
  std::vector<std::string> names;
  for (auto item : name2parser_) {
    names.push_back(item.first);
  }
  return names;
}

static int32_t _limit(int64_t i64) {
  if (i64 > (int64_t)(1 << 30)) {
    return 1 << 30;
  }
  if (i64 < (int64_t)(-(1 << 30))) {
    return (-(1 << 30));
  }
  return i64;
}

void OnnxNodeParser::parse(
    ace::OpT *aceOp, const onnx::NodeProto *onnxNode,
    std::vector<const onnx::TensorProto *> initializers) {
  auto extra = new ExtraT;
  aceOp->main.type = OpParameter_Extra;
  aceOp->main.value = extra;

  extra->engine = "ONNX";
  extra->type = onnxNode->op_type();

  for (auto srcAttr : onnxNode->attribute()) {
    std::unique_ptr<AttributeT> attr(new AttributeT);
    attr->key = srcAttr.name();
    switch (srcAttr.type()) {
      case onnx::AttributeProto_AttributeType_INTS:
        attr->list.reset(new ListValueT);
        attr->list->i.resize(srcAttr.ints_size());
        for (int i = 0; i < srcAttr.ints_size(); ++i) {
          attr->list->i[i] = _limit(srcAttr.ints(i));
        }
        break;
      case onnx::AttributeProto_AttributeType_FLOATS:
        attr->list.reset(new ListValueT);
        attr->list->f.resize(srcAttr.floats_size());
        for (int i = 0; i < srcAttr.floats_size(); ++i) {
          attr->list->f[i] = srcAttr.floats(i);
        }
        break;
      case onnx::AttributeProto_AttributeType_TENSOR:
        attr->tensor.reset(OnnxTensorToBlob(&srcAttr.t()));
        break;
      default:
        break;
    }

    attr->i = _limit(srcAttr.i());
    attr->s = srcAttr.s();
    attr->f = srcAttr.f();
    extra->attr.emplace_back(std::move(attr));
  }
}

}  // namespace parser
}  // namespace ace