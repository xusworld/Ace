#pragma once

#include <glog/logging.h>

#include <map>

#include "ace/ir/graph_generated.h"
#include "ir/op_generated.h"
#include "ir/op_option_generated.h"
#include "onnx.pb.h"
#include "utils.h"

namespace ace {
namespace model {

class OnnxNodeParser {
 public:
  OnnxNodeParser() = default;
  virtual ~OnnxNodeParser() = default;

  virtual void parse(ace::OpT*, const onnx::NodeProto*,
                     std::vector<const onnx::TensorProto*>);

  virtual ace::OpType type() { return OpType_Default; }

 protected:
  std::string onnxOpType_;
  std::string name_;
};

class OnnxNodeParserManager final {
 public:
  OnnxNodeParserManager() = default;
  ~OnnxNodeParserManager();

  // return onnx node parser manager instance
  static OnnxNodeParserManager* Global();
  // setter
  void Set(const std::string& name, OnnxNodeParser* parser);
  // getter
  OnnxNodeParser* Get(const std::string& name);
  std::vector<std::string> Names();

 private:
  static OnnxNodeParserManager* instance_;
  std::map<std::string, OnnxNodeParser*> name2parser_;
};

template <typename T>
class OnnxNodeParserRegister {
 public:
  OnnxNodeParserRegister() = delete;
  virtual ~OnnxNodeParserRegister() = default;

  OnnxNodeParserRegister(const std::string& name) {
    T* parser = new T;
    OnnxNodeParserManager* manager = OnnxNodeParserManager::Global();
    manager->Set(name, parser);
  }
};

#define DECLARE_ONNX_NODE_PARSER(clsname)                                   \
  class clsname : public OnnxNodeParser {                                   \
   public:                                                                  \
    clsname() = default;                                                    \
    virtual ~clsname() = default;                                           \
    virtual void parse(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,    \
                       std::vector<const onnx::TensorProto*> initializers); \
    virtual ace::OpType type();                                             \
  };

}  // namespace model
}  // namespace ace