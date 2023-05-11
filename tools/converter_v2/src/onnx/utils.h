#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "ace/ir/graph_generated.h"
#include "onnx.pb.h"

namespace ace {
namespace model {

ace::DataType OnnxDataTypeToAceDataType(const onnx::TensorProto_DataType dtype);

ace::TensorT* OnnxTensorToAceTensor(const onnx::TensorProto* constantTp);

}  // namespace model
}  // namespace ace
