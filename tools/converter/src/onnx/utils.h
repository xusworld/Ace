#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "ace_generated.h"
#include "onnx.pb.h"

namespace ace {
namespace parser {

bool OnnxReadProtoFromBinary(const char* filepath,
                             google::protobuf::Message* message);

bool OnnxReadProtoFromBinary(const std::string& filepath,
                             google::protobuf::Message* message);

bool OnnxWriteProtoFromBinary(const char* filepath,
                              const google::protobuf::Message* message);

bool OnnxWriteProtoFromBinary(const std::string& filepath,
                              const google::protobuf::Message* message);

ace::DataType ToAceDataType(const onnx::TensorProto_DataType dtype);

ace::BlobT* OnnxTensorToBlob(const onnx::TensorProto* constantTp);
}  // namespace parser
}  // namespace ace
