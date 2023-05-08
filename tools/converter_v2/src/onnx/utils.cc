#include <glog/logging.h>
#include <stdio.h>

#include <fstream>

#include "ir/types_generated.h"
#include "utils.h"
namespace ace {
namespace model {

static int32_t _limit(int64_t i64) {
  if (i64 > (int64_t)(1 << 30)) {
    return 1 << 30;
  }
  if (i64 < (int64_t)(-(1 << 30))) {
    return (-(1 << 30));
  }
  return i64;
}

ace::DataType ToAceDataType(const onnx::TensorProto_DataType dtype) {
  switch (dtype) {
    case onnx::TensorProto_DataType_FLOAT:
      return ace::DataType_FLOAT_32;
    case onnx::TensorProto_DataType_INT8:
      return ace::DataType_INT_8;
    case onnx::TensorProto_DataType_INT32:
      return ace::DataType_INT_32;
    case onnx::TensorProto_DataType_INT64:
      return ace::DataType_INT_64;
    case onnx::TensorProto_DataType_DOUBLE:
      return ace::DataType_FLOAT_32;
    case onnx::TensorProto_DataType_UINT8:
      return ace::DataType_UINT_8;
    default:
      return ace::DataType_NONE;
  }
}

ace::TensorT* OnnxTensorToBlob(const onnx::TensorProto* constantTp) {
  /*
  auto blob = new ace::TensorT;
  auto dataType = ToAceDataType(constantTp->data_type());

  blob->dataType = dataType;
  blob->dataFormat = ace::DATA_FORMAT_NCHW;

  size_t dimSize = constantTp->dims().size();
  blob->dims.resize(dimSize);
  size_t dataSize = 1;
  for (int i = 0; i < dimSize; ++i) {
    blob->dims[i] = constantTp->dims(i);
    dataSize = dataSize * constantTp->dims(i);
  }
  std::vector<int64_t> alignContent(
      (constantTp->raw_data().size() + sizeof(int64_t) - 1) / sizeof(int64_t));
  ::memcpy(alignContent.data(), constantTp->raw_data().data(),
           constantTp->raw_data().size());

  const void* tensor_content = (const void*)alignContent.data();

  switch (constantTp->data_type()) {
#define CASE_DATA_TYPE(src, dst)                        \
  case src:                                             \
    if (constantTp->dst##_data_size() != 0) {           \
      tensor_content = constantTp->dst##_data().data(); \
    }                                                   \
    break;
    CASE_DATA_TYPE(onnx::TensorProto_DataType_DOUBLE, double);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_INT64, int64);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_INT32, int32);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_FLOAT, float);
    default:
      break;
  }
  if (0 == dataSize) {
    // Empty blob
    return blob;
  }

  if (!tensor_content) {
    DLOG(FATAL) << "Convert no data, "
                   "Please make sure ";
  }

  switch (constantTp->data_type()) {
    case onnx::TensorProto_DataType_DOUBLE: {
      blob->float32s.resize(dataSize);
      auto source = (double*)tensor_content;

      for (int i = 0; i < dataSize; ++i) {
        blob->float32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT64: {
      blob->int32s.resize(dataSize);
      auto source = (int64_t*)tensor_content;

      for (int i = 0; i < dataSize; ++i) {
        blob->int32s[i] = _limit(source[i]);
      }
      break;
    }
    case onnx::TensorProto_DataType_INT32: {
      auto source = (int32_t*)tensor_content;
      blob->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT16: {
      auto source = (uint16_t*)tensor_content;
      blob->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT16: {
      auto source = (int16_t*)tensor_content;
      blob->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_BOOL: {
      auto source = (bool*)tensor_content;
      blob->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT8: {
      auto source = (int8_t*)tensor_content;
      blob->int8s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->int8s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT8: {
      auto source = (uint8_t*)tensor_content;
      blob->uint8s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->uint8s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      float* tempFloatData = (float*)tensor_content;
      blob->float32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        blob->float32s[i] = tempFloatData[i];
      }
      break;
    }
    default: {
      DLOG(FATAL) << "Don't support " << constantTp->data_type();
      break;
    }
  }

  return blob;
*/
  return nullptr;
}

}  // namespace model
}  // namespace ace