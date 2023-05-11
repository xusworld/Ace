#include <glog/logging.h>
#include <stdio.h>

#include <fstream>

#include "ir/tensor_generated.h"
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

ace::DataType OnnxDataTypeToAceDataType(
    const onnx::TensorProto_DataType dtype) {
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

ace::TensorT* OnnxTensorToAceTensor(const onnx::TensorProto* constantTp) {
  ace::TensorT* tensor = new ace::TensorT;
  tensor->name = "";
  tensor->dtype = OnnxDataTypeToAceDataType(constantTp->data_type());
  tensor->dformat = ace::DataFormat_NCHW;

  // Set TensorShape
  size_t dimSize = constantTp->dims().size();
  auto shape = new ace::TensorShapeT;
  shape->dims.resize(dimSize);
  size_t dataSize = 1;
  for (int i = 0; i < dimSize; ++i) {
    shape->dims[i] = constantTp->dims(i);
    dataSize = dataSize * constantTp->dims(i);
  }
  tensor->shape.reset(shape);

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
    return tensor;
  }

  if (!tensor_content) {
    DLOG(FATAL) << "Convert no data, Please make sure ";
  }

  auto cache_data = new ace::CacheDataT;
  switch (constantTp->data_type()) {
    case onnx::TensorProto_DataType_UINT8: {
      auto source = (uint8_t*)tensor_content;
      cache_data->u.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        cache_data->u[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT8: {
      auto source = (int8_t*)tensor_content;
      cache_data->c.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        cache_data->c[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT32: {
      auto source = (int32_t*)tensor_content;
      cache_data->i.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        cache_data->c[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      float* tempFloatData = (float*)tensor_content;
      cache_data->i.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        cache_data->i[i] = tempFloatData[i];
      }
      break;
    }
    default: {
      LOG(FATAL) << "Don't support " << constantTp->data_type();
      break;
    }
  }

  return tensor;
}

}  // namespace model
}  // namespace ace