//
//  tfOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdlib.h>

#include "OpCount.hpp"
#include "tfOpConverter.hpp"
using namespace tars;
#define FUNCTION(dstType, srcType, contentType)                            \
  static void _##dstType##srcType##_##contentType(                         \
      BlobT* dst, const ::tensorflow::TensorProto& tensor, int dataSize) { \
    dst->dstType.resize(dataSize);                                         \
    if (tensor.srcType##_size() == 1) {                                    \
      for (int i = 0; i < dataSize; ++i) {                                 \
        dst->dstType[i] = tensor.srcType(0);                               \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    if (tensor.srcType().empty()) {                                        \
      contentType* source = (contentType*)tensor.tensor_content().data();  \
      for (int i = 0; i < dataSize; ++i) {                                 \
        dst->dstType[i] = source[i];                                       \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    for (int i = 0; i < dataSize; ++i) {                                   \
      dst->dstType[i] = tensor.srcType(i);                                 \
    }                                                                      \
  }

FUNCTION(float32s, double_val, double);
FUNCTION(float32s, float_val, float);
FUNCTION(int32s, int_val, int32_t);
FUNCTION(int32s, int64_val, int64_t);
FUNCTION(uint8s, int64_val, uint8_t);
FUNCTION(int8s, int64_val, int8_t);
FUNCTION(int32s, bool_val, uint8_t);
FUNCTION(strings, string_val, uint8_t);

typedef void (*proc)(BlobT* dst, const ::tensorflow::TensorProto& tensor,
                     int dataSize);
void tfOpConverter::convertTensorToBlob(
    tars::BlobT* parameter, const ::tensorflow::TensorProto& tensor) {
  parameter->dataFormat = tars::MNN_DATA_FORMAT_NHWC;
  tars::DataType dataType = tars::DataType_DT_INVALID;
  dataType = (tars::DataType)tensor.dtype();

  // origin type in tensorflow, mnn's data type, tensor_content's data type
  std::map<tars::DataType, std::pair<tars::DataType, proc> > supporting{
      {DataType_DT_DOUBLE, {DataType_DT_FLOAT, _float32sdouble_val_double}},
      {DataType_DT_FLOAT, {DataType_DT_FLOAT, _float32sfloat_val_float}},
      {DataType_DT_INT32, {DataType_DT_INT32, _int32sint_val_int32_t}},
      {DataType_DT_INT64, {DataType_DT_INT32, _int32sint64_val_int64_t}},
      {DataType_DT_INT8, {DataType_DT_INT8, _int8sint64_val_int8_t}},
      {DataType_DT_UINT8, {DataType_DT_UINT8, _uint8sint64_val_uint8_t}},
      {DataType_DT_BOOL, {DataType_DT_INT32, _int32sbool_val_uint8_t}},
      {DataType_DT_STRING, {DataType_DT_STRING, _stringsstring_val_uint8_t}},
  };
  bool isSupport = supporting.find(dataType) != supporting.end();
  CHECK(isSupport) << "Const Data Type Not Supported!!!==> " << dataType;
  CHECK(dataType <= tars::DataType_MAX)
      << "Const Data Type Not Supported!!!==> " << dataType;
  auto convert = supporting[dataType];
  parameter->dataType = convert.first;
  size_t dimSize = tensor.tensor_shape().dim_size();
  parameter->dims.resize(dimSize);
  size_t dataSize = 1;
  for (int i = 0; i < dimSize; i++) {
    dataSize = dataSize * tensor.tensor_shape().dim(i).size();
    parameter->dims[i] = tensor.tensor_shape().dim(i).size();
  }
  convert.second(parameter, tensor, dataSize);
}

tfOpConverterSuit* tfOpConverterSuit::global = nullptr;
class DefaultTfOpConverter : public tfOpConverter {
 public:
  virtual void run(tars::OpT* dstOp, TmpNode* srcNode) override {
    dstOp->main.value = new tars::ExtraT;
    dstOp->main.AsExtra()->engine = "Tensorflow";
    dstOp->main.AsExtra()->type = srcNode->opType;
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr =
        srcNode->tfNode->attr();
    for (auto iter = attr.begin(); iter != attr.end(); iter++) {
      auto attrExtr =
          ConvertTfAttribute(iter->first /*attr key*/, iter->second /*attr*/);
      dstOp->main.AsExtra()->attr.emplace_back(std::move(attrExtr));
    }
  }
  virtual tars::OpParameter type() override { return tars::OpParameter_Extra; }
  virtual tars::OpType opType() override { return tars::OpType_Extra; }

 private:
  std::unique_ptr<tars::AttributeT> ConvertTfAttribute(
      const std::string& attr_name, const tensorflow::AttrValue& tf_attr) const;
};

std::unique_ptr<tars::AttributeT> DefaultTfOpConverter::ConvertTfAttribute(
    const std::string& attr_name, const tensorflow::AttrValue& tf_attr) const {
  std::unique_ptr<tars::AttributeT> attrExtr(new tars::AttributeT);
  attrExtr->key = attr_name;
  attrExtr->s = tf_attr.s();
  attrExtr->f = tf_attr.f();
  attrExtr->i = (int)tf_attr.i();
  attrExtr->b = tf_attr.b();
  if (tf_attr.has_tensor()) {
    attrExtr->tensor.reset(new BlobT);
    convertTensorToBlob(attrExtr->tensor.get(), tf_attr.tensor());
  }
  attrExtr->type = (tars::DataType)tf_attr.type();
  if (tf_attr.has_list()) {
    auto& listValue = tf_attr.list();
    attrExtr->list.reset(new tars::ListValueT);
    for (int j = 0; j < listValue.s_size(); ++j) {
      attrExtr->list->s.push_back(listValue.s(j));
    }
    for (int j = 0; j < listValue.b_size(); ++j) {
      attrExtr->list->b.push_back(listValue.b(j));
    }
    for (int j = 0; j < listValue.i_size(); ++j) {
      attrExtr->list->i.push_back(listValue.i(j));
    }
    for (int j = 0; j < listValue.f_size(); ++j) {
      attrExtr->list->f.push_back(listValue.f(j));
    }
    for (int j = 0; j < listValue.type_size(); ++j) {
      attrExtr->list->type.push_back((tars::DataType)listValue.type(j));
    }
  }
  if (tf_attr.has_func()) {
    auto& func = tf_attr.func();
    attrExtr->func.reset(new tars::NamedAttrListT);
    attrExtr->func->name = func.name();
    for (const auto& it : func.attr()) {
      auto func_attr = ConvertTfAttribute(it.first, it.second);
      attrExtr->func->attr.push_back(std::move(func_attr));
    }
  }
  return std::move(attrExtr);
}

tfOpConverter* tfOpConverterSuit::search(const std::string& name) {
  auto iter = mTests.find(name);
  if (iter == mTests.end()) {
    static DefaultTfOpConverter converter;
    return &converter;
  }
  return iter->second;
}

tfOpConverterSuit* tfOpConverterSuit::get() {
  if (global == nullptr) global = new tfOpConverterSuit;
  return global;
}

tfOpConverterSuit::~tfOpConverterSuit() {
  for (auto& iter : mTests) {
    delete iter.second;
  }
  mTests.clear();
}

void tfOpConverterSuit::insert(tfOpConverter* t, const char* name) {
  OpCount::get()->insertOp("TF", name);
  mTests.insert(std::make_pair(name, t));
}
