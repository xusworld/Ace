// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_TYPES_ACE_H_
#define FLATBUFFERS_GENERATED_TYPES_ACE_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 2 &&
              FLATBUFFERS_VERSION_MINOR == 0 &&
              FLATBUFFERS_VERSION_REVISION == 8,
             "Non-compatible flatbuffers version included");

namespace ace {

enum FrontendFramework : int8_t {
  FrontendFramework_NONE = 0,
  FrontendFramework_ONNX = 1,
  FrontendFramework_CAFFE = 2,
  FrontendFramework_TENSORFLOW = 3,
  FrontendFramework_TFLITE = 4,
  FrontendFramework_TORCH = 5,
  FrontendFramework_MIN = FrontendFramework_NONE,
  FrontendFramework_MAX = FrontendFramework_TORCH
};

inline const FrontendFramework (&EnumValuesFrontendFramework())[6] {
  static const FrontendFramework values[] = {
    FrontendFramework_NONE,
    FrontendFramework_ONNX,
    FrontendFramework_CAFFE,
    FrontendFramework_TENSORFLOW,
    FrontendFramework_TFLITE,
    FrontendFramework_TORCH
  };
  return values;
}

inline const char * const *EnumNamesFrontendFramework() {
  static const char * const names[7] = {
    "NONE",
    "ONNX",
    "CAFFE",
    "TENSORFLOW",
    "TFLITE",
    "TORCH",
    nullptr
  };
  return names;
}

inline const char *EnumNameFrontendFramework(FrontendFramework e) {
  if (flatbuffers::IsOutRange(e, FrontendFramework_NONE, FrontendFramework_TORCH)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesFrontendFramework()[index];
}

enum DeviceType : int8_t {
  DeviceType_NONE = 0,
  DeviceType_X86 = 1,
  DeviceType_CUDA = 2,
  DeviceType_MIN = DeviceType_NONE,
  DeviceType_MAX = DeviceType_CUDA
};

inline const DeviceType (&EnumValuesDeviceType())[3] {
  static const DeviceType values[] = {
    DeviceType_NONE,
    DeviceType_X86,
    DeviceType_CUDA
  };
  return values;
}

inline const char * const *EnumNamesDeviceType() {
  static const char * const names[4] = {
    "NONE",
    "X86",
    "CUDA",
    nullptr
  };
  return names;
}

inline const char *EnumNameDeviceType(DeviceType e) {
  if (flatbuffers::IsOutRange(e, DeviceType_NONE, DeviceType_CUDA)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDeviceType()[index];
}

enum DataType : int8_t {
  DataType_NONE = 0,
  DataType_INT_8 = 1,
  DataType_INT_16 = 2,
  DataType_INT_32 = 3,
  DataType_INT_64 = 4,
  DataType_UINT_8 = 5,
  DataType_UINT_16 = 6,
  DataType_UINT_32 = 7,
  DataType_UINT_64 = 8,
  DataType_FLOAT_16 = 9,
  DataType_FLOAT_32 = 10,
  DataType_FLOAT_64 = 11,
  DataType_MIN = DataType_NONE,
  DataType_MAX = DataType_FLOAT_64
};

inline const DataType (&EnumValuesDataType())[12] {
  static const DataType values[] = {
    DataType_NONE,
    DataType_INT_8,
    DataType_INT_16,
    DataType_INT_32,
    DataType_INT_64,
    DataType_UINT_8,
    DataType_UINT_16,
    DataType_UINT_32,
    DataType_UINT_64,
    DataType_FLOAT_16,
    DataType_FLOAT_32,
    DataType_FLOAT_64
  };
  return values;
}

inline const char * const *EnumNamesDataType() {
  static const char * const names[13] = {
    "NONE",
    "INT_8",
    "INT_16",
    "INT_32",
    "INT_64",
    "UINT_8",
    "UINT_16",
    "UINT_32",
    "UINT_64",
    "FLOAT_16",
    "FLOAT_32",
    "FLOAT_64",
    nullptr
  };
  return names;
}

inline const char *EnumNameDataType(DataType e) {
  if (flatbuffers::IsOutRange(e, DataType_NONE, DataType_FLOAT_64)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDataType()[index];
}

enum DataFormat : int8_t {
  DataFormat_NONE = 0,
  DataFormat_NCHW = 1,
  DataFormat_NHWC = 2,
  DataFormat_NC4HW4 = 3,
  DataFormat_NHWC4 = 4,
  DataFormat_MIN = DataFormat_NONE,
  DataFormat_MAX = DataFormat_NHWC4
};

inline const DataFormat (&EnumValuesDataFormat())[5] {
  static const DataFormat values[] = {
    DataFormat_NONE,
    DataFormat_NCHW,
    DataFormat_NHWC,
    DataFormat_NC4HW4,
    DataFormat_NHWC4
  };
  return values;
}

inline const char * const *EnumNamesDataFormat() {
  static const char * const names[6] = {
    "NONE",
    "NCHW",
    "NHWC",
    "NC4HW4",
    "NHWC4",
    nullptr
  };
  return names;
}

inline const char *EnumNameDataFormat(DataFormat e) {
  if (flatbuffers::IsOutRange(e, DataFormat_NONE, DataFormat_NHWC4)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDataFormat()[index];
}

enum OptimLevel : int8_t {
  OptimLevel_NONE = 0,
  OptimLevel_O0 = 1,
  OptimLevel_O1 = 2,
  OptimLevel_O2 = 3,
  OptimLevel_O3 = 4,
  OptimLevel_MIN = OptimLevel_NONE,
  OptimLevel_MAX = OptimLevel_O3
};

inline const OptimLevel (&EnumValuesOptimLevel())[5] {
  static const OptimLevel values[] = {
    OptimLevel_NONE,
    OptimLevel_O0,
    OptimLevel_O1,
    OptimLevel_O2,
    OptimLevel_O3
  };
  return values;
}

inline const char * const *EnumNamesOptimLevel() {
  static const char * const names[6] = {
    "NONE",
    "O0",
    "O1",
    "O2",
    "O3",
    nullptr
  };
  return names;
}

inline const char *EnumNameOptimLevel(OptimLevel e) {
  if (flatbuffers::IsOutRange(e, OptimLevel_NONE, OptimLevel_O3)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesOptimLevel()[index];
}

inline const flatbuffers::TypeTable *FrontendFrameworkTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::FrontendFrameworkTypeTable
  };
  static const char * const names[] = {
    "NONE",
    "ONNX",
    "CAFFE",
    "TENSORFLOW",
    "TFLITE",
    "TORCH"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 6, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *DeviceTypeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::DeviceTypeTypeTable
  };
  static const char * const names[] = {
    "NONE",
    "X86",
    "CUDA"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 3, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *DataTypeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::DataTypeTypeTable
  };
  static const char * const names[] = {
    "NONE",
    "INT_8",
    "INT_16",
    "INT_32",
    "INT_64",
    "UINT_8",
    "UINT_16",
    "UINT_32",
    "UINT_64",
    "FLOAT_16",
    "FLOAT_32",
    "FLOAT_64"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 12, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *DataFormatTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::DataFormatTypeTable
  };
  static const char * const names[] = {
    "NONE",
    "NCHW",
    "NHWC",
    "NC4HW4",
    "NHWC4"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 5, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *OptimLevelTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ace::OptimLevelTypeTable
  };
  static const char * const names[] = {
    "NONE",
    "O0",
    "O1",
    "O2",
    "O3"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 5, type_codes, type_refs, nullptr, nullptr, names
  };
  return &tt;
}

}  // namespace ace

#endif  // FLATBUFFERS_GENERATED_TYPES_ACE_H_