#include "utils.h"

namespace ace {

int DataType2Bytes(DataType dtype) {
  switch (dtype) {
    case DataType_NONE:
      return 0;
    case DataType_INT_8:
      return 1;
    case DataType_INT_16:
      return 2;
    case DataType_INT_32:
      return 4;
    case DataType_INT_64:
      return 8;
    case DataType_UINT_8:
      return 1;
    case DataType_UINT_16:
      return 2;
    case DataType_UINT_32:
      return 4;
    case DataType_UINT_64:
      return 8;
    case DataType_FLOAT_16:
      return 2;
    case DataType_FLOAT_32:
      return 4;
    case DataType_FLOAT_64:
      return 8;
      break;
  }
  return 0;
}

}  // namespace ace