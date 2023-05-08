#include "pocketlite/lite/flatbuffer_model.h"

namespace pocketlite {

FlatBufferModel::FlatBufferModel(const char* filepath_or_rawmodel,
                                 bool allow_mmap)
    : file_cont_(filepath_or_rawmodel, allow_mmap) {
  if (file_cont_.Get() == nullptr) {
    model_ = GetFbsModel(filepath_or_rawmodel);
  } else {
    model_ = GetFbsModel(file_cont_.Get());
  }
}

}  // namespace pocketlite
