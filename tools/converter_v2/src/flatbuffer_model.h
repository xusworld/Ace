#ifndef POCKETLITE_LITE_FLATBUFFER_MODEL_H_
#define POCKETLITE_LITE_FLATBUFFER_MODEL_H_

#include "pocketlite/lite/file_content.h"

namespace pocketlite {

class FlatBufferModel final {
 public:
  PL_DELETE_COPY_AND_MOVE(FlatBufferModel);
  FlatBufferModel() = delete;
  ~FlatBufferModel() = default;

  FlatBufferModel(const char* filepath_or_rawmodel, bool allow_mmap);

  FlatBufferModel(const char* filepath_or_rawmodel)
      : FlatBufferModel(filepath_or_rawmodel, true) {}
  FlatBufferModel(const std::string& filepath_or_rawmodel)
      : FlatBufferModel(filepath_or_rawmodel.c_str()) {}

  const FbsModel* Get() const { return model_; }

 private:
  FileContent file_cont_;
  const FbsModel* model_;
};

}  // namespace pocketlite

#endif  // POCKETLITE_LITE_FLATBUFFER_MODEL_H_
