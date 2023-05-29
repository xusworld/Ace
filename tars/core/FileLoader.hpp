#include <mutex>
#include <vector>

#include "core/AutoStorage.h"

namespace tars {

class FileLoader final {
 public:
  FileLoader(const char* file);

  ~FileLoader();

  bool read();

  static bool write(const char* filePath,
                    std::pair<const void*, size_t> cacheInfo);

  bool valid() const { return mFile != nullptr; }
  inline size_t size() const { return mTotalSize; }

  bool merge(AutoStorage<uint8_t>& buffer);

  int offset(int64_t offset);

  bool read(char* buffer, int64_t size);

 private:
  std::vector<std::pair<size_t, void*>> mBlocks;
  FILE* mFile = nullptr;
  static const int gCacheSize = 4096;
  size_t mTotalSize = 0;
  const char* mFilePath = nullptr;
};
}  // namespace tars
