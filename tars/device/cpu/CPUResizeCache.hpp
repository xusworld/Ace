
#ifndef CPURESIZECACHE_HPP
#define CPURESIZECACHE_HPP
#include <map>

#include "MNN_generated.h"
#include "core/tensor.h"

namespace tars {
// FIXME: Move outside
class MNN_PUBLIC CPUResizeCache {
 public:
  CPUResizeCache() {
    // Do nothing
  }
  ~CPUResizeCache() {
    // Do nothing
  }
  Tensor* findCacheTensor(const Tensor* src, MNN_DATA_FORMAT format) const;
  // Return cache tensor
  void pushCacheTensor(std::shared_ptr<Tensor> dst, const Tensor* src,
                       MNN_DATA_FORMAT format);
  void reset();

 private:
  std::map<std::pair<const Tensor*, MNN_DATA_FORMAT>, std::shared_ptr<Tensor>>
      mFormatCache;
};
}  // namespace tars

#endif
