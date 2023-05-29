#ifndef ArgMaxExecution_hpp
#define ArgMaxExecution_hpp

#include <vector>

#include "core/operation.h"
#include "device/cuda/core/CUDABackend.hpp"

namespace tars {
namespace cuda {

class ArgMaxOp : public Operation {
 public:
  ArgMaxOp(const Op *op, Device *backend);
  virtual ~ArgMaxOp();
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) override;
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) override;

 private:
  const Op *mOp;
  int mAxis;
  int mInside;
  int mOutside;
  int mDim;
};

}  // namespace cuda
}  // namespace tars

#endif
