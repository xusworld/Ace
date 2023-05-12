#include "elementwise_impl.h"

namespace ace {
namespace device {

Status init(const OpParam &param, std::vector<Tensor *> inputs,
            std::vector<Tensor *> outputs) {}

Status create(const OpParam &param, std::vector<Tensor *> inputs,
              std::vector<Tensor *> outputs) {}

Status dispatch(const OpParam &param, std::vector<Tensor *> inputs,
                std::vector<Tensor *> outputs) {}

}  // namespace device
}  // namespace ace