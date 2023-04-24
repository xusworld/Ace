#include "converter/src/optimizer/passes/Pass.hpp"
#include "converter/src/optimizer/passes/PassRegistry.hpp"

namespace ace {
namespace passes {

REGISTER_REWRITE_PASS(NoPass)
    .Verify([](PassContext* context) { return false; })
    .Rewrite([](PassContext* context) { return false; });

}  // namespace passes
}  // namespace ace
