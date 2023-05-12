#include "onednn.h"

namespace onednn {

DnnlRuntime* DnnlRuntime::instance_ = nullptr;
dnnl::engine::kind DnnlRuntime::kind = dnnl::engine::kind::cpu;

}  // namespace onednn