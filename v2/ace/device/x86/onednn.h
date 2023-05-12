#pragma once

#include <glog/logging.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

namespace onednn {

class DnnlRuntime {
 public:
  DnnlRuntime() = default;
  ~DnnlRuntime() = default;

 public:
  static void Init() {
    if (instance_ == nullptr) {
      instance_ = new DnnlRuntime();
      // Create execution dnnl::engine.
      instance_->engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
      // Create dnnl::stream.
      instance_->stream = dnnl::stream(instance_->engine);
    }
  }

  static const dnnl::engine& GetDnnlEngine() {
    Init();
    return instance_->engine;
  }

  static const dnnl::stream& GetDnnlStream() {
    Init();
    return instance_->stream;
  }

  static void SetDnnlEngine(const dnnl::engine& engine) {}
  static void SetDnnlStream(const dnnl::stream& stream) {}

  static DnnlRuntime* instance_;
  static dnnl::engine::kind kind;

 private:
  dnnl::engine engine;
  dnnl::stream stream;
};

}  // namespace onednn