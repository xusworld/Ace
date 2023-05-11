#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "thread_safe_macros.h"
#include "type_traits_extend.h"

namespace ace {
namespace device {

class ThreadPool {
 public:
  ThreadPool(int num_thread) : _num_thread(num_thread) {}
  virtual ~ThreadPool();

  void launch();

  /**
   *  \brief Lanuch the normal function task in sync.
   */
  template <typename functor, typename... ParamTypes>
  typename function_traits<functor>::return_type RunSync(functor function,
                                                         ParamTypes... args);

  /**
   *  \brief Lanuch the normal function task in async.
   */
  template <typename functor, typename... ParamTypes>
  typename std::future<typename function_traits<functor>::return_type> RunAsync(
      functor function, ParamTypes... args);

  /// Stop the pool.
  void stop();

 private:
  /// The initial function should be overrided by user who derive the ThreadPool
  /// class.
  virtual void init();

  /// Auxiliary function should be overrided when you want to do other things in
  /// the derived class.
  virtual void auxiliary_funcs();

 private:
  int _num_thread;
  std::vector<std::thread> _workers;
  std::queue<std::function<void(void)> > _tasks GUARDED_BY(_mut);
  std::mutex _mut;
  std::condition_variable _cv;
  bool _stop{false};
};

}  // namespace device
}  // namespace ace