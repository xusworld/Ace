//
//  ThreadPool.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPU_INTHREADPOOL_H
#define CPU_INTHREADPOOL_H
#ifdef MNN_USE_THREAD_POOL
#include <ace/MNNDefine.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
namespace ace {

class MNN_PUBLIC ThreadPool {
 public:
  typedef std::pair<std::function<void(int)>, int> TASK;

  int number() const { return mNumberThread; }
  static void enqueue(TASK&& task, int index);

  static void active();
  static void deactive();

  static int acquireWorkIndex();
  static void releaseWorkIndex(int index);

  static int init(int number);
  static void destroy();

 private:
  void enqueueInternal(TASK&& task, int index);

  static ThreadPool* gInstance;
  ThreadPool(int number = 0);
  ~ThreadPool();

  std::vector<std::thread> mWorkers;
  std::vector<bool> mTaskAvailable;
  std::atomic<bool> mStop = {false};

  std::vector<std::pair<TASK, std::vector<std::atomic_bool*>>> mTasks;
  std::condition_variable mCondition;
  std::mutex mQueueMutex;

  int mNumberThread = 0;
  std::atomic_int mActiveCount = {0};
};
}  // namespace ace
#endif
#endif