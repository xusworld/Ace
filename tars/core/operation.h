//
//  Operation.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Operation_hpp
#define Operation_hpp

#include <MNN/MNNForwardType.h>

#include <memory>
#include <string>

#include "NonCopyable.hpp"
#include "core/status.h"
#include "core/tensor.h"

namespace tars {
class Device;
struct Op;

// abstract class for operation
class Operation : public NonCopyable {
 public:
  Operation() = delete;
  Operation(Device *backend) : mBackEnd(backend) {}

  virtual ~Operation() = default;

  // response shape change of input or output tensors.
  virtual Status onResize(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs) {
    return Status::OK();
  }

  // run Operation.
  virtual Status onExecute(const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs) = 0;

  /**
   * @brief clone Operation, new Operation will share weight from this Operation
   * @param bn   the cloned' Operation's backend
   * @param dst if dst = nullptr, just return whether Operation can clone,
   * otherwise clone the Operation into dst
   * @return Operation result
   */
  virtual bool onClone(Device *bn, const Op *op, Operation **dst) {
    return false;
  }

 public:
  /**
   * @brief designed for plugin system. not ready yet.
   */
  class Creator : public NonCopyable {
   public:
    /**
     * @brief deinitializer.
     */
    virtual ~Creator() = default;
    /**
     * @brief create Operation for given op on given backend.
     * @param backend   given backend.
     * @param op        given op.
     * @return Operation.
     */
    virtual Operation *onCreate(Device *backend, const Op *op) const = 0;
  };

  // Search for extra creator, if not found, return nullptr
  MNN_PUBLIC static const Creator *searchExtraCreator(const std::string &key,
                                                      MNNForwardType type);

  /**
   * @brief register creator for given key and backend type.
   * @param creator registering creator.
   * @param key given key.
   * @param type given backend type.
   * @return false if registered creator for same key and type exists, true
   * otherwise.
   */
  MNN_PUBLIC static bool insertExtraCreator(std::shared_ptr<Creator> creator,
                                            const std::string &key,
                                            MNNForwardType type);

  /**
   * @brief unregister creator for given key and backend type.
   * @param key given key.
   * @param type given backend type.
   * @return true if registered creator for given key and type exists, false
   * otherwise.
   */
  MNN_PUBLIC static bool removeExtraCreator(const std::string &key,
                                            MNNForwardType type);

 public:
  /**
   * @brief check if Operation is valid.
   * @return valid or not.
   */
  inline bool valid() const { return mValid; }
  /**
   * @brief get backend.
   * @return backend.
   */
  Device *backend() const { return mBackEnd; }

 protected:
  bool mValid = true;

 private:
  Device *mBackEnd;
};

}  // namespace tars

#endif /* Operation_hpp */
