#pragma once

namespace ace {

class DeviceConfig {};

class Device {
 public:
  Device();
  ~Device();

  void create_stream();

 private:
  DeviceConfig config_;
  Runtime runtime_;
  Allocator allocator_;
  int32_t device_id_;
  int32_t threads_num_;
};

void RegisterDevice();

}  // namespace ace