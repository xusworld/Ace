#include <glog/logging.h>
#include <stdio.h>

#include <fstream>

#include "onnx_helper.h"

namespace ace {
namespace model {

bool OnnxReadProtoFromBinary(const char* filepath,
                             google::protobuf::Message* message) {
  std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    fprintf(stderr, "open failed %s\n", filepath);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  google::protobuf::io::CodedInputStream codedstr(&input);

  bool success = message->ParseFromCodedStream(&codedstr);

  fs.close();

  return success;
}

bool OnnxReadProtoFromBinary(const std::string& filepath,
                             google::protobuf::Message* message) {
  return OnnxReadProtoFromBinary(filepath.c_str(), message);
}

bool OnnxWriteProtoFromBinary(const char* filepath,
                              const google::protobuf::Message* message) {
  std::ofstream fs(filepath);
  if (fs.fail()) {
    fprintf(stderr, "open failed %s\n", filepath);
    return false;
  }
  message->SerializeToOstream(&fs);
  fs.close();
  return true;
}

bool OnnxWriteProtoFromBinary(const std::string& filepath,
                              const google::protobuf::Message* message) {
  return OnnxWriteProtoFromBinary(filepath.c_str(), message);
}

}  // namespace model
}  // namespace ace