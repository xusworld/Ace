#pragma once

#include <string>

namespace ace {

enum class RetType {
  SUCCESS = 0,
  ERROR,
  FATAL,
  UNIMPLEMENTED,
};

class Status {
 public:
  Status() : is_suc_(RetType::SUCCESS), msg_("") {}

  Status(RetType ret) : is_suc_(ret), msg_("") {}

  Status(RetType ret, const char* err_msg = "Not known")
      : is_suc_(ret), msg_(err_msg) {}

  static Status OK(const char* msg = "") {
    return Status{RetType::SUCCESS, msg};
  }

  static Status ERROR(const char* msg = "Not known") {
    return Status{RetType::ERROR, msg};
  }

  static Status FATAL(const char* msg = "succeessfully exit") {
    return Status{RetType::FATAL, msg};
  }

  static Status UNIMPLEMENTED(const char* msg = "") {
    return Status{RetType::SUCCESS, msg};
  }

  operator bool() const { return (is_suc_ == RetType::SUCCESS); }

  const char* info() const { return msg_.c_str(); }

  bool operator==(const Status& status);
  bool operator!=(const Status& status);

  /// copy and move
  Status(const Status& status);
  Status(const Status&& status);
  Status& operator=(const Status& status);
  Status& operator=(const Status&& status);

 private:
  std::string msg_;
  RetType is_suc_{RetType::SUCCESS};
};

inline bool Status::operator==(const Status& status) {
  return (this->is_suc_ == status.is_suc_);
}

inline bool Status::operator!=(const Status& status) {
  return (this->is_suc_ != status.is_suc_);
}

inline Status::Status(const Status& status) {
  this->msg_ = status.msg_;
  this->is_suc_ = status.is_suc_;
}

inline Status::Status(const Status&& status) {
  this->msg_ = std::move(status.msg_);
  this->is_suc_ = status.is_suc_;
}

inline Status& Status::operator=(const Status& status) {
  this->msg_ = status.msg_;
  this->is_suc_ = status.is_suc_;
  return *(this);
}

inline Status& Status::operator=(const Status&& status) {
  this->msg_ = std::move(status.msg_);
  this->is_suc_ = status.is_suc_;
  return *(this);
}

}  // namespace ace