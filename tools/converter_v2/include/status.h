#pragma once

#include <string>

namespace ace {
namespace model {

enum class RetType {
  SUC,        ///< succeess
  ERR,        ///< error
  IMME_EXIT,  ///< need immediately exit
};

class Status {
 public:
  Status() : _is_suc(RetType::SUC), _error_msg("") {}
  Status(RetType ret) : _is_suc(ret), _error_msg("") {}
  Status(RetType ret, const char* err_msg = "Not known")
      : _is_suc(ret), _error_msg(err_msg) {}

  static Status OK(const char* msg = "") { return Status{RetType::SUC, msg}; }
  static Status FAIL(const char* msg = "Not known") {
    return Status{RetType::ERR, msg};
  }
  static Status EXIT(const char* msg = "succeessfully exit") {
    return Status{RetType::IMME_EXIT, msg};
  }

  operator bool() const {
    return (_is_suc == RetType::SUC) || (_is_suc == RetType::IMME_EXIT);
  }

  const char* info() const { return _error_msg.c_str(); }

  bool operator==(const Status& status);
  bool operator!=(const Status& status);

  /// copy and move
  Status(const Status& status);
  Status(const Status&& status);
  Status& operator=(const Status& status);
  Status& operator=(const Status&& status);

 private:
  std::string _error_msg;
  RetType _is_suc{RetType::SUC};
};

inline bool Status::operator==(const Status& status) {
  return (this->_is_suc == status._is_suc);
}

inline bool Status::operator!=(const Status& status) {
  return (this->_is_suc != status._is_suc);
}

inline Status::Status(const Status& status) {
  this->_error_msg = status._error_msg;
  this->_is_suc = status._is_suc;
}

inline Status::Status(const Status&& status) {
  this->_error_msg = std::move(status._error_msg);
  this->_is_suc = status._is_suc;
}

inline Status& Status::operator=(const Status& status) {
  this->_error_msg = status._error_msg;
  this->_is_suc = status._is_suc;
  return *(this);
}

inline Status& Status::operator=(const Status&& status) {
  this->_error_msg = std::move(status._error_msg);
  this->_is_suc = status._is_suc;
  return *(this);
}

}  // namespace model
}  // namespace ace