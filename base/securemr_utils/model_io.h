// Copyright (2025) Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SECUREMR_UTILS_MODEL_IO_H_
#define SECUREMR_UTILS_MODEL_IO_H_

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SecureMR {

struct ModelBinding {
  std::string operatorName;
  std::string modelNodeName;
  std::string tensorName;
};

inline std::vector<ModelBinding> NormalizeModelBindings(
    const std::vector<std::pair<std::string, std::string>>& mappedBindings) {
  std::vector<ModelBinding> normalized;
  normalized.reserve(mappedBindings.size());
  for (const auto& [modelNodeName, tensorName] : mappedBindings) {
    // Operator I/O names only need to be unique within this operator. Using
    // the model node name preserves distinct model slots that intentionally
    // read from or write to the same package tensor.
    normalized.push_back({modelNodeName, modelNodeName, tensorName});
  }
  return normalized;
}

inline bool IsValidModelName(const std::string& value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isalnum(character) != 0 || character == '_';
         });
}

inline std::string NormalizeModelName(const std::string& value) {
  const std::string normalized = value.empty() ? "main" : value;
  if (!IsValidModelName(normalized)) {
    throw std::runtime_error(
        "model_name must contain only letters, digits, and underscores");
  }
  return normalized;
}

template <size_t Capacity>
void CopyOperatorIoName(char (&destination)[Capacity], const std::string& value, const char* field) {
  if (value.find('\0') != std::string::npos || value.size() >= Capacity) {
    throw std::runtime_error(
        std::string(field) + " must be shorter than " + std::to_string(Capacity) +
        " bytes and contain no NUL bytes");
  }
  std::copy(value.begin(), value.end(), destination);
  destination[value.size()] = '\0';
}

}  // namespace SecureMR

#endif  // SECUREMR_UTILS_MODEL_IO_H_
