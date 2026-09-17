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

#include "securemr_utils/serialization.h"

#include <charconv>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <system_error>
#include <stdexcept>
#include <unordered_map>

#include "oxr_utils/common.h"
#include "model_io.h"
#ifndef SECUREMR_SERIALIZATION_PARSE_ONLY
#include "oxr_utils/logger.h"
#include "pipeline.h"
#include "rendercommand.h"
#endif
#include "tensor.h"

#ifdef XR_USE_PLATFORM_ANDROID
#include <android/asset_manager.h>
extern AAssetManager* g_assetManager;
#endif

namespace SecureMR {

std::string NormalizePackageRelativePath(const std::string& value, const char* what) {
  std::string normalized = value;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  if (normalized.empty() || normalized.front() == '/' || normalized.front() == '\\' ||
      (normalized.size() >= 2 && std::isalpha(static_cast<unsigned char>(normalized[0])) &&
       normalized[1] == ':') ||
      (normalized.size() >= 2 && normalized[0] == '/' && normalized[1] == '/')) {
    throw std::runtime_error(Fmt("%s must be a package-relative path", what));
  }

  if (normalized.back() == '/') {
    throw std::runtime_error(Fmt("%s must not contain absolute, empty, '.', or '..' path components", what));
  }

  std::string component;
  std::istringstream stream(normalized);
  while (std::getline(stream, component, '/')) {
    if (component.empty() || component == "." || component == "..") {
      throw std::runtime_error(Fmt("%s must not contain absolute, empty, '.', or '..' path components", what));
    }
  }
  return normalized;
}

namespace {

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string Trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\n\r");
  return value.substr(first, last - first + 1);
}

XrSecureMrTensorDataTypePICO ParseDataType(
    const Json& value,
    XrSecureMrTensorDataTypePICO defaultType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO) {
  if (value.is_number_integer()) {
    return static_cast<XrSecureMrTensorDataTypePICO>(value.get<int>());
  }
  if (!value.is_string()) {
    return defaultType;
  }
  const std::string type = ToLower(value.get<std::string>());
  if (type == "float32" || type == "fp32") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  }
  if (type == "float64" || type == "double") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO;
  }
  if (type == "int32") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO;
  }
  if (type == "int16") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO;
  }
  if (type == "int8") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO;
  }
  if (type == "uint16") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO;
  }
  if (type == "uint8") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO;
  }
  return defaultType;
}

XrSecureMrTensorTypePICO ParseTensorUsage(const Json& value,
                                          XrSecureMrTensorTypePICO defaultUsage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO) {
  if (value.is_number_integer()) {
    return static_cast<XrSecureMrTensorTypePICO>(value.get<int>());
  }
  if (!value.is_string()) {
    return defaultUsage;
  }
  const std::string usage = ToLower(value.get<std::string>());
  if (usage == "mat" || usage == "matrix") {
    return XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  }
  return defaultUsage;
}

bool JsonToSpecialTensorAttribute(const Json& j, TensorAttribute& out) {
  auto typeIt = j.find("tensor_type");
  if (typeIt == j.end() || !typeIt->is_string()) {
    return false;
  }
  const std::string tensorType = ToLower(typeIt->get<std::string>());
  const size_t size = j.value("size", 1);
  const auto dataTypeIt = j.find("data_type");
  const XrSecureMrTensorDataTypePICO dataType =
      dataTypeIt == j.end() ? XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO : ParseDataType(*dataTypeIt);

  if (tensorType == "scalar_array") {
    out = TensorAttribute_ScalarArray{.size = size, .dataType = dataType};
    return true;
  }
  if (tensorType == "point2_array") {
    out = TensorAttribute_Point2Array{.size = size, .dataType = dataType};
    return true;
  }
  if (tensorType == "point3_array") {
    out = TensorAttribute_Point3Array{.size = size, .dataType = dataType};
    return true;
  }
  if (tensorType == "rgba_array") {
    out = TensorAttribute_RGBA_Array{.size = size};
    return true;
  }
  if (tensorType == "timestamp") {
    out = TensorAttribute_TimeStamp{};
    return true;
  }
  return false;
}

#ifndef SECUREMR_SERIALIZATION_PARSE_ONLY

struct TensorReference {
  std::shared_ptr<PipelineTensor> tensor;
  std::optional<PipelineTensor::Slice> slice;
};

TensorReference ResolveTensorReference(
    const std::string& tensorName,
    const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (tensorName.find('[') != std::string::npos || tensorName.find(']') != std::string::npos) {
    throw std::runtime_error(Fmt("bracketed tensor reference '%s' is not supported in package JSON",
                                 tensorName.c_str()));
  }
  return {.tensor = requireTensor(tensorName), .slice = std::nullopt};
}

std::vector<std::vector<int>> ParseAssignmentSlices(const Json& value, const char* key) {
  if (!value.is_array() || value.empty()) {
    throw std::runtime_error(Fmt("%s must be a non-empty array", key));
  }

  std::vector<std::vector<int>> slices;
  slices.reserve(value.size());
  for (const auto& dimension : value) {
    if (!dimension.is_array() || dimension.empty() || dimension.size() > 3) {
      throw std::runtime_error(Fmt("%s entries must contain 1 to 3 integers", key));
    }

    std::vector<int> parsed;
    parsed.reserve(dimension.size());
    for (const auto& element : dimension) {
      if (!element.is_number_integer()) {
        throw std::runtime_error(Fmt("%s entries must contain 1 to 3 integers", key));
      }
      parsed.push_back(element.get<int>());
    }
    if (parsed.size() == 1) {
      parsed.push_back(parsed.front() + 1);
    }
    slices.push_back(std::move(parsed));
  }
  return slices;
}

std::vector<int> ParseAssignmentChannelSlice(const Json& value, const char* key) {
  if (!value.is_array() || value.empty() || value.size() > 3) {
    throw std::runtime_error(Fmt("%s must contain 1 to 3 integers", key));
  }

  std::vector<int> slice;
  slice.reserve(value.size());
  for (const auto& element : value) {
    if (!element.is_number_integer()) {
      throw std::runtime_error(Fmt("%s must contain 1 to 3 integers", key));
    }
    slice.push_back(element.get<int>());
  }
  return slice;
}

PipelineTensor::Slice FullTensorSlice(const std::shared_ptr<PipelineTensor>& tensor, const char* key) {
  const auto attribute = tensor->getAttribute();
  if (!std::holds_alternative<TensorAttribute>(attribute)) {
    throw std::runtime_error(Fmt("%s cannot slice a glTF tensor", key));
  }

  const auto& tensorAttribute = std::get<TensorAttribute>(attribute);
  if (tensorAttribute.dimensions.empty()) {
    throw std::runtime_error(Fmt("%s requires a tensor with dimensions", key));
  }

  std::vector<std::vector<int>> slices;
  slices.reserve(tensorAttribute.dimensions.size());
  for (const int dimension : tensorAttribute.dimensions) {
    if (dimension <= 0) {
      throw std::runtime_error(Fmt("%s cannot slice a tensor with non-positive dimensions", key));
    }
    slices.push_back({0, dimension});
  }
  return (*tensor)[slices];
}

PipelineTensor::Slice ApplyAssignmentChannelSlice(PipelineTensor::Slice slice, const Json& value, const char* key) {
  const auto channelSlice = ParseAssignmentChannelSlice(value, key);
  if (channelSlice.size() == 1) {
    slice[channelSlice[0]];
  } else if (channelSlice.size() == 2) {
    slice[std::array<int, 2>{channelSlice[0], channelSlice[1]}];
  } else {
    slice[std::array<int, 3>{channelSlice[0], channelSlice[1], channelSlice[2]}];
  }
  return slice;
}

TensorReference ResolveAssignmentTensorReference(
    const std::string& tensorName,
    const Json& opSpec,
    const char* slicesKey,
    const char* channelSliceKey,
    const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  const bool hasInlineSlices = opSpec.contains(slicesKey);
  auto reference = ResolveTensorReference(tensorName, requireTensor);

  if (hasInlineSlices) {
    reference.slice = (*reference.tensor)[ParseAssignmentSlices(opSpec.at(slicesKey), slicesKey)];
  } else if (opSpec.contains(channelSliceKey) && !reference.slice.has_value()) {
    reference.slice = FullTensorSlice(reference.tensor, channelSliceKey);
  }

  if (opSpec.contains(channelSliceKey)) {
    reference.slice = ApplyAssignmentChannelSlice(
        reference.slice.has_value() ? *reference.slice : FullTensorSlice(reference.tensor, channelSliceKey),
        opSpec.at(channelSliceKey), channelSliceKey);
  }
  return reference;
}

Pipeline::ElementwiseOp ParseElementwiseOp(const std::string& op) {
  const std::string value = ToLower(op);
  if (value == "min") return Pipeline::ElementwiseOp::MIN;
  if (value == "max") return Pipeline::ElementwiseOp::MAX;
  if (value == "multiply") return Pipeline::ElementwiseOp::MULTIPLY;
  if (value == "or") return Pipeline::ElementwiseOp::OR;
  if (value == "and") return Pipeline::ElementwiseOp::AND;
  throw std::runtime_error(Fmt("unsupported elementwise op '%s'", op.c_str()));
}

XrSecureMrComparisonPICO ParseComparison(const std::string& compare) {
  const std::string value = ToLower(compare);
  if (value == ">") {
    return XR_SECURE_MR_COMPARISON_LARGER_THAN_PICO;
  }
  if (value == ">=") {
    return XR_SECURE_MR_COMPARISON_LARGER_OR_EQUAL_PICO;
  }
  if (value == "<") {
    return XR_SECURE_MR_COMPARISON_SMALLER_THAN_PICO;
  }
  if (value == "<=") {
    return XR_SECURE_MR_COMPARISON_SMALLER_OR_EQUAL_PICO;
  }
  if (value == "==") {
    return XR_SECURE_MR_COMPARISON_EQUAL_TO_PICO;
  }
  if (value == "!=") {
    return XR_SECURE_MR_COMPARISON_NOT_EQUAL_PICO;
  }
  throw std::runtime_error(Fmt("unsupported comparison '%s'", compare.c_str()));
}

XrSecureMrModelTypePICO ParseModelType(const std::string& modelType) {
  const std::string value = ToLower(modelType);
  if (value == "tflite") {
    return XR_SECURE_MR_MODEL_TYPE_LITE_RT_MODEL_PICO;
  }
  throw std::runtime_error(Fmt("unsupported model_type '%s'", modelType.c_str()));
}


XrSecureMrModelTargetPICO ParseModelTarget(const std::string& modelTarget) {
  const std::string value = ToLower(modelTarget);
  if (value == "npu") {
    return XR_SECURE_MR_MODEL_TARGET_NPU_PICO;
  }
  if (value == "gpu") {
    return XR_SECURE_MR_MODEL_TARGET_GPU_PICO;
  }
  if (value == "cpu") {
    return XR_SECURE_MR_MODEL_TARGET_CPU_PICO;
  }
  throw std::runtime_error(Fmt("unsupported model_target '%s'", modelTarget.c_str()));
}

Pipeline::NormalizeType ParseNormalizeType(const std::string& normalizeType) {
  const std::string value = ToLower(normalizeType);
  if (value == "l2") return Pipeline::NormalizeType::L2;
  if (value == "l1") return Pipeline::NormalizeType::L1;
  if (value == "inf") {
    return Pipeline::NormalizeType::INF;
  }
  if (value == "minmax") {
    return Pipeline::NormalizeType::MINMAX;
  }
  throw std::runtime_error(Fmt("unsupported normalize_type '%s'", normalizeType.c_str()));
}

XrSecureMrMatrixSortTypePICO ParseMatrixSortType(const std::string& sortType) {
  if (sortType == "ROW") {
    return XR_SECURE_MR_MATRIX_SORT_TYPE_ROW_PICO;
  }
  if (sortType == "COLUMN") {
    return XR_SECURE_MR_MATRIX_SORT_TYPE_COLUMN_PICO;
  }
  throw std::runtime_error(Fmt("unsupported sort_type '%s'", sortType.c_str()));
}

XrSecureMrAudioFormatPcmPICO ParseAudioFormat(const std::string& sampleFormat) {
  if (sampleFormat == "PCM_16BIT") {
    return XR_SECURE_MR_AUDIO_FORMAT_PCM_16BIT_PICO;
  }
  if (sampleFormat == "PCM_32BIT") {
    return XR_SECURE_MR_AUDIO_FORMAT_PCM_32BIT_PICO;
  }
  if (sampleFormat == "PCM_FLOAT") {
    return XR_SECURE_MR_AUDIO_FORMAT_PCM_FLOAT_PICO;
  }
  throw std::runtime_error(Fmt("unsupported audio format '%s'", sampleFormat.c_str()));
}

RenderCommand_UpdateMaterial::MaterialAttribute ParseMaterialAttribute(const std::string& attribute) {
  if (attribute == "material::metallic_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_METALLIC;
  }
  if (attribute == "material::roughness_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_ROUGHNESS;
  }
  if (attribute == "material::emissive_strength") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_EMISSIVE_STRENGTH;
  }
  if (attribute == "material::base_color_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_BASE_COLOR;
  }
  if (attribute == "material::emissive_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_EMISSIVE;
  }
  if (attribute == "material::occlusion_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_OCCLUSION_MAP;
  }
  if (attribute == "material::emissive_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_EMISSIVE;
  }
  if (attribute == "material::base_color_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
  }
  if (attribute == "material::normal_map_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_NORMAL_MAP;
  }
  if (attribute == "material::metallic_roughness_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_METALLIC_ROUGHNESS;
  }
  throw std::runtime_error(Fmt("unsupported update_gltf material attribute '%s'", attribute.c_str()));
}

std::vector<uint16_t> JsonToUInt16Vector(const Json& value) {
  if (value.is_number_integer()) {
    return {static_cast<uint16_t>(value.get<int>())};
  }
  if (!value.is_array() || value.empty()) {
    throw std::runtime_error("expected a uint16 value or non-empty uint16 array");
  }
  std::vector<uint16_t> result;
  result.reserve(value.size());
  for (const auto& item : value) {
    if (!item.is_number_integer() || item.get<int>() < 0 || item.get<int>() > std::numeric_limits<uint16_t>::max()) {
      throw std::runtime_error("expected uint16 value or array");
    }
    result.push_back(static_cast<uint16_t>(item.get<int>()));
  }
  return result;
}

std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>> ParseTensorOrUInt16Vector(
    const Json& value, const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  return JsonToUInt16Vector(value);
}

std::variant<std::shared_ptr<PipelineTensor>, uint16_t> ParseTensorOrUInt16(
    const Json& value, const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  const auto values = JsonToUInt16Vector(value);
  if (values.size() != 1) {
    throw std::runtime_error("expected one uint16 value or tensor name");
  }
  return values.front();
}

std::variant<std::shared_ptr<PipelineTensor>, float> ParseTensorOrFloat(
    const Json& value, const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  if (!value.is_number()) {
    throw std::runtime_error("expected float value or tensor name");
  }
  return value.get<float>();
}

std::variant<std::shared_ptr<PipelineTensor>, std::vector<float>, std::vector<uint16_t>,
             std::vector<std::array<uint8_t, 4>>>
ParseMaterialValues(const Json& value, const RenderCommand_UpdateMaterial::MaterialAttribute attribute,
                    const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  switch (attribute) {
    case RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_METALLIC:
    case RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_ROUGHNESS:
    case RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_EMISSIVE_STRENGTH: {
      if (value.is_number()) {
        return std::vector<float>{value.get<float>()};
      }
      if (!value.is_array()) {
        throw std::runtime_error("expected material float value or array");
      }
      std::vector<float> result;
      result.reserve(value.size());
      for (const auto& item : value) {
        if (!item.is_number()) {
          throw std::runtime_error("expected numeric material value");
        }
        result.push_back(item.get<float>());
      }
      return result;
    }
    case RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_BASE_COLOR:
    case RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_EMISSIVE: {
      if (!value.is_array()) {
        throw std::runtime_error("expected material RGBA value or array");
      }
      std::vector<std::array<uint8_t, 4>> result;
      auto parseColor = [](const Json& color) {
        if (!color.is_array() || color.size() != 4) {
          throw std::runtime_error("expected RGBA material values");
        }
        std::array<uint8_t, 4> parsed{};
        for (size_t index = 0; index < 4; ++index) {
          if (!color[index].is_number_integer() || color[index].get<int>() < 0 || color[index].get<int>() > 255) {
            throw std::runtime_error("expected RGBA material values in the range 0..255");
          }
          parsed[index] = static_cast<uint8_t>(color[index].get<int>());
        }
        return parsed;
      };
      if (value.size() == 4 && value[0].is_number_integer()) {
        result.push_back(parseColor(value));
      } else {
        result.reserve(value.size());
        for (const auto& color : value) {
          result.push_back(parseColor(color));
        }
      }
      return result;
    }
    default:
      return JsonToUInt16Vector(value);
  }
}

std::string ReadAttrString(const Json& opSpec, size_t index, const char* opName, bool required,
                           const char* defaultValue = "") {
  const auto attrsIt = opSpec.find("attrs");
  if (attrsIt == opSpec.end()) {
    if (required) {
      throw std::runtime_error(Fmt("%s requires attrs[%zu]", opName, index));
    }
    return defaultValue;
  }
  if (!attrsIt->is_array()) {
    throw std::runtime_error(Fmt("%s attrs must be an array", opName));
  }
  if (index >= attrsIt->size()) {
    if (required) {
      throw std::runtime_error(Fmt("%s requires attrs[%zu]", opName, index));
    }
    return defaultValue;
  }
  if (!(*attrsIt)[index].is_string()) {
    throw std::runtime_error(Fmt("%s attrs[%zu] must be a string", opName, index));
  }
  return (*attrsIt)[index].get<std::string>();
}

void ValidateAttrCount(const Json& opSpec, size_t minCount, size_t maxCount, const char* opName) {
  const auto attrsIt = opSpec.find("attrs");
  const size_t count = attrsIt == opSpec.end() ? 0 : attrsIt->size();
  if (attrsIt != opSpec.end() && !attrsIt->is_array()) {
    throw std::runtime_error(Fmt("%s attrs must be an array", opName));
  }
  if (count < minCount || count > maxCount) {
    throw std::runtime_error(Fmt("%s has invalid attrs count: %zu", opName, count));
  }
  for (size_t idx = 0; idx < count; ++idx) {
    if (!(*attrsIt)[idx].is_string()) {
      throw std::runtime_error(Fmt("%s attrs[%zu] must be a string", opName, idx));
    }
  }
}

void RejectLegacyFields(const Json& opSpec) {
  if (!opSpec.is_object()) {
    throw std::runtime_error("operator entry must be an object");
  }
  static const std::set<std::string> kAllowedKeys = {"type", "inputs", "outputs", "attrs", "model",
                                                    "src_slices", "dst_slices", "src_channel_slice",
                                                    "dst_channel_slice"};
  for (auto it = opSpec.begin(); it != opSpec.end(); ++it) {
    if (kAllowedKeys.find(it.key()) == kAllowedKeys.end()) {
      throw std::runtime_error(Fmt("operator field '%s' is not supported in package JSON", it.key().c_str()));
    }
  }
}

RenderCommand_DrawText::TypeFaceTypes ParseTypeFace(const std::string& value) {
  const std::string typeface = ToLower(value);
  if (typeface == "sans_serif" || typeface == "sans-serif") return RenderCommand_DrawText::TypeFaceTypes::SANS_SERIF;
  if (typeface == "serif") return RenderCommand_DrawText::TypeFaceTypes::SERIF;
  if (typeface == "monospace") return RenderCommand_DrawText::TypeFaceTypes::MONOSPACE;
  if (typeface == "bold") return RenderCommand_DrawText::TypeFaceTypes::BOLD;
  if (typeface == "italic") return RenderCommand_DrawText::TypeFaceTypes::ITALIC;
  return RenderCommand_DrawText::TypeFaceTypes::DEFAULT;
}

void ValidateOperatorArity(const std::string& type, size_t inputCount, size_t outputCount) {
  size_t minInputs = 0;
  size_t maxInputs = 0;
  size_t minOutputs = 0;
  size_t maxOutputs = 0;
  bool known = true;
  if (type == "camera_access") { minOutputs = maxOutputs = 4; }
  else if (type == "camera_space_to_world") { minInputs = maxInputs = 1; minOutputs = maxOutputs = 2; }
  else if (type == "uv_to_3d_in_camera_space") { minInputs = maxInputs = 5; minOutputs = maxOutputs = 1; }
  else if (type == "get_affine") { minInputs = maxInputs = 2; minOutputs = maxOutputs = 1; }
  else if (type == "apply_affine" || type == "apply_affine_point") {
    minInputs = maxInputs = 2; minOutputs = maxOutputs = 1;
  } else if (type == "assignment" || type == "type_convert") {
    minInputs = maxInputs = 1; minOutputs = maxOutputs = 1;
  } else if (type == "customized_compare" || type == "elementwise_min" || type == "elementwise_max" ||
             type == "elementwise_multiply" || type == "elementwise_or" || type == "elementwise_and") {
    minInputs = maxInputs = 2; minOutputs = maxOutputs = 1;
  } else if (type == "all" || type == "any" || type == "argmax" || type == "cvt_color" ||
             type == "inversion" || type == "norm" || type == "swap_hwc_chw") {
    minInputs = maxInputs = 1; minOutputs = maxOutputs = 1;
  } else if (type == "normalize") {
    minInputs = 1; maxInputs = 2; minOutputs = maxOutputs = 1;
  } else if (type == "arithmetic") {
    minInputs = maxInputs = 10; minOutputs = maxOutputs = 1;
  } else if (type == "nms") {
    minInputs = maxInputs = 2; minOutputs = maxOutputs = 3;
  } else if (type == "solve_p_n_p") {
    minInputs = maxInputs = 3; minOutputs = maxOutputs = 2;
  } else if (type == "sort_vec" || type == "sort_mat") {
    minInputs = maxInputs = 1; minOutputs = maxOutputs = 2;
  } else if (type == "svd") {
    minInputs = maxInputs = 1; minOutputs = maxOutputs = 3;
  } else if (type == "get_transform_mat") {
    minInputs = maxInputs = 3; minOutputs = maxOutputs = 1;
  } else if (type == "load_texture") {
    minInputs = maxInputs = 2; minOutputs = maxOutputs = 1;
  } else if (type == "switch_gltf_render_status") {
    minInputs = maxInputs = 4;
  } else if (type == "update_gltf") {
    minInputs = maxInputs = 3;
  } else if (type == "render_text") {
    minInputs = maxInputs = 6;
  } else if (type == "scenegraph_visibility" || type == "update_component") {
    minInputs = maxInputs = 2;
  } else if (type == "microphone") {
    minOutputs = maxOutputs = 4;
  } else if (type == "depth") {
    minOutputs = maxOutputs = 1;
  } else if (type == "speaker") {
    minInputs = maxInputs = 1;
  } else if (type == "javascript") {
    maxInputs = std::numeric_limits<size_t>::max();
    minOutputs = 1; maxOutputs = std::numeric_limits<size_t>::max();
  } else if (type == "run_algorithm") {
    minInputs = 1; maxInputs = std::numeric_limits<size_t>::max();
    minOutputs = 1; maxOutputs = std::numeric_limits<size_t>::max();
  } else {
    known = false;
  }
  if (!known) return;
  if (inputCount < minInputs || inputCount > maxInputs || outputCount < minOutputs || outputCount > maxOutputs) {
    throw std::runtime_error(Fmt("%s has invalid arity: inputs=%zu outputs=%zu", type.c_str(), inputCount, outputCount));
  }
}

#endif  // SECUREMR_SERIALIZATION_PARSE_ONLY

}  // namespace

Json TensorAttributeToJson(const TensorAttribute& attr) {
  Json j;
  j["dimensions"] = attr.dimensions;
  j["channels"] = attr.channels;
  j["usage"] = static_cast<int>(attr.usage);
  j["data_type"] = static_cast<int>(attr.dataType);
  return j;
}

Json TensorAttributeVariantToJson(const std::variant<std::monostate, TensorAttribute>& attr) {
  Json j;
  if (std::holds_alternative<TensorAttribute>(attr)) {
    j = TensorAttributeToJson(std::get<TensorAttribute>(attr));
  } else {
    j["is_gltf"] = true;
  }
  return j;
}

Json TensorListToJson(const std::vector<std::string>& tensors) {
  Json arr = Json::array();
  for (const auto& name : tensors) {
    arr.push_back(name);
  }
  return arr;
}

Json MappedTensorListToJson(const std::vector<std::pair<std::string, std::string>>& mapping) {
  Json arr = Json::array();
  for (const auto& [alias, tensor] : mapping) {
    Json entry;
    entry["name"] = alias;
    entry["tensor"] = tensor;
    arr.push_back(entry);
  }
  return arr;
}

void SetInputs(Json& spec, const std::vector<std::string>& inputs) {
  spec["inputs"] = TensorListToJson(inputs);
}

void SetOutputs(Json& spec, const std::vector<std::string>& outputs) {
  spec["outputs"] = TensorListToJson(outputs);
}

bool WriteJsonToFile(const std::filesystem::path& filePath, const Json& spec) {
#ifdef SECUREMR_SERIALIZATION_PARSE_ONLY
  (void)filePath;
  (void)spec;
  return false;
#else
  if (filePath.empty()) {
    Log::Write(Log::Level::Error, "WriteJsonToFile failed: writable path unavailable");
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(filePath.parent_path(), ec);
  std::ofstream ofs(filePath);
  if (!ofs) {
    Log::Write(Log::Level::Error,
               Fmt("WriteJsonToFile failed: cannot open %s", filePath.string().c_str()));
    return false;
  }
  ofs << spec.dump(2);
  return true;
#endif
}

bool JsonToTensorAttribute(const Json& j, TensorAttribute& out) {
  if (JsonToSpecialTensorAttribute(j, out)) {
    return true;
  }
  if (j.find("dimensions") == j.end()) {
    return false;
  }
  if (!j["dimensions"].is_array()) {
    return false;
  }
  out.dimensions.clear();
  for (const auto& dim : j["dimensions"]) {
    if (!dim.is_number_integer()) {
      return false;
    }
    out.dimensions.push_back(dim.get<int>());
  }
  out.channels = static_cast<int8_t>(j.value("channels", 1));
  out.usage = j.contains("usage") ? ParseTensorUsage(j["usage"]) : XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  out.dataType = j.contains("data_type") ? ParseDataType(j["data_type"]) : XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  return true;
}

std::vector<std::string> ParseTensorList(const Json& arr) {
  if (!arr.is_array()) {
    throw std::runtime_error("tensor references must be arrays");
  }
  std::vector<std::string> tensors;
  tensors.reserve(arr.size());
  for (const auto& each : arr) {
    if (each.is_string()) {
      tensors.push_back(each.get<std::string>());
    } else if (each.is_object() && each.contains("tensor") && each["tensor"].is_string()) {
      tensors.push_back(each["tensor"].get<std::string>());
    } else {
      throw std::runtime_error("tensor references must be strings or objects with string tensor");
    }
  }
  return tensors;
}

std::vector<std::pair<std::string, std::string>> ParseMappedTensorList(const Json& arr) {
  if (!arr.is_array()) {
    throw std::runtime_error("mapped tensor references must be arrays");
  }
  std::vector<std::pair<std::string, std::string>> mapping;
  mapping.reserve(arr.size());
  for (const auto& each : arr) {
    std::string tensorName;
    std::string alias;
    if (each.is_object()) {
      if (auto aliasIt = each.find("name"); aliasIt != each.end() && aliasIt->is_string()) {
        alias = aliasIt->get<std::string>();
      }
      if (auto tensorIt = each.find("tensor"); tensorIt != each.end() && tensorIt->is_string()) {
        tensorName = tensorIt->get<std::string>();
      }
    } else if (each.is_string()) {
      tensorName = each.get<std::string>();
      alias = tensorName;
    }
    if (!tensorName.empty()) {
      if (alias.empty()) {
        alias = tensorName;
      }
      mapping.emplace_back(alias, tensorName);
    } else {
      throw std::runtime_error("mapped tensor references require tensor and optional name");
    }
  }
  return mapping;
}

struct TensorSlot {
  std::string name;
  std::string tensor;
  bool connected = false;
};

std::vector<TensorSlot> ParseOperatorTensorSlots(const Json& arr, const char* key) {
  if (!arr.is_array()) {
    throw std::runtime_error(Fmt("%s must be an array", key));
  }
  std::vector<TensorSlot> slots;
  slots.reserve(arr.size());
  for (size_t index = 0; index < arr.size(); ++index) {
    const auto& entry = arr[index];
    if (entry.is_null()) {
      slots.push_back({});
      continue;
    }
    if (!entry.is_object()) {
      throw std::runtime_error(Fmt("%s[%zu] must be an object or null", key, index));
    }
    for (auto unexpected = entry.begin(); unexpected != entry.end(); ++unexpected) {
      if (unexpected.key() != "tensor" && unexpected.key() != "name") {
        throw std::runtime_error(Fmt("%s[%zu] has unknown key '%s'", key, index, unexpected.key().c_str()));
      }
    }
    if (auto unexpected = entry.find("name");
        unexpected != entry.end() && !unexpected->is_string()) {
      throw std::runtime_error(Fmt("%s[%zu].name must be a string", key, index));
    }
    const auto tensorIt = entry.find("tensor");
    if (tensorIt == entry.end() || !tensorIt->is_string() || tensorIt->get<std::string>().empty()) {
      throw std::runtime_error(Fmt("%s[%zu] requires a non-empty tensor string", key, index));
    }
    TensorSlot slot;
    slot.tensor = tensorIt->get<std::string>();
    slot.name = entry.contains("name") ? entry["name"].get<std::string>() : slot.tensor;
    slot.connected = true;
    if (slot.name.empty()) {
      throw std::runtime_error(Fmt("%s[%zu].name must not be empty", key, index));
    }
    slots.push_back(std::move(slot));
  }
  return slots;
}

void ValidateOperatorSlots(const std::string& type, const std::vector<TensorSlot>& inputs,
                           const std::vector<TensorSlot>& outputs) {
  auto requireConnected = [&](const std::vector<TensorSlot>& slots, size_t index, const char* kind) {
    if (index >= slots.size() || !slots[index].connected) {
      throw std::runtime_error(Fmt("%s requires %s[%zu]", type.c_str(), kind, index));
    }
  };
  auto requireNullableOnly = [&](const std::vector<TensorSlot>& slots, const std::set<size_t>& nullable,
                                 const char* kind) {
    for (size_t index = 0; index < slots.size(); ++index) {
      if (!slots[index].connected && nullable.find(index) == nullable.end()) {
        throw std::runtime_error(Fmt("%s does not allow null %s slot %zu", type.c_str(), kind, index));
      }
    }
  };
  auto requireAnyOutput = [&](const char* message) {
    if (std::none_of(outputs.begin(), outputs.end(), [](const TensorSlot& slot) { return slot.connected; })) {
      throw std::runtime_error(message);
    }
  };

  if (type == "camera_space_to_world") {
    requireConnected(inputs, 0, "inputs");
    requireConnected(outputs, 0, "outputs");
    requireNullableOnly(outputs, {1}, "outputs");
  } else if (type == "normalize") {
    requireConnected(inputs, 0, "inputs");
    requireConnected(outputs, 0, "outputs");
    requireNullableOnly(inputs, {1}, "inputs");
  } else if (type == "arithmetic") {
    requireConnected(outputs, 0, "outputs");
  } else if (type == "get_transform_mat") {
    requireConnected(inputs, 0, "inputs");
    requireConnected(inputs, 1, "inputs");
    requireNullableOnly(inputs, {2}, "inputs");
    requireConnected(outputs, 0, "outputs");
  } else if (type == "nms") {
    requireConnected(inputs, 0, "inputs");
    requireConnected(inputs, 1, "inputs");
    requireNullableOnly(outputs, {0, 1, 2}, "outputs");
    requireAnyOutput("nms requires at least one connected output slot");
  } else if (type == "sort_vec" || type == "sort_mat") {
    requireConnected(inputs, 0, "inputs");
    requireNullableOnly(outputs, {0, 1}, "outputs");
    requireAnyOutput(type == "sort_vec" ? "sort_vec requires at least one connected output slot"
                                             : "sort_mat requires at least one connected output slot");
  } else if (type == "svd") {
    requireConnected(inputs, 0, "inputs");
    requireNullableOnly(outputs, {0, 1, 2}, "outputs");
    requireAnyOutput("svd requires at least one connected output slot");
  } else if (type == "microphone") {
    requireNullableOnly(outputs, {0, 1, 2, 3}, "outputs");
    requireAnyOutput("microphone requires at least one connected output slot");
  } else if (type == "switch_gltf_render_status" || type == "update_gltf") {
    requireConnected(inputs, 0, "inputs");
    requireNullableOnly(inputs, type == "switch_gltf_render_status" ? std::set<size_t>{1, 2, 3}
                                                                         : std::set<size_t>{1, 2},
                       "inputs");
  } else if (type == "scenegraph_visibility") {
    requireConnected(inputs, 0, "inputs");
    requireNullableOnly(inputs, {1}, "inputs");
  } else if (type == "update_component") {
    requireConnected(inputs, 0, "inputs");
    requireConnected(inputs, 1, "inputs");
  } else if (type == "javascript") {
    requireNullableOnly(inputs, {}, "inputs");
    requireNullableOnly(outputs, {}, "outputs");
  } else if (type == "run_algorithm") {
    requireNullableOnly(inputs, {}, "inputs");
    requireNullableOnly(outputs, {}, "outputs");
  } else {
    for (size_t index = 0; index < inputs.size(); ++index) {
      requireConnected(inputs, index, "inputs");
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
      requireConnected(outputs, index, "outputs");
    }
  }
}

void ValidateTensorSpec(const std::string& tensorName, const Json& tensorSpec) {
  if (!tensorSpec.is_object()) {
    throw std::runtime_error(Fmt("tensor '%s' descriptor must be an object", tensorName.c_str()));
  }

  static const std::set<std::string> kAllowedKeys = {
      "dimensions", "channels", "data_type", "is_placeholder", "usage",
      "flag", "data", "is_gltf", "asset",
  };
  for (auto it = tensorSpec.begin(); it != tensorSpec.end(); ++it) {
    if (kAllowedKeys.find(it.key()) == kAllowedKeys.end()) {
      throw std::runtime_error(Fmt("tensor '%s' has unknown key '%s'", tensorName.c_str(), it.key().c_str()));
    }
  }

  for (const char* key : {"dimensions", "channels", "data_type", "is_placeholder", "usage"}) {
    if (!tensorSpec.contains(key)) {
      throw std::runtime_error(Fmt("tensor '%s' missing required key '%s'", tensorName.c_str(), key));
    }
  }
  if (!tensorSpec["dimensions"].is_array() || tensorSpec["dimensions"].empty()) {
    throw std::runtime_error(Fmt("tensor '%s' dimensions must be a non-empty array of positive integers",
                                 tensorName.c_str()));
  }
  for (const auto& dimension : tensorSpec["dimensions"]) {
    if (!dimension.is_number_integer() || dimension.get<int64_t>() <= 0 ||
        dimension.get<int64_t>() > std::numeric_limits<int>::max()) {
      throw std::runtime_error(Fmt("tensor '%s' dimensions must be positive integers", tensorName.c_str()));
    }
  }
  if (!tensorSpec["channels"].is_number_integer() || tensorSpec["channels"].get<int64_t>() <= 0 ||
      tensorSpec["channels"].get<int64_t>() > std::numeric_limits<int8_t>::max()) {
    throw std::runtime_error(Fmt("tensor '%s' channels must be a positive integer no greater than %d",
                                 tensorName.c_str(), std::numeric_limits<int8_t>::max()));
  }
  if (!tensorSpec["data_type"].is_number_integer() ||
      tensorSpec["data_type"].get<int>() < XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO ||
      tensorSpec["data_type"].get<int>() > XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO) {
    throw std::runtime_error(Fmt("tensor '%s' data_type must be an integer code from 1 through 7",
                                 tensorName.c_str()));
  }
  if (!tensorSpec["is_placeholder"].is_boolean()) {
    throw std::runtime_error(Fmt("tensor '%s' is_placeholder must be a bool", tensorName.c_str()));
  }
  if (!tensorSpec["usage"].is_number_integer() ||
      tensorSpec["usage"].get<int>() < XR_SECURE_MR_TENSOR_TYPE_POINT_PICO ||
      tensorSpec["usage"].get<int>() > XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO) {
    throw std::runtime_error(Fmt("tensor '%s' usage must be an integer code from 1 through 8",
                                 tensorName.c_str()));
  }
  if (tensorSpec.contains("flag") && !tensorSpec["flag"].is_number_integer()) {
    throw std::runtime_error(Fmt("tensor '%s' flag must be an integer", tensorName.c_str()));
  }
  if (tensorSpec.contains("is_gltf") && !tensorSpec["is_gltf"].is_boolean()) {
    throw std::runtime_error(Fmt("tensor '%s' is_gltf must be a bool", tensorName.c_str()));
  }
  if (tensorSpec.contains("asset")) {
    if (!tensorSpec["asset"].is_string()) {
      throw std::runtime_error(Fmt("tensor '%s' asset must be a string", tensorName.c_str()));
    }
  }
  const bool isGltf = tensorSpec.value("is_gltf", false);
  if (isGltf && !tensorSpec["is_placeholder"].get<bool>()) {
    throw std::runtime_error(Fmt("glTF tensor '%s' requires is_placeholder=true", tensorName.c_str()));
  }
  if (isGltf && tensorSpec["usage"].get<int>() != XR_SECURE_MR_TENSOR_TYPE_GLTF_PICO) {
    throw std::runtime_error(Fmt("glTF tensor '%s' must use usage %d", tensorName.c_str(),
                                 XR_SECURE_MR_TENSOR_TYPE_GLTF_PICO));
  }
  if (!isGltf) {
    const int usage = tensorSpec["usage"].get<int>();
    const int channels = tensorSpec["channels"].get<int>();
    const size_t rank = tensorSpec["dimensions"].size();
    if (usage == XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO && (channels != 1 || rank != 1)) {
      throw std::runtime_error(Fmt("tensor '%s' SCALAR usage requires one channel and one dimension",
                                   tensorName.c_str()));
    }
    if (usage == XR_SECURE_MR_TENSOR_TYPE_POINT_PICO &&
        ((channels != 2 && channels != 3) || rank != 1)) {
      throw std::runtime_error(Fmt("tensor '%s' POINT usage requires 2 or 3 channels and one dimension",
                                   tensorName.c_str()));
    }
    if (usage == XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO &&
        ((channels != 3 && channels != 4) || rank != 1 ||
         tensorSpec["data_type"].get<int>() != XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO)) {
      throw std::runtime_error(Fmt("tensor '%s' COLOR usage requires 3 or 4 UINT8 channels and one dimension",
                                   tensorName.c_str()));
    }
    if (usage == XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO &&
        (channels != 4 || rank != 1 || tensorSpec["dimensions"][0].get<int>() != 1 ||
         tensorSpec["data_type"].get<int>() != XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO)) {
      throw std::runtime_error(Fmt("tensor '%s' TIMESTAMP usage requires dimensions [1], 4 channels, and INT32 data",
                                   tensorName.c_str()));
    }
    if (usage == XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO &&
        ((channels != 2 && channels != 3) || rank != 1 ||
         tensorSpec["data_type"].get<int>() != XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO)) {
      throw std::runtime_error(Fmt("tensor '%s' SLICE usage requires 2 or 3 INT32 channels and one dimension",
                                   tensorName.c_str()));
    }
  }
  if (tensorSpec.contains("data")) {
    if (!tensorSpec["data"].is_array()) {
      throw std::runtime_error(Fmt("tensor '%s' data must be an array", tensorName.c_str()));
    }
    for (const auto& value : tensorSpec["data"]) {
      if (!value.is_number()) {
        throw std::runtime_error(Fmt("tensor '%s' data must contain only numeric values", tensorName.c_str()));
      }
    }
    size_t expectedValues = static_cast<size_t>(tensorSpec["channels"].get<int>());
    for (const auto& dimension : tensorSpec["dimensions"]) {
      const size_t value = static_cast<size_t>(dimension.get<int>());
      if (expectedValues > std::numeric_limits<size_t>::max() / value) {
        throw std::runtime_error(Fmt("tensor '%s' dimensions overflow preload element count", tensorName.c_str()));
      }
      expectedValues *= value;
    }
    if (tensorSpec["data"].size() != expectedValues) {
      throw std::runtime_error(Fmt("tensor '%s' data must contain exactly %zu values",
                                   tensorName.c_str(), expectedValues));
    }
  }
  if (tensorSpec["usage"].get<int>() == XR_SECURE_MR_TENSOR_TYPE_MAT_PICO &&
      tensorSpec["dimensions"].size() < 2) {
    throw std::runtime_error(Fmt("tensor '%s' MAT usage requires at least two dimensions", tensorName.c_str()));
  }
}

void ValidateTopLevelTensorLists(const Json& spec, const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& tensorMap) {
  for (const char* key : {"inputs", "outputs"}) {
    const auto it = spec.find(key);
    if (it == spec.end() || !it->is_array()) {
      throw std::runtime_error(Fmt("pipeline %s must be an array", key));
    }
    for (const auto& value : *it) {
      if (!value.is_string() || value.get<std::string>().empty()) {
        throw std::runtime_error(Fmt("pipeline %s must contain non-empty tensor names", key));
      }
      if (tensorMap.find(value.get<std::string>()) == tensorMap.end()) {
        throw std::runtime_error(Fmt("pipeline %s references unknown tensor '%s'", key,
                                     value.get<std::string>().c_str()));
      }
    }
  }
}

void ValidateOperatorTensorReferences(const std::vector<TensorSlot>& slots, const char* key,
                                      const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& tensorMap) {
  for (size_t index = 0; index < slots.size(); ++index) {
    if (slots[index].connected && tensorMap.find(slots[index].tensor) == tensorMap.end()) {
      throw std::runtime_error(Fmt("%s[%zu] references unknown tensor '%s'", key, index,
                                   slots[index].tensor.c_str()));
    }
  }
}

void ValidateModelMetadata(const Json& model, const std::vector<TensorSlot>& inputs,
                           const std::vector<TensorSlot>& outputs) {
  if (!model.is_object()) {
    throw std::runtime_error("run_algorithm model must be an object");
  }
  for (const char* key : {"bin_path", "model_type", "model_target", "input", "output"}) {
    if (!model.contains(key)) {
      throw std::runtime_error(Fmt("run_algorithm model metadata missing required key '%s'", key));
    }
  }
  if (!model["bin_path"].is_string() || model["bin_path"].get<std::string>().empty()) {
    throw std::runtime_error("run_algorithm model.bin_path must be a non-empty string");
  }
  if (!model["model_type"].is_string() || ToLower(model["model_type"].get<std::string>()) != "tflite") {
    throw std::runtime_error("run_algorithm model.model_type must be 'tflite'");
  }
  if (!model["model_target"].is_string()) {
    throw std::runtime_error("run_algorithm model.model_target must be a string");
  }
  const std::string target = ToLower(model["model_target"].get<std::string>());
  if (target != "cpu" && target != "gpu" && target != "npu") {
    throw std::runtime_error("run_algorithm model.model_target must be cpu, gpu, or npu");
  }
  if (auto modelName = model.find("model_name"); modelName != model.end() &&
      (!modelName->is_string() || !IsValidModelName(modelName->get<std::string>()))) {
    throw std::runtime_error(
        "run_algorithm model.model_name must contain only letters, digits, and underscores");
  }
  for (const char* key : {"input", "output"}) {
    if (!model[key].is_array()) {
      throw std::runtime_error(Fmt("run_algorithm model.%s must be an array", key));
    }
    for (size_t index = 0; index < model[key].size(); ++index) {
      const auto& entry = model[key][index];
      if (!entry.is_object() || !entry.contains("name") || !entry.contains("shape") ||
          !entry.contains("encoding_type") || !entry["name"].is_string() ||
          !entry["shape"].is_array() || !entry["encoding_type"].is_string()) {
        throw std::runtime_error(Fmt("run_algorithm model.%s[%zu] requires name, shape, and encoding_type",
                                     key, index));
      }
      const std::string bindingName = entry["name"].get<std::string>();
      if (bindingName.empty() || bindingName.find('\0') != std::string::npos ||
          bindingName.size() >= XR_MAX_OPERATOR_NODE_NAME_PICO) {
        throw std::runtime_error(Fmt(
            "run_algorithm model.%s[%zu].name must be non-empty, contain no NUL bytes, and be shorter than %d bytes",
            key, index, XR_MAX_OPERATOR_NODE_NAME_PICO));
      }
      const auto& slots = std::string(key) == "input" ? inputs : outputs;
      if (index >= slots.size() || bindingName != slots[index].name) {
        throw std::runtime_error(Fmt(
            "run_algorithm model.%s[%zu].name must match the operator binding name", key, index));
      }
      for (const auto& dimension : entry["shape"]) {
        if (!dimension.is_number_integer() || dimension.get<int64_t>() <= 0) {
          throw std::runtime_error(Fmt("run_algorithm model.%s[%zu].shape must contain positive integers",
                                       key, index));
        }
      }
    }
  }
  if (inputs.size() != model["input"].size() || outputs.size() != model["output"].size()) {
    throw std::runtime_error(Fmt("run_algorithm model expects %zu input(s)/%zu output(s), got %zu/%zu",
                                 model["input"].size(), model["output"].size(), inputs.size(), outputs.size()));
  }
}

TensorAttribute RequireTensorAttribute(
    const std::vector<TensorSlot>& slots, size_t index, const char* kind,
    const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& tensorMap) {
  if (index >= slots.size() || !slots[index].connected) {
    throw std::runtime_error(Fmt("%s[%zu] is required", kind, index));
  }
  const auto it = tensorMap.find(slots[index].tensor);
  if (it == tensorMap.end() || it->second == nullptr) {
    throw std::runtime_error(Fmt("%s[%zu] references unknown tensor '%s'", kind, index,
                                 slots[index].tensor.c_str()));
  }
  const auto attribute = it->second->getAttribute();
  const auto* tensorAttribute = std::get_if<TensorAttribute>(&attribute);
  if (tensorAttribute == nullptr) {
    throw std::runtime_error(Fmt("%s[%zu] tensor '%s' must be a numeric tensor", kind, index,
                                 slots[index].tensor.c_str()));
  }
  return *tensorAttribute;
}

bool IsFloatingDataType(const TensorAttribute& attribute) {
  return attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO ||
         attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO;
}

bool IsIntegerDataType(const TensorAttribute& attribute) {
  return attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO ||
         attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO ||
         attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO ||
         attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO ||
         attribute.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO;
}

size_t TensorElementCount(const TensorAttribute& attribute) {
  size_t count = 1;
  for (const int dimension : attribute.dimensions) {
    if (dimension <= 0 || count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension)) {
      throw std::runtime_error("tensor dimensions overflow element count");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void RequireUsage(const TensorAttribute& attribute, XrSecureMrTensorTypePICO usage, const char* description) {
  if (attribute.usage != usage) {
    throw std::runtime_error(Fmt("%s must use tensor usage %d", description, static_cast<int>(usage)));
  }
}

void RequireMat(const TensorAttribute& attribute, const char* description) {
  RequireUsage(attribute, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, description);
  if (attribute.dimensions.size() < 2) {
    throw std::runtime_error(Fmt("%s must have at least two dimensions", description));
  }
}

void ValidateOperatorTensorContracts(
    const std::string& type, const std::vector<TensorSlot>& inputs, const std::vector<TensorSlot>& outputs,
    const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& tensorMap) {
  if (type == "elementwise_min" || type == "elementwise_max" || type == "elementwise_multiply" ||
      type == "elementwise_or" || type == "elementwise_and") {
    const auto left = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    const auto right = RequireTensorAttribute(inputs, 1, "inputs", tensorMap);
    const auto result = RequireTensorAttribute(outputs, 0, "outputs", tensorMap);
    if (left.dimensions != right.dimensions || left.dimensions != result.dimensions ||
        left.channels != right.channels || left.channels != result.channels) {
      throw std::runtime_error(Fmt("%s input and output shapes/channels must match", type.c_str()));
    }
    if ((type == "elementwise_or" || type == "elementwise_and") &&
        (!IsIntegerDataType(left) || !IsIntegerDataType(right) || !IsIntegerDataType(result))) {
      throw std::runtime_error(Fmt("%s requires integer tensors", type.c_str()));
    }
  } else if (type == "cvt_color") {
    RequireMat(RequireTensorAttribute(inputs, 0, "inputs", tensorMap), "cvt_color input");
    RequireMat(RequireTensorAttribute(outputs, 0, "outputs", tensorMap), "cvt_color output");
  } else if (type == "normalize") {
    const auto source = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    const auto result = RequireTensorAttribute(outputs, 0, "outputs", tensorMap);
    if (source.usage != result.usage || source.dimensions != result.dimensions || source.channels != result.channels ||
        source.dataType != result.dataType) {
      throw std::runtime_error("normalize source and result tensor types/shapes must match");
    }
    if (inputs.size() > 1 && inputs[1].connected) {
      const auto alphaBeta = RequireTensorAttribute(inputs, 1, "inputs", tensorMap);
      if (!IsFloatingDataType(alphaBeta) || TensorElementCount(alphaBeta) * static_cast<size_t>(alphaBeta.channels) != 2) {
        throw std::runtime_error("normalize alpha_beta must be a floating-point tensor containing exactly two values");
      }
    }
  } else if (type == "norm") {
    const auto source = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    const auto result = RequireTensorAttribute(outputs, 0, "outputs", tensorMap);
    RequireMat(source, "norm input");
    if (!IsFloatingDataType(source) || !IsFloatingDataType(result) || result.channels != 1 ||
        TensorElementCount(result) != 1) {
      throw std::runtime_error("norm requires floating-point input and one-value floating-point result");
    }
  } else if (type == "inversion") {
    const auto source = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    const auto result = RequireTensorAttribute(outputs, 0, "outputs", tensorMap);
    RequireMat(source, "inversion input");
    RequireMat(result, "inversion output");
    if (source.dimensions.size() != 2 || source.dimensions[0] != source.dimensions[1] ||
        result.dimensions != source.dimensions) {
      throw std::runtime_error("inversion requires a square input and identically shaped result");
    }
  } else if (type == "get_transform_mat") {
    for (size_t index = 0; index < 2; ++index) {
      const auto value = RequireTensorAttribute(inputs, index, "inputs", tensorMap);
      RequireMat(value, index == 0 ? "transform rotation" : "transform translation");
      if (!IsFloatingDataType(value) || TensorElementCount(value) != 3) {
        throw std::runtime_error("transform rotation and translation must be 3-value floating-point tensors");
      }
    }
    if (inputs.size() > 2 && inputs[2].connected) {
      const auto scale = RequireTensorAttribute(inputs, 2, "inputs", tensorMap);
      RequireMat(scale, "transform scale");
      if (!IsFloatingDataType(scale) || TensorElementCount(scale) != 3) {
        throw std::runtime_error("transform scale must be a 3-value floating-point tensor");
      }
    }
    const auto result = RequireTensorAttribute(outputs, 0, "outputs", tensorMap);
    RequireMat(result, "transform result");
    if (!IsFloatingDataType(result) || result.channels != 1 || result.dimensions != std::vector<int>({4, 4})) {
      throw std::runtime_error("transform result must be a one-channel floating-point 4x4 matrix");
    }
  } else if (type == "swap_hwc_chw") {
    const auto source = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    const auto result = RequireTensorAttribute(outputs, 0, "outputs", tensorMap);
    RequireMat(source, "CHW_HWC input");
    RequireMat(result, "CHW_HWC output");
    const bool sourceHwc = source.dimensions.size() == 2 && source.channels >= 1;
    const bool sourceChw = source.dimensions.size() == 3 && source.channels == 1;
    const bool resultHwc = result.dimensions.size() == 2 && result.channels >= 1;
    const bool resultChw = result.dimensions.size() == 3 && result.channels == 1;
    if ((!sourceHwc || !resultChw) && (!sourceChw || !resultHwc)) {
      throw std::runtime_error("CHW_HWC requires one 2D multi-channel MAT and one 3D one-channel MAT");
    }
    if (source.dataType != result.dataType) {
      throw std::runtime_error("CHW_HWC input and output data types must match");
    }
  } else if (type == "nms") {
    const auto scores = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    const auto boxes = RequireTensorAttribute(inputs, 1, "inputs", tensorMap);
    RequireMat(scores, "NMS scores");
    RequireMat(boxes, "NMS boxes");
    if (!IsFloatingDataType(scores) || !IsFloatingDataType(boxes) ||
        !(scores.dimensions.size() == 2 && (scores.dimensions[0] == 1 || scores.dimensions[1] == 1)) ||
        !(boxes.dimensions.size() == 2 && (boxes.dimensions[0] == 4 || boxes.dimensions[1] == 4 || boxes.channels == 4))) {
      throw std::runtime_error("NMS requires floating-point score and box matrices");
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
      if (!outputs[index].connected) continue;
      const auto result = RequireTensorAttribute(outputs, index, "outputs", tensorMap);
      if (index < 2 && !IsFloatingDataType(result)) {
        throw std::runtime_error("NMS score/box results must be floating-point");
      }
      if (index == 2 && !IsIntegerDataType(result)) {
        throw std::runtime_error("NMS index result must be integer");
      }
    }
  } else if (type == "sort_vec" || type == "sort_mat") {
    const auto source = RequireTensorAttribute(inputs, 0, "inputs", tensorMap);
    if (type == "sort_vec" && source.dimensions.size() != 1) {
      throw std::runtime_error("sort_vec input must be one-dimensional");
    }
    if (type == "sort_mat" && (source.dimensions.size() != 2 || source.channels != 1)) {
      throw std::runtime_error("sort_mat input must be a one-channel 2D matrix");
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
      if (!outputs[index].connected) continue;
      const auto result = RequireTensorAttribute(outputs, index, "outputs", tensorMap);
      if (result.dimensions != source.dimensions || result.channels != 1) {
        throw std::runtime_error(Fmt("%s result shape must match its input", type.c_str()));
      }
      if (index == 1 && !IsIntegerDataType(result)) {
        throw std::runtime_error(Fmt("%s index result must be integer", type.c_str()));
      }
    }
  }
}

bool JsonToFloatArray(const Json& arr, std::array<float, 6>& dest) {
  if (!arr.is_array() || arr.size() != dest.size()) {
    return false;
  }
  for (size_t i = 0; i < dest.size(); ++i) {
    if (!arr[i].is_number()) {
      return false;
    }
    dest[i] = arr[i].get<float>();
  }
  return true;
}

std::vector<float> JsonToFloatVector(const Json& arr, const char* what) {
  if (!arr.is_array() || arr.empty()) {
    throw std::runtime_error(Fmt("%s requires a non-empty float array", what));
  }
  std::vector<float> values;
  values.reserve(arr.size());
  for (const auto& value : arr) {
    if (!value.is_number()) {
      throw std::runtime_error(Fmt("%s contains a non-numeric entry", what));
    }
    values.push_back(value.get<float>());
  }
  return values;
}

Json LoadJsonFromFile(const std::filesystem::path& filePath) {
#ifdef SECUREMR_SERIALIZATION_PARSE_ONLY
  (void)filePath;
  return Json();
#else
  Json parsed;
  if (filePath.empty()) {
    Log::Write(Log::Level::Error, "LoadJsonFromFile failed: path empty");
    return parsed;
  }
  std::ifstream ifs(filePath);
  if (!ifs) {
    Log::Write(Log::Level::Error,
               Fmt("LoadJsonFromFile failed: cannot open %s", filePath.string().c_str()));
    return parsed;
  }
  try {
    ifs >> parsed;
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error, Fmt("LoadJsonFromFile failed: %s", e.what()));
    parsed = Json();
  }
  return parsed;
#endif
}

std::string FormatOperatorType(const std::string& typeName) {
  if (typeName.empty()) {
    return typeName;
  }
  static const std::unordered_map<std::string, std::string> kCanonicalNames = {
      {"XR_SECURE_MR_OPERATOR_TYPE_UNKNOWN_PICO", "unknown"},
      {"XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO", "camera_access"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO", "camera_space_to_world"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO", "uv_to_3d_in_camera_space"},
      {"XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO", "get_affine"},
      {"XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO", "apply_affine"},
      {"XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_POINT_PICO", "apply_affine_point"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", "assignment"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO", "customized_compare"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO", "all"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO", "any"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO", "argmax"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO", "cvt_color"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO", "normalize"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO", "arithmetic"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO", "elementwise_min"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MAX_PICO", "elementwise_max"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MULTIPLY_PICO", "elementwise_multiply"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_OR_PICO", "elementwise_or"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_AND_PICO", "elementwise_and"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO", "nms"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO", "solve_p_n_p"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO", "sort_vec"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO", "sort_mat"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO", "svd"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NORM_PICO", "norm"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CHW_HWC_PICO", "swap_hwc_chw"},
      {"XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO", "inversion"},
      {"XR_SECURE_MR_OPERATOR_TYPE_MAKE_TRANSFORM_MAT_PICO", "get_transform_mat"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UPLOAD_TEXTURE_TO_GLTF_PICO", "load_texture"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO", "switch_gltf_render_status"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO", "update_gltf"},
      {"XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO", "render_text"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO", "scenegraph_visibility"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO", "update_component"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO", "scenegraph_visibility"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO", "update_component"},
      {"XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO", "microphone"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SPEAKER_PICO", "speaker"},
      {"XR_SECURE_MR_OPERATOR_TYPE_DEPTH_PICO", "depth"},
      {"XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO", "javascript"},
      {"XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO", "javascript"},
      {"XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO", "run_algorithm"},
  };

  if (auto it = kCanonicalNames.find(typeName); it != kCanonicalNames.end()) {
    return it->second;
  }
  throw std::runtime_error(Fmt("operator type '%s' is not a supported canonical package operator", typeName.c_str()));
}

void RemoveOperatorsAndPromoteOutputsToInputs(
    Json& pipelineJson, const std::unordered_set<std::string>& operatorTypes) {
  auto operatorsIt = pipelineJson.find("operators");
  if (operatorsIt == pipelineJson.end() || !operatorsIt->is_array()) {
    return;
  }

  Json kept = Json::array();
  for (const auto& opSpec : *operatorsIt) {
    if (!opSpec.is_object() ||
        operatorTypes.find(FormatOperatorType(opSpec.value("type", ""))) == operatorTypes.end()) {
      kept.push_back(opSpec);
      continue;
    }

    Json& inputs = pipelineJson["inputs"];
    if (!inputs.is_array()) {
      inputs = Json::array();
    }
    for (const auto& outputName : ParseTensorList(opSpec.value("outputs", Json::array()))) {
      const bool alreadyInput = std::any_of(inputs.begin(), inputs.end(), [&](const Json& existing) {
        return existing.is_string() && existing.get<std::string>() == outputName;
      });
      if (!alreadyInput) {
        inputs.push_back(outputName);
      }

      auto tensorsIt = pipelineJson.find("tensors");
      if (tensorsIt == pipelineJson.end() || !tensorsIt->is_object()) {
        throw std::runtime_error("cannot promote operator output without a tensors object");
      }
      auto tensorIt = tensorsIt->find(outputName);
      if (tensorIt == tensorsIt->end() || !tensorIt->is_object()) {
        throw std::runtime_error(Fmt("cannot promote unknown operator output '%s'", outputName.c_str()));
      }
      (*tensorIt)["is_placeholder"] = true;
    }
  }
  *operatorsIt = std::move(kept);
}

bool DeserializePipelineFromJson(const Json& spec,
                                 const std::shared_ptr<FrameworkSession>& session,
                                 PipelineDeserializationResult& outResult,
                                 std::string& outError) {
#ifdef SECUREMR_SERIALIZATION_PARSE_ONLY
  (void)spec;
  (void)session;
  outResult = {};
  outError = "DeserializePipelineFromJson is unavailable in parser-only test builds";
  return false;
#else
  outResult = {};
  outError.clear();
  if (!spec.is_object()) {
    outError = "JSON is not an object";
    return false;
  }
  for (auto it = spec.begin(); it != spec.end(); ++it) {
    if (it.key() != "tensors" && it.key() != "operators" && it.key() != "inputs" &&
        it.key() != "outputs") {
      outError = Fmt("pipeline has unknown root key '%s'", it.key().c_str());
      return false;
    }
  }
  for (const char* key : {"tensors", "operators", "inputs", "outputs"}) {
    if (!spec.contains(key)) {
      outError = Fmt("pipeline requires '%s'", key);
      return false;
    }
  }

  const auto tensorsIt = spec.find("tensors");
  if (tensorsIt == spec.end() || !tensorsIt->is_object()) {
    outError = "tensors section missing or invalid";
    return false;
  }

  auto pipeline = std::make_shared<Pipeline>(session);
  for (auto it = tensorsIt->begin(); it != tensorsIt->end(); ++it) {
    const std::string tensorName = it.key();
    const Json& tensorSpec = *it;
    try {
      ValidateTensorSpec(tensorName, tensorSpec);
    } catch (const std::exception& e) {
      outError = e.what();
      return false;
    }
    const bool isPlaceholder = tensorSpec["is_placeholder"].get<bool>();
    const bool isGltf = tensorSpec.value("is_gltf", false);
    std::shared_ptr<PipelineTensor> tensor;
    TensorAttribute attr{};
    try {
      if (isGltf) {
        if (!isPlaceholder) {
          outError = Fmt("glTF tensor '%s' requires is_placeholder=true", tensorName.c_str());
          return false;
        }
        tensor = PipelineTensor::PipelineGLTFPlaceholder(pipeline);
      } else {
        if (!JsonToTensorAttribute(tensorSpec, attr)) {
          outError = Fmt("invalid tensor attribute for %s", tensorName.c_str());
          return false;
        }
        tensor = std::make_shared<PipelineTensor>(pipeline, attr, isPlaceholder);
        if (!isPlaceholder && !isGltf) {
          const Json* valueIt = nullptr;
          if (auto it = tensorSpec.find("value"); it != tensorSpec.end()) {
            valueIt = &(*it);
          } else if (auto it = tensorSpec.find("data"); it != tensorSpec.end()) {
            valueIt = &(*it);
          }
          if (valueIt != nullptr && !valueIt->is_null()) {
            if (!valueIt->is_array()) {
              outError = Fmt("invalid tensor value for %s: expected array", tensorName.c_str());
              return false;
            }
            if (attr.channels <= 0) {
              outError = Fmt("invalid tensor attribute for %s: channels must be positive", tensorName.c_str());
              return false;
            }
            size_t elementCount = 1;
            for (int dim : attr.dimensions) {
              if (dim <= 0) {
                outError = Fmt("invalid tensor attribute for %s: non-positive dimension", tensorName.c_str());
                return false;
              }
              elementCount *= static_cast<size_t>(dim);
            }
            const size_t expectedValues = elementCount * static_cast<size_t>(attr.channels);
            if (valueIt->size() != expectedValues) {
              outError = Fmt("invalid tensor value for %s: expected %zu entries but found %zu", tensorName.c_str(),
                             expectedValues, valueIt->size());
              return false;
            }

            auto ensureInteger = [&](const Json& value, const char* what, int64_t minValue,
                                     int64_t maxValue) -> std::optional<int64_t> {
              if (!value.is_number_integer()) {
                outError = Fmt("invalid tensor value for %s: expected integer for %s", tensorName.c_str(), what);
                return std::nullopt;
              }
              const int64_t numeric = value.get<int64_t>();
              if (numeric < minValue || numeric > maxValue) {
                outError = Fmt("invalid tensor value for %s: %s out of range [%lld, %lld]", tensorName.c_str(), what,
                               static_cast<long long>(minValue), static_cast<long long>(maxValue));
                return std::nullopt;
              }
              return numeric;
            };

            switch (attr.dataType) {
              case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO: {
                std::vector<float> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  const auto& value = (*valueIt)[idx];
                  if (!value.is_number()) {
                    outError = Fmt("invalid tensor value for %s: non-numeric entry at index %zu", tensorName.c_str(),
                                   idx);
                    return false;
                  }
                  buffer[idx] = static_cast<float>(value.get<double>());
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(float));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO: {
                std::vector<double> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  const auto& value = (*valueIt)[idx];
                  if (!value.is_number()) {
                    outError = Fmt("invalid tensor value for %s: non-numeric entry at index %zu", tensorName.c_str(),
                                   idx);
                    return false;
                  }
                  buffer[idx] = value.get<double>();
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(double));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO: {
                std::vector<int32_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "INT32", std::numeric_limits<int32_t>::min(),
                                               std::numeric_limits<int32_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<int32_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(int32_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO: {
                std::vector<int16_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "INT16", std::numeric_limits<int16_t>::min(),
                                               std::numeric_limits<int16_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<int16_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(int16_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO: {
                std::vector<int8_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "INT8", std::numeric_limits<int8_t>::min(),
                                               std::numeric_limits<int8_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<int8_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(int8_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO: {
                std::vector<uint16_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "UINT16", 0, std::numeric_limits<uint16_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<uint16_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(uint16_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO: {
                std::vector<uint8_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "UINT8", 0, std::numeric_limits<uint8_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<uint8_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(uint8_t));
                break;
              }
              default:
                Log::Write(Log::Level::Warning,
                           Fmt("DeserializePipelineFromJson: unsupported data_type %d for tensor %s initial value",
                               static_cast<int>(attr.dataType), tensorName.c_str()));
                break;
            }
          }
        }
      }
    } catch (const std::exception& e) {
      outError = Fmt("failed to create tensor '%s': %s", tensorName.c_str(), e.what());
      return false;
    }
    outResult.tensorMap.emplace(tensorName, std::move(tensor));
  }

  try {
    ValidateTopLevelTensorLists(spec, outResult.tensorMap);
    std::set<std::string> boundaryTensors;
    for (const char* key : {"inputs", "outputs"}) {
      for (const auto& value : spec.at(key)) {
        boundaryTensors.insert(value.get<std::string>());
      }
    }
    for (auto it = tensorsIt->begin(); it != tensorsIt->end(); ++it) {
      const bool isPlaceholder = it.value()["is_placeholder"].get<bool>();
      const bool isBoundary = boundaryTensors.find(it.key()) != boundaryTensors.end();
      if (isBoundary && !isPlaceholder) {
        throw std::runtime_error(Fmt("pipeline boundary tensor '%s' requires is_placeholder=true",
                                     it.key().c_str()));
      }
    }
  } catch (const std::exception& e) {
    outError = e.what();
    return false;
  }

  const auto requireTensor = [&](const std::string& name) -> std::shared_ptr<PipelineTensor> {
    auto it = outResult.tensorMap.find(name);
    if (it == outResult.tensorMap.end()) {
      throw std::runtime_error(Fmt("tensor '%s' not found", name.c_str()));
    }
    return it->second;
  };

  const auto operatorsIt = spec.find("operators");
  if (operatorsIt == spec.end() || !operatorsIt->is_array()) {
    outError = "operators section missing or invalid";
    return false;
  }

  try {
    for (const auto& opSpec : *operatorsIt) {
      RejectLegacyFields(opSpec);
      if (!opSpec.contains("type") || !opSpec["type"].is_string() || opSpec["type"].get<std::string>().empty()) {
        throw std::runtime_error("operator requires non-empty canonical type");
      }
      if (!opSpec.contains("inputs") || !opSpec.contains("outputs")) {
        throw std::runtime_error("operator requires inputs and outputs arrays");
      }
      const std::string rawType = opSpec["type"].get<std::string>();
      const std::string type = FormatOperatorType(rawType);
      if (type != "run_algorithm" && opSpec.contains("model")) {
        throw std::runtime_error(Fmt("operator type '%s' does not accept model metadata", rawType.c_str()));
      }
      if (type != "assignment" &&
          (opSpec.contains("src_slices") || opSpec.contains("dst_slices") ||
           opSpec.contains("src_channel_slice") || opSpec.contains("dst_channel_slice"))) {
        throw std::runtime_error(Fmt("operator type '%s' does not accept assignment slice fields", rawType.c_str()));
      }
      const auto inputs = ParseOperatorTensorSlots(opSpec["inputs"], "inputs");
      const auto outputs = ParseOperatorTensorSlots(opSpec["outputs"], "outputs");
      ValidateOperatorTensorReferences(inputs, "inputs", outResult.tensorMap);
      ValidateOperatorTensorReferences(outputs, "outputs", outResult.tensorMap);
      ValidateOperatorArity(type, inputs.size(), outputs.size());
      ValidateOperatorSlots(type, inputs, outputs);
      if (type == "run_algorithm") {
        if (!opSpec.contains("model")) {
          throw std::runtime_error("run_algorithm requires inline model metadata");
        }
        ValidateModelMetadata(opSpec["model"], inputs, outputs);
      }
      ValidateOperatorTensorContracts(type, inputs, outputs, outResult.tensorMap);

      auto requireByIndex = [&](const std::vector<TensorSlot>& container, size_t index,
                                const char* what) -> std::shared_ptr<PipelineTensor> {
        if (index >= container.size() || !container[index].connected) {
          throw std::runtime_error(Fmt("%s index %zu out of range", what, index));
        }
        return requireTensor(container[index].tensor);
      };

      auto optionalByIndex = [&](const std::vector<TensorSlot>& container, size_t index,
                                 const char* what) -> std::shared_ptr<PipelineTensor> {
        if (index >= container.size() || !container[index].connected) {
          return nullptr;
        }
        return requireByIndex(container, index, what);
      };

      auto requireTensorNameByIndex = [&](const std::vector<TensorSlot>& container, size_t index,
                                          const char* what) -> std::string {
        if (index >= container.size() || !container[index].connected) {
          throw std::runtime_error(Fmt("%s index %zu out of range", what, index));
        }
        return container[index].tensor;
      };

      auto mappedSlots = [&](const std::vector<TensorSlot>& container, const char* what) {
        std::vector<std::pair<std::string, std::string>> mapping;
        mapping.reserve(container.size());
        for (size_t idx = 0; idx < container.size(); ++idx) {
          if (!container[idx].connected) {
            throw std::runtime_error(Fmt("%s does not support null slot %zu", what, idx));
          }
          mapping.emplace_back(container[idx].name, container[idx].tensor);
        }
        return mapping;
      };

      auto hasAnyConnectedSlot = [](const std::vector<TensorSlot>& container) {
        return std::any_of(container.begin(), container.end(),
                           [](const TensorSlot& slot) { return slot.connected; });
      };

      if (type == "camera_access") {
        ValidateAttrCount(opSpec, 0, 0, "camera_access");
        pipeline->cameraAccess(requireByIndex(outputs, 0, "camera_access output"),
                               requireByIndex(outputs, 1, "camera_access output"),
                               requireByIndex(outputs, 2, "camera_access output"),
                               requireByIndex(outputs, 3, "camera_access output"));
      } else if (type == "get_affine") {
        ValidateAttrCount(opSpec, 0, 0, "get_affine");
        pipeline->getAffine(requireByIndex(inputs, 0, "get_affine input"),
                            requireByIndex(inputs, 1, "get_affine input"),
                            requireByIndex(outputs, 0, "get_affine output"));
      } else if (type == "apply_affine") {
        ValidateAttrCount(opSpec, 0, 0, "apply_affine");
        pipeline->applyAffine(requireByIndex(inputs, 0, "apply_affine input"),
                              requireByIndex(inputs, 1, "apply_affine input"),
                              requireByIndex(outputs, 0, "apply_affine output"));
      } else if (type == "apply_affine_point") {
        ValidateAttrCount(opSpec, 0, 0, "apply_affine_point");
        pipeline->applyAffinePoint(requireByIndex(inputs, 0, "apply_affine_point input"),
                                   requireByIndex(inputs, 1, "apply_affine_point input"),
                                   requireByIndex(outputs, 0, "apply_affine_point output"));
      } else if (type == "assignment") {
        const auto srcRef = ResolveAssignmentTensorReference(requireTensorNameByIndex(inputs, 0, "assignment input"),
                                                             opSpec, "src_slices", "src_channel_slice", requireTensor);
        const auto dstRef = ResolveAssignmentTensorReference(requireTensorNameByIndex(outputs, 0, "assignment output"),
                                                             opSpec, "dst_slices", "dst_channel_slice", requireTensor);
        if (srcRef.slice.has_value() && dstRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, *dstRef.slice);
        } else if (srcRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, dstRef.tensor);
        } else if (dstRef.slice.has_value()) {
          pipeline->assignment(srcRef.tensor, *dstRef.slice);
        } else {
          pipeline->assignment(srcRef.tensor, dstRef.tensor);
        }
      } else if (type == "cvt_color") {
        ValidateAttrCount(opSpec, 1, 1, "cvt_color");
        const int flag = std::stoi(ReadAttrString(opSpec, 0, "cvt_color", true));
        pipeline->cvtColor(flag, requireByIndex(inputs, 0, "cvt_color input"),
                           requireByIndex(outputs, 0, "cvt_color output"));
      } else if (type == "type_convert") {
        ValidateAttrCount(opSpec, 0, 0, "type_convert");
        const std::string srcName = requireTensorNameByIndex(inputs, 0, "type_convert input");
        const std::string dstName = requireTensorNameByIndex(outputs, 0, "type_convert output");
        const auto srcRef = ResolveTensorReference(srcName, requireTensor);
        const auto dstRef = ResolveTensorReference(dstName, requireTensor);
        if (srcRef.slice.has_value() && dstRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, *dstRef.slice);
        } else if (srcRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, dstRef.tensor);
        } else if (dstRef.slice.has_value()) {
          pipeline->assignment(srcRef.tensor, *dstRef.slice);
        } else {
          pipeline->typeConvert(srcRef.tensor, dstRef.tensor);
        }
      } else if (type == "arithmetic") {
        ValidateAttrCount(opSpec, 1, 1, "arithmetic");
        const std::string expression = ReadAttrString(opSpec, 0, "arithmetic", true);
        std::vector<std::shared_ptr<PipelineTensor>> operands;
        operands.reserve(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
          operands.push_back(optionalByIndex(inputs, idx, "arithmetic input"));
        }
        pipeline->arithmetic(expression, operands, requireByIndex(outputs, 0, "arithmetic output"));
      } else if (type == "elementwise_min" || type == "elementwise_max" ||
                 type == "elementwise_multiply" || type == "elementwise_or" ||
                 type == "elementwise_and") {
        ValidateAttrCount(opSpec, 0, 0, "elementwise");
        const std::string elementwiseOp = type.substr(std::string("elementwise_").size());
        pipeline->elementwise(ParseElementwiseOp(elementwiseOp),
                              {requireByIndex(inputs, 0, "elementwise input"),
                               requireByIndex(inputs, 1, "elementwise input")},
                              requireByIndex(outputs, 0, "elementwise output"));
      } else if (type == "all") {
        ValidateAttrCount(opSpec, 0, 0, "all");
        pipeline->all(requireByIndex(inputs, 0, "all input"), requireByIndex(outputs, 0, "all output"));
      } else if (type == "any") {
        ValidateAttrCount(opSpec, 0, 0, "any");
        pipeline->any(requireByIndex(inputs, 0, "any input"), requireByIndex(outputs, 0, "any output"));
      } else if (type == "nms") {
        auto boxes = requireByIndex(inputs, 1, "nms input");
        ValidateAttrCount(opSpec, 0, 1, "nms");
        const float threshold = std::stof(ReadAttrString(opSpec, 0, "nms", false, "0.95"));
        if (!hasAnyConnectedSlot(outputs)) {
          throw std::runtime_error("nms requires at least one connected output slot");
        }
        pipeline->nms(requireByIndex(inputs, 0, "nms input"), boxes,
                      optionalByIndex(outputs, 0, "nms scores output"),
                      optionalByIndex(outputs, 1, "nms boxes output"),
                      optionalByIndex(outputs, 2, "nms indices output"), threshold);
      } else if (type == "solve_p_n_p") {
        ValidateAttrCount(opSpec, 0, 0, "solve_p_n_p");
        pipeline->solvePnP(requireByIndex(inputs, 0, "solve_p_n_p input"),
                           requireByIndex(inputs, 1, "solve_p_n_p input"),
                           requireByIndex(inputs, 2, "solve_p_n_p input"),
                           requireByIndex(outputs, 0, "solve_p_n_p output"),
                           requireByIndex(outputs, 1, "solve_p_n_p output"));
      } else if (type == "uv_to_3d_in_camera_space") {
        ValidateAttrCount(opSpec, 0, 0, "uv_to_3d_in_camera_space");
        pipeline->uv2Cam(requireByIndex(inputs, 0, "uv_to_3d_in_camera_space input"), requireByIndex(inputs, 1, "uv_to_3d_in_camera_space input"),
                         requireByIndex(inputs, 2, "uv_to_3d_in_camera_space input"), requireByIndex(inputs, 3, "uv_to_3d_in_camera_space input"),
                         requireByIndex(inputs, 4, "uv_to_3d_in_camera_space input"), requireByIndex(outputs, 0, "uv_to_3d_in_camera_space output"));
      } else if (type == "get_transform_mat") {
        ValidateAttrCount(opSpec, 0, 0, "get_transform_mat");
        pipeline->transform(requireByIndex(inputs, 0, "get_transform_mat input"),
                            requireByIndex(inputs, 1, "get_transform_mat input"),
                            optionalByIndex(inputs, 2, "get_transform_mat input"),
                            requireByIndex(outputs, 0, "get_transform_mat output"));
      } else if (type == "camera_space_to_world") {
        ValidateAttrCount(opSpec, 0, 0, "camera_space_to_world");
        pipeline->camSpace2XrLocal(
            requireByIndex(inputs, 0, "camera_space_to_world input"),
            requireByIndex(outputs, 0, "camera_space_to_world right-eye output"),
            optionalByIndex(outputs, 1, "camera_space_to_world left-eye output"));
      } else if (type == "customized_compare") {
        ValidateAttrCount(opSpec, 1, 1, "customized_compare");
        PipelineTensor::Compare compare;
        compare.left = requireByIndex(inputs, 0, "customized_compare input");
        compare.right = requireByIndex(inputs, 1, "customized_compare input");
        compare.comparison = ParseComparison(ReadAttrString(opSpec, 0, "customized_compare", true));
        pipeline->compareTo(compare, requireByIndex(outputs, 0, "customized_compare output"));
      } else if (type == "normalize") {
        ValidateAttrCount(opSpec, 0, 1, "normalize");
        pipeline->normalize(requireByIndex(inputs, 0, "normalize input"), requireByIndex(outputs, 0, "normalize output"),
                            ParseNormalizeType(ReadAttrString(opSpec, 0, "normalize", false, "L2")),
                            optionalByIndex(inputs, 1, "normalize alpha_beta input"));
      } else if (type == "argmax") {
        ValidateAttrCount(opSpec, 0, 0, "argmax");
        pipeline->argMax(requireByIndex(inputs, 0, "argmax input"), requireByIndex(outputs, 0, "argmax output"));
      } else if (type == "sort_vec") {
        ValidateAttrCount(opSpec, 0, 0, "sort_vec");
        if (!hasAnyConnectedSlot(outputs)) {
          throw std::runtime_error("sort_vec requires at least one connected output slot");
        }
        pipeline->sortVec(requireByIndex(inputs, 0, "sort_vector input"),
                          optionalByIndex(outputs, 0, "sort_vector values output"),
                          optionalByIndex(outputs, 1, "sort_vec output"));
      } else if (type == "inversion") {
        ValidateAttrCount(opSpec, 0, 0, "inversion");
        pipeline->inversion(requireByIndex(inputs, 0, "inversion input"),
                            requireByIndex(outputs, 0, "inversion output"));
      } else if (type == "sort_mat") {
        ValidateAttrCount(opSpec, 0, 1, "sort_mat");
        const auto sortType = ParseMatrixSortType(ReadAttrString(opSpec, 0, "sort_mat", false, "ROW"));
        if (!hasAnyConnectedSlot(outputs)) {
          throw std::runtime_error("sort_mat requires at least one connected output slot");
        }
        if (sortType == XR_SECURE_MR_MATRIX_SORT_TYPE_COLUMN_PICO) {
          pipeline->sortMatByColumn(requireByIndex(inputs, 0, "sort_matrix input"),
                                    optionalByIndex(outputs, 0, "sort_matrix values output"),
                                    optionalByIndex(outputs, 1, "sort_matrix indices output"));
        } else {
          pipeline->sortMatByRow(requireByIndex(inputs, 0, "sort_matrix input"),
                                 optionalByIndex(outputs, 0, "sort_matrix values output"),
                                 optionalByIndex(outputs, 1, "sort_matrix indices output"));
        }
      } else if (type == "svd") {
        ValidateAttrCount(opSpec, 0, 0, "svd");
        if (!hasAnyConnectedSlot(outputs)) {
          throw std::runtime_error("svd requires at least one connected output slot");
        }
        pipeline->singularValueDecomposition(
            requireByIndex(inputs, 0, "svd input"), optionalByIndex(outputs, 0, "svd w output"),
            optionalByIndex(outputs, 1, "svd u output"), optionalByIndex(outputs, 2, "svd vt output"));
      } else if (type == "norm") {
        ValidateAttrCount(opSpec, 0, 1, "norm");
        pipeline->norm(requireByIndex(inputs, 0, "norm input"), requireByIndex(outputs, 0, "norm output"));
      } else if (type == "swap_hwc_chw") {
        ValidateAttrCount(opSpec, 0, 0, "swap_hwc_chw");
        pipeline->convertHWC_CHW(requireByIndex(inputs, 0, "swap_hwc_chw input"),
                                 requireByIndex(outputs, 0, "swap_hwc_chw output"));
      } else if (type == "microphone") {
        ValidateAttrCount(opSpec, 1, 1, "microphone");
        std::vector<std::shared_ptr<PipelineTensor>> results;
        results.reserve(outputs.size());
        for (size_t idx = 0; idx < outputs.size(); ++idx) {
          results.push_back(optionalByIndex(outputs, idx, "microphone output"));
        }
        if (!hasAnyConnectedSlot(outputs)) {
          throw std::runtime_error("microphone requires at least one connected output slot");
        }
        const std::string config = ReadAttrString(opSpec, 0, "microphone", true);
        const auto separator = config.find(';');
        if (separator == std::string::npos || separator != config.rfind(';') || separator == 0 ||
            separator + 1 == config.size()) {
          throw std::runtime_error("microphone attrs[0] must be exactly <sample_rate>;<encoding>");
        }

        const std::string sampleRateText = config.substr(0, separator);
        if (!std::all_of(sampleRateText.begin(), sampleRateText.end(), [](const char value) {
              return value >= '0' && value <= '9';
            })) {
          throw std::runtime_error("microphone attrs[0] sample rate must be an integer in the range 8000..96000");
        }

        int32_t sampleRate = 0;
        const auto parseResult = std::from_chars(sampleRateText.data(), sampleRateText.data() + sampleRateText.size(),
                                                 sampleRate);
        if (parseResult.ec != std::errc{} || parseResult.ptr != sampleRateText.data() + sampleRateText.size() ||
            sampleRate < 8000 || sampleRate > 96000) {
          throw std::runtime_error("microphone attrs[0] sample rate must be an integer in the range 8000..96000");
        }

        const XrSecureMrAudioFormatPcmPICO sampleFormat = ParseAudioFormat(config.substr(separator + 1));
        pipeline->microphone(results, sampleFormat, sampleRate);
      } else if (type == "speaker") {
        ValidateAttrCount(opSpec, 1, 1, "speaker");
        pipeline->speaker(requireByIndex(inputs, 0, "speaker input"),
                          std::stoi(ReadAttrString(opSpec, 0, "speaker", true)));
      } else if (type == "depth") {
        ValidateAttrCount(opSpec, 0, 0, "depth");
        pipeline->depth(requireByIndex(outputs, 0, "depth output"));
      } else if (type == "run_algorithm") {
        ValidateAttrCount(opSpec, 0, 0, "run_algorithm");
        auto mappedInputs = mappedSlots(inputs, "run_algorithm inputs");
        auto mappedOutputs = mappedSlots(outputs, "run_algorithm outputs");
        if (mappedInputs.empty() || mappedOutputs.empty()) {
          throw std::runtime_error("run_algorithm inputs/outputs malformed");
        }

        std::vector<std::pair<std::string, std::shared_ptr<PipelineTensor>>> inputBindings;
        std::unordered_map<std::string, std::string> operandAliasing;
        for (const auto& binding : NormalizeModelBindings(mappedInputs)) {
          inputBindings.emplace_back(binding.operatorName, requireTensor(binding.tensorName));
          operandAliasing.emplace(binding.operatorName, binding.modelNodeName);
        }
        std::vector<std::pair<std::string, std::shared_ptr<PipelineTensor>>> outputBindings;
        std::unordered_map<std::string, std::string> resultAliasing;
        for (const auto& binding : NormalizeModelBindings(mappedOutputs)) {
          outputBindings.emplace_back(binding.operatorName, requireTensor(binding.tensorName));
          resultAliasing.emplace(binding.operatorName, binding.modelNodeName);
        }

        const Json* model = nullptr;
        if (auto modelIt = opSpec.find("model"); modelIt != opSpec.end() && modelIt->is_object()) {
          model = &(*modelIt);
        }
        auto modelValue = [&](const char* key, const std::string& fallback = std::string{}) {
          if (model != nullptr) {
            if (auto it = model->find(key); it != model->end() && it->is_string()) {
              return it->get<std::string>();
            }
          }
          return fallback;
        };
        const std::string modelName = NormalizeModelName(modelValue("model_name"));

        // Schema-v2 model metadata is inline under model.bin_path. Package
        // loaders resolve that path before deserialization.
        std::vector<char> modelBuffer;
        if (model != nullptr && model->contains("bin_path") && (*model)["bin_path"].is_string()) {
          const std::string modelPath = (*model)["bin_path"].get<std::string>();
#ifdef XR_USE_PLATFORM_ANDROID
          if (std::filesystem::is_regular_file(std::filesystem::path(modelPath))) {
            std::ifstream input(modelPath, std::ios::binary);
            modelBuffer.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
          } else {
            if (g_assetManager == nullptr) {
              throw std::runtime_error("run_algorithm: AssetManager not available for model.bin_path");
            }
            AAsset* asset = AAssetManager_open(g_assetManager, modelPath.c_str(), AASSET_MODE_BUFFER);
            if (asset == nullptr) {
              throw std::runtime_error(Fmt("run_algorithm: unable to open model asset '%s'", modelPath.c_str()));
            }
            const off_t length = AAsset_getLength(asset);
            modelBuffer.resize(static_cast<size_t>(length));
            const int64_t read = AAsset_read(asset, modelBuffer.data(), length);
            AAsset_close(asset);
            if (read != length) {
              modelBuffer.clear();
              throw std::runtime_error(Fmt("run_algorithm: read %ld of %ld bytes from model asset '%s'",
                                           static_cast<long>(read), static_cast<long>(length), modelPath.c_str()));
            }
          }
          if (modelBuffer.empty()) {
            throw std::runtime_error(Fmt("run_algorithm: model file or asset '%s' is empty or read failed",
                                         modelPath.c_str()));
          }
#else
          std::ifstream ifs(modelPath, std::ios::binary);
          if (!ifs) {
            throw std::runtime_error(Fmt("run_algorithm: cannot open model file '%s'", modelPath.c_str()));
          }
          modelBuffer.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
          if (modelBuffer.empty()) {
            throw std::runtime_error(Fmt("run_algorithm: model file '%s' is empty or read failed", modelPath.c_str()));
          }
#endif
        } else {
          throw std::runtime_error("run_algorithm requires model.bin_path");
        }

        const XrSecureMrModelTypePICO modelType = ParseModelType(modelValue("model_type", "tflite"));
        const XrSecureMrModelTargetPICO modelTarget = ParseModelTarget(modelValue("model_target", "npu"));
        const int32_t cpuTargetNumThreads = model != nullptr ? model->value("cpu_target_num_threads", 1) : 1;
        pipeline->runAlgorithmOrdered(modelBuffer.data(), modelBuffer.size(), inputBindings, operandAliasing,
                                      outputBindings, resultAliasing, modelName, modelType, modelTarget,
                                      cpuTargetNumThreads);
      } else if (type == "javascript") {
        ValidateAttrCount(opSpec, 1, 1, "javascript");
        auto mappedInputs = mappedSlots(inputs, "javascript inputs");
        auto mappedOutputs = mappedSlots(outputs, "javascript outputs");
        std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> scriptOperands;
        for (const auto& [alias, tensorName] : mappedInputs) {
          scriptOperands.emplace(alias, requireTensor(tensorName));
        }
        std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> scriptResults;
        for (const auto& [alias, tensorName] : mappedOutputs) {
          scriptResults.emplace(alias, requireTensor(tensorName));
        }

        std::string script = ReadAttrString(opSpec, 0, "javascript", true);
        if (script.empty()) {
          throw std::runtime_error("javascript requires non-empty attrs[0]");
        }
        pipeline->runJavascript(script.data(), script.size(), scriptOperands, scriptResults);
      } else if (type == "render_text") {
        ValidateAttrCount(opSpec, 1, 1, "render_text");
        const std::string config = ReadAttrString(opSpec, 0, "render_text", true);
        const auto first = config.find('#');
        const auto second = first == std::string::npos ? std::string::npos : config.find('#', first + 1);
        const auto third = second == std::string::npos ? std::string::npos : config.find('#', second + 1);
        if (first == std::string::npos || second == std::string::npos || third == std::string::npos ||
            config.find('#', third + 1) != std::string::npos) {
          throw std::runtime_error("render_text config must be typeface#language#width#height");
        }
        const std::string typeface = config.substr(0, first);
        const std::string language = config.substr(first + 1, second - first - 1);
        const int canvasWidth = std::stoi(config.substr(second + 1, third - second - 1));
        const int canvasHeight = std::stoi(config.substr(third + 1));
        if (typeface.empty() || language.empty() || canvasWidth <= 0 || canvasHeight <= 0) {
          throw std::runtime_error("render_text config must contain a typeface, language, positive width and positive height");
        }
        pipeline->execRenderCommand(std::make_shared<RenderCommand_DrawText>(
            requireByIndex(inputs, 3, "render_text gltf input"), language, ParseTypeFace(typeface), canvasWidth,
            canvasHeight, requireByIndex(inputs, 0, "render_text text input"),
            requireByIndex(inputs, 1, "render_text start input"), requireByIndex(inputs, 5, "render_text font input"),
            requireByIndex(inputs, 2, "render_text colors input"),
            requireByIndex(inputs, 4, "render_text texture input")));
      } else if (type == "load_texture") {
        ValidateAttrCount(opSpec, 0, 0, "load_texture");
        pipeline->newTextureToGLTF(requireByIndex(inputs, 0, "load_texture gltf input"),
                                   requireByIndex(inputs, 1, "load_texture image input"),
                                   requireByIndex(outputs, 0, "load_texture output"));
      } else if (type == "update_gltf") {
        ValidateAttrCount(opSpec, 1, 1, "update_gltf");
        const std::string attribute = ReadAttrString(opSpec, 0, "update_gltf", true);
        if (attribute.empty()) {
          throw std::runtime_error("update_gltf requires non-empty attrs[0]");
        }
        if (attribute == "texture") {
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateTextures>(
              requireByIndex(inputs, 0, "update_gltf gltf input"), requireByIndex(inputs, 1, "update_gltf texture ID"),
              requireByIndex(inputs, 2, "update_gltf rgb image")));
        } else if (attribute == "animation") {
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateAnimation>(
              requireByIndex(inputs, 0, "update_gltf gltf input"), requireByIndex(inputs, 1, "update_gltf animation ID"),
              optionalByIndex(inputs, 2, "update_gltf animation timer") == nullptr
                  ? std::variant<std::shared_ptr<PipelineTensor>, float>{RenderCommand_UpdateAnimation::STOP_TO_PLAY}
                  : std::variant<std::shared_ptr<PipelineTensor>, float>{
                        requireByIndex(inputs, 2, "update_gltf animation timer")}));
        } else if (attribute == "world pose") {
          pipeline->execRenderCommand(
              std::make_shared<RenderCommand_UpdatePose>(requireByIndex(inputs, 0, "update_gltf gltf input"),
                                                         optionalByIndex(inputs, 1, "update_gltf pose input")));
        } else if (attribute == "local") {
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateNodesLocalPoses>(
              requireByIndex(inputs, 0, "update_gltf gltf input"), requireByIndex(inputs, 1, "update_gltf node ID"),
              requireByIndex(inputs, 2, "update_gltf transform input")));
        } else {
          const auto materialAttribute = ParseMaterialAttribute(attribute);
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateMaterial>(
              requireByIndex(inputs, 0, "update_gltf gltf input"), requireByIndex(inputs, 1, "update_gltf material ID"),
              materialAttribute, requireByIndex(inputs, 2, "update_gltf material value")));
        }
      } else if (type == "switch_gltf_render_status") {
        ValidateAttrCount(opSpec, 0, 0, "switch_gltf_render_status");
        auto command = std::make_shared<RenderCommand_Render>();
        command->gltfTensor = requireByIndex(inputs, 0, "switch_gltf_render_status gltf input");
        command->pose = optionalByIndex(inputs, 1, "switch_gltf_render_status pose");
        if (auto viewLocked = optionalByIndex(inputs, 3, "switch_gltf_render_status view locked")) {
          command->viewLocked = viewLocked;
        }
        command->visible = optionalByIndex(inputs, 2, "switch_gltf_render_status visible");
        pipeline->execRenderCommand(command);
      } else if (type == "scenegraph_visibility") {
        ValidateAttrCount(opSpec, 0, 0, "scenegraph_visibility");
        pipeline->scenegraphVisibility(requireByIndex(inputs, 0, "scenegraph_visibility scenegraph input"),
                                       requireByIndex(inputs, 1, "scenegraph_visibility visible input"));
      } else if (type == "update_component") {
        ValidateAttrCount(opSpec, 1, 1, "update_component");
        const std::string componentPath = ReadAttrString(opSpec, 0, "update_component", true);
        if (componentPath.empty() || componentPath.front() != '/' || componentPath.find(':') == std::string::npos) {
          throw std::runtime_error("update_component attrs[0] must be /entity/path:component.field");
        }
        requireByIndex(inputs, 0, "update_component scenegraph input");
        requireByIndex(inputs, 1, "update_component data input");
        throw std::runtime_error("update_component component-path dispatch is unsupported by the native Pipeline API");
      } else {
        throw std::runtime_error(Fmt("unsupported operator type '%s'",
                                     rawType.empty() ? type.c_str() : rawType.c_str()));
      }
    }
  } catch (const std::exception& e) {
    if (outError.empty()) {
      outError = e.what();
    }
    return false;
  }

  outResult.pipeline = std::move(pipeline);
  return true;
#endif
}

}  // namespace SecureMR
