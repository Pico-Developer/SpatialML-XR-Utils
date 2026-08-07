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

#include "securemr_utils/utils.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <exception>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "common.h"
#include "logger.h"
#include "pipeline.h"

#ifdef XR_USE_PLATFORM_ANDROID
#include <android/asset_manager.h>
extern AAssetManager* g_assetManager;
#endif

namespace SecureMR {
namespace {

std::string JoinInts(const std::vector<int>& values) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    out.append(std::to_string(values[i]));
    if (i + 1 < values.size()) {
      out.push_back('x');
    }
  }
  return out;
}

XrSecureMrTensorDataTypePICO MapQnnType(const std::string& type, bool& warnedFloat16) {
  if (type == "QNN_DATATYPE_FLOAT_32") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  }
  if (type == "QNN_DATATYPE_FLOAT_16") {
    warnedFloat16 = true;
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  }
  if (type == "QNN_DATATYPE_INT_32") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO;
  }
  if (type == "QNN_DATATYPE_INT_16") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO;
  }
  if (type == "QNN_DATATYPE_INT_8") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO;
  }
  if (type == "QNN_DATATYPE_UINT_16") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO;
  }
  if (type == "QNN_DATATYPE_UINT_8") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO;
  }
  return XR_SECURE_MR_TENSOR_DATA_TYPE_MAX_ENUM_PICO;
}

std::optional<TensorBinding> BuildBinding(const Json& info) {
  if (!info.is_object()) {
    return std::nullopt;
  }

  TensorBinding binding;
  if (auto nameIt = info.find("name"); nameIt != info.end() && nameIt->is_string()) {
    binding.name = nameIt->get<std::string>();
  }
  if (auto typeIt = info.find("dataType"); typeIt != info.end() && typeIt->is_string()) {
    binding.qnnType = typeIt->get<std::string>();
  }
  if (auto dimsIt = info.find("dimensions"); dimsIt != info.end() && dimsIt->is_array()) {
    for (const auto& dim : *dimsIt) {
      if (dim.is_number_integer()) {
        binding.qnnDims.push_back(dim.get<int>());
      }
    }
  }
  if (binding.name.empty() || binding.qnnType.empty() || binding.qnnDims.empty()) {
    return std::nullopt;
  }

  bool warnedFloat16 = false;
  binding.attr.dataType = MapQnnType(binding.qnnType, warnedFloat16);
  if (binding.attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_MAX_ENUM_PICO) {
    Log::Write(Log::Level::Error,
               Fmt("Tensor %s has unsupported data type %s", binding.name.c_str(), binding.qnnType.c_str()));
    return std::nullopt;
  }

  std::vector<int> dims = binding.qnnDims;
  if (dims.size() > 1 && dims.front() == 1) {
    dims.erase(dims.begin());
  }

  int channels = 1;
  if (dims.size() >= 2) {
    channels = dims.back();
    dims.pop_back();
  } else if (!dims.empty()) {
    channels = 1;
  }
  if (dims.empty()) {
    dims.push_back(1);
  }

  if (channels <= 0 || channels > std::numeric_limits<int8_t>::max()) {
    Log::Write(Log::Level::Error,
               Fmt("Tensor %s has unsupported channel count %d", binding.name.c_str(), channels));
    return std::nullopt;
  }

  if (channels > 4) {
    dims.push_back(channels);
    channels = 1;
  }

  binding.attr.dimensions = dims;
  binding.attr.channels = static_cast<int8_t>(channels);

  if (binding.attr.dimensions.size() <= 1 && binding.attr.channels == 1) {
    binding.attr.usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO;
  } else {
    binding.attr.usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  }

  if (binding.attr.usage == XR_SECURE_MR_TENSOR_TYPE_MAT_PICO && binding.attr.dimensions.size() < 2) {
    binding.attr.dimensions.insert(binding.attr.dimensions.begin(), 1);
    Log::Write(Log::Level::Warning,
               Fmt("Tensor %s mapped to MAT but had 1 dimension; promoting shape to 1x%d to satisfy MAT requirements",
                   binding.name.c_str(), binding.attr.dimensions.back()));
  }

  if (warnedFloat16) {
    Log::Write(Log::Level::Warning,
               Fmt("Tensor %s uses QNN float16; mapping to FLOAT32 for SecureMR tensor", binding.name.c_str()));
  }

  Log::Write(Log::Level::Info,
             Fmt("Tensor %s | qnn dims=%s type=%s -> attr dims=%s channels=%d type=%d",
                 binding.name.c_str(), JoinInts(binding.qnnDims).c_str(), binding.qnnType.c_str(),
                 JoinInts(binding.attr.dimensions).c_str(), binding.attr.channels, binding.attr.dataType));

  return binding;
}

bool ParseBindings(const Json& graphInfo, const char* key, std::vector<TensorBinding>& outBindings) {
  auto tensorsIt = graphInfo.find(key);
  if (tensorsIt == graphInfo.end() || !tensorsIt->is_array()) {
    Log::Write(Log::Level::Error, Fmt("Model JSON missing %s array", key));
    return false;
  }

  for (const auto& entry : *tensorsIt) {
    if (!entry.is_object()) {
      continue;
    }
    auto infoIt = entry.find("info");
    if (infoIt == entry.end()) {
      continue;
    }
    auto binding = BuildBinding(*infoIt);
    if (binding.has_value()) {
      outBindings.emplace_back(std::move(*binding));
    }
  }
  if (outBindings.empty()) {
    Log::Write(Log::Level::Error, Fmt("No valid %s entries found", key));
    return false;
  }
  return true;
}

std::string JoinAssetPath(const std::string& root, const std::string& relativePath) {
  if (root.empty()) {
    return relativePath;
  }
  if (relativePath.empty()) {
    return root;
  }
  if (root.back() == '/') {
    return root + relativePath;
  }
  return root + "/" + relativePath;
}

std::filesystem::path JoinFilePath(const std::filesystem::path& root, const std::string& relativePath) {
  std::filesystem::path path(relativePath);
  if (path.is_absolute()) {
    return path;
  }
  return root / path;
}

std::string ReadStringValue(const Json& object, const std::vector<std::string>& path) {
  const Json* cursor = &object;
  for (const auto& key : path) {
    if (!cursor->is_object()) {
      return {};
    }
    auto it = cursor->find(key);
    if (it == cursor->end()) {
      return {};
    }
    cursor = &(*it);
  }
  return cursor->is_string() ? cursor->get<std::string>() : std::string{};
}

std::string ReadModelValue(const Json& opSpec, const std::string& key) {
  if (auto modelIt = opSpec.find("model"); modelIt != opSpec.end() && modelIt->is_object()) {
    if (auto valueIt = modelIt->find(key); valueIt != modelIt->end() && valueIt->is_string()) {
      return valueIt->get<std::string>();
    }
  }
  if (auto valueIt = opSpec.find(key); valueIt != opSpec.end() && valueIt->is_string()) {
    return valueIt->get<std::string>();
  }
  return {};
}

struct ManifestPipelineSpec {
  std::string id;
  std::string path;
};

std::vector<ManifestPipelineSpec> ReadManifestPipelineSpecs(const Json& manifest) {
  std::vector<ManifestPipelineSpec> specs;
  if (auto pipelinesIt = manifest.find("pipelines"); pipelinesIt != manifest.end() && pipelinesIt->is_array()) {
    for (size_t idx = 0; idx < pipelinesIt->size(); ++idx) {
      const auto& pipelineSpec = (*pipelinesIt)[idx];
      if (!pipelineSpec.is_object()) {
        continue;
      }
      ManifestPipelineSpec spec;
      spec.id = pipelineSpec.value("id", Fmt("pipeline_%zu", idx));
      spec.path = pipelineSpec.value("path", "");
      if (!spec.path.empty()) {
        specs.emplace_back(std::move(spec));
      }
    }
  }

  if (specs.empty()) {
    const std::string legacyPipelinePath = ReadStringValue(manifest, {"pipeline", "path"});
    if (!legacyPipelinePath.empty()) {
      specs.push_back({"detection", legacyPipelinePath});
    }
  }
  return specs;
}

std::string ToLowerLocal(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool ManifestSupportsXr(const Json& manifest) {
  auto runtimeIt = manifest.find("runtime");
  if (runtimeIt == manifest.end() || !runtimeIt->is_object()) {
    return false;
  }
  auto modesIt = runtimeIt->find("supported_modes");
  if (modesIt == runtimeIt->end() || !modesIt->is_array()) {
    return false;
  }
  for (const auto& mode : *modesIt) {
    if (mode.is_string() && ToLowerLocal(mode.get<std::string>()) == "xr") {
      return true;
    }
  }
  return false;
}

std::string AttributeSummary(const TensorAttribute& attr) {
  return Fmt("dims=%s channels=%d usage=%d data_type=%d", JoinInts(attr.dimensions).c_str(), attr.channels,
             static_cast<int>(attr.usage), static_cast<int>(attr.dataType));
}

bool TensorAttributesMatch(const TensorAttribute& left, const TensorAttribute& right) {
  return left.dimensions == right.dimensions && left.channels == right.channels && left.usage == right.usage &&
         left.dataType == right.dataType;
}

template <typename TensorT>
std::optional<TensorAttribute> TensorAttributeFromTensor(const std::shared_ptr<TensorT>& tensor) {
  if (tensor == nullptr) {
    return std::nullopt;
  }
  auto attr = tensor->getAttribute();
  if (!std::holds_alternative<TensorAttribute>(attr)) {
    return std::nullopt;
  }
  return std::get<TensorAttribute>(attr);
}

bool ValidateTensorAttributeMatch(const std::string& tensorName, const std::string& pipelineId,
                                  const std::shared_ptr<PipelineTensor>& expectedTensor,
                                  const std::shared_ptr<GlobalTensor>& existingTensor, std::string& outError) {
  const auto expectedVariant = expectedTensor == nullptr ? std::variant<std::monostate, TensorAttribute>{}
                                                         : expectedTensor->getAttribute();
  const auto existingVariant = existingTensor == nullptr ? std::variant<std::monostate, TensorAttribute>{}
                                                         : existingTensor->getAttribute();
  if (std::holds_alternative<std::monostate>(expectedVariant) &&
      std::holds_alternative<std::monostate>(existingVariant)) {
    return true;
  }
  const auto expectedAttr = std::get_if<TensorAttribute>(&expectedVariant);
  const auto existingAttr = std::get_if<TensorAttribute>(&existingVariant);
  if (expectedAttr == nullptr || existingAttr == nullptr) {
    outError = Fmt("cannot compare attributes for shared tensor '%s' in pipeline '%s'", tensorName.c_str(),
                   pipelineId.c_str());
    return false;
  }
  if (!TensorAttributesMatch(*expectedAttr, *existingAttr)) {
    outError = Fmt("shared tensor '%s' attribute mismatch for pipeline '%s': expected %s but existing global is %s",
                   tensorName.c_str(), pipelineId.c_str(), AttributeSummary(*expectedAttr).c_str(),
                   AttributeSummary(*existingAttr).c_str());
    return false;
  }
  return true;
}

std::string SanitizeModelName(std::string modelName) {
  for (char& ch : modelName) {
    if (ch == '-' || ch == '.' || ch == ' ') {
      ch = '_';
    }
  }
  return modelName;
}

void PatchModelOperators(Json& pipelineJson, const std::string& packageAssetRoot, const std::string& defaultModelPath,
                         const std::string& defaultModelName) {
  auto operatorsIt = pipelineJson.find("operators");
  if (operatorsIt == pipelineJson.end() || !operatorsIt->is_array()) {
    return;
  }
  for (auto& opSpec : *operatorsIt) {
    if (!opSpec.is_object()) {
      continue;
    }
    const std::string type = FormatOperatorType(opSpec.value("type", ""));
    if (type != "run_algorithm") {
      continue;
    }
    std::string modelPath = ReadModelValue(opSpec, "bin_path");
    if (modelPath.empty()) {
      modelPath = ReadModelValue(opSpec, "model_file");
    }
    if (modelPath.empty()) {
      modelPath = defaultModelPath;
    }
    if (!modelPath.empty()) {
      opSpec["model_asset"] = JoinAssetPath(packageAssetRoot, modelPath);
    }
    opSpec.erase("model_file");
    opSpec.erase("model_file_host");

    std::string modelName = ReadModelValue(opSpec, "model_name");
    if (modelName.empty()) {
      modelName = defaultModelName;
    }
    if (!modelName.empty()) {
      opSpec["model_name"] = SanitizeModelName(modelName);
    }

    std::string modelType = ReadModelValue(opSpec, "model_type");
    if (!modelType.empty()) {
      opSpec["model_type"] = modelType;
    }
    std::string modelTarget = ReadModelValue(opSpec, "model_target");
    if (!modelTarget.empty()) {
      opSpec["model_target"] = modelTarget;
    }
  }
}

void PatchModelOperatorsForFiles(Json& pipelineJson, const std::filesystem::path& packageRoot,
                                 const std::string& defaultModelPath, const std::string& defaultModelName) {
  auto operatorsIt = pipelineJson.find("operators");
  if (operatorsIt == pipelineJson.end() || !operatorsIt->is_array()) {
    return;
  }
  for (auto& opSpec : *operatorsIt) {
    if (!opSpec.is_object()) {
      continue;
    }
    const std::string type = FormatOperatorType(opSpec.value("type", ""));
    if (type != "run_algorithm") {
      continue;
    }

    std::string modelPath = ReadModelValue(opSpec, "bin_path");
    if (modelPath.empty()) {
      modelPath = ReadModelValue(opSpec, "model_file");
    }
    if (modelPath.empty()) {
      modelPath = ReadModelValue(opSpec, "model_file_host");
    }
    if (modelPath.empty()) {
      modelPath = defaultModelPath;
    }
    if (!modelPath.empty()) {
      opSpec["model_file"] = JoinFilePath(packageRoot, modelPath).string();
    }
    opSpec.erase("model_asset");
    opSpec.erase("model_file_host");

    std::string modelName = ReadModelValue(opSpec, "model_name");
    if (modelName.empty()) {
      modelName = defaultModelName;
    }
    if (!modelName.empty()) {
      opSpec["model_name"] = SanitizeModelName(modelName);
    }

    std::string modelType = ReadModelValue(opSpec, "model_type");
    if (!modelType.empty()) {
      opSpec["model_type"] = modelType;
    }
    std::string modelTarget = ReadModelValue(opSpec, "model_target");
    if (!modelTarget.empty()) {
      opSpec["model_target"] = modelTarget;
    }
  }
}

void ResolvePackageAssetPaths(Json& pipelineJson, const std::string& packageAssetRoot) {
  if (packageAssetRoot.empty()) {
    return;
  }

  auto tensorsIt = pipelineJson.find("tensors");
  if (tensorsIt == pipelineJson.end() || !tensorsIt->is_object()) {
    return;
  }

  for (auto& tensorSpec : tensorsIt->items()) {
    Json& spec = tensorSpec.value();
    if (!spec.is_object()) {
      continue;
    }
    auto assetIt = spec.find("asset");
    if (assetIt == spec.end() || !assetIt->is_string()) {
      continue;
    }

    const std::string assetPath = assetIt->get<std::string>();
    if (assetPath.empty() || assetPath == packageAssetRoot || assetPath.rfind(packageAssetRoot + "/", 0) == 0) {
      continue;
    }
    spec["asset"] = JoinAssetPath(packageAssetRoot, assetPath);
  }
}

void ResolvePackageFileAssetPaths(Json& pipelineJson, const std::filesystem::path& packageRoot) {
  auto tensorsIt = pipelineJson.find("tensors");
  if (tensorsIt == pipelineJson.end() || !tensorsIt->is_object()) {
    return;
  }

  for (auto& tensorSpec : tensorsIt->items()) {
    Json& spec = tensorSpec.value();
    if (!spec.is_object()) {
      continue;
    }
    auto assetIt = spec.find("asset");
    if (assetIt == spec.end() || !assetIt->is_string()) {
      continue;
    }

    const std::string assetPath = assetIt->get<std::string>();
    if (assetPath.empty()) {
      continue;
    }
    spec["asset"] = JoinFilePath(packageRoot, assetPath).string();
  }
}

void AppendUniqueTensorName(Json& names, const std::string& tensorName) {
  if (tensorName.empty()) {
    return;
  }
  if (!names.is_array()) {
    names = Json::array();
  }
  for (const auto& existing : names) {
    if (existing.is_string() && existing.get<std::string>() == tensorName) {
      return;
    }
  }
  names.push_back(tensorName);
}

void RemoveOperatorsByType(Json& pipelineJson, const std::unordered_set<std::string>& types,
                           const bool promoteRemovedOutputsToInputs = false) {
  auto operatorsIt = pipelineJson.find("operators");
  if (operatorsIt == pipelineJson.end() || !operatorsIt->is_array()) {
    return;
  }
  Json kept = Json::array();
  for (const auto& opSpec : *operatorsIt) {
    if (!opSpec.is_object()) {
      kept.push_back(opSpec);
      continue;
    }
    const std::string type = FormatOperatorType(opSpec.value("type", ""));
    if (types.find(type) == types.end()) {
      kept.push_back(opSpec);
    } else if (promoteRemovedOutputsToInputs) {
      Json& inputs = pipelineJson["inputs"];
      for (const auto& outputName : ParseTensorList(opSpec.value("outputs", Json::array()))) {
        AppendUniqueTensorName(inputs, outputName);
      }
    }
  }
  *operatorsIt = std::move(kept);
}

std::shared_ptr<PipelineTensor> FindPackageTensor(const ModelPackagePipeline& package, const std::string& tensorName) {
  const auto tensorIt = package.tensorMap.find(tensorName);
  if (tensorIt == package.tensorMap.end()) {
    return nullptr;
  }
  return tensorIt->second;
}

std::shared_ptr<GlobalTensor> CreateGlobalLikePipelineTensor(const std::shared_ptr<FrameworkSession>& session,
                                                             const std::shared_ptr<PipelineTensor>& tensor,
                                                             std::string& outError) {
  if (tensor == nullptr) {
    outError = "pipeline tensor is null";
    return nullptr;
  }

  auto attr = tensor->getAttribute();
  if (!std::holds_alternative<TensorAttribute>(attr)) {
    outError = "cannot auto-create global binding for glTF tensor";
    return nullptr;
  }
  return std::make_shared<GlobalTensor>(session, std::get<TensorAttribute>(attr));
}

bool BindPackageGltfAssets(const std::string& pipelineId, const std::shared_ptr<FrameworkSession>& session,
                           ModelPackagePipelineBundle& outBundle, ModelPackagePipeline& package,
                           std::string& outError) {
  auto tensorsIt = package.pipelineJson.find("tensors");
  if (tensorsIt == package.pipelineJson.end() || !tensorsIt->is_object()) {
    return true;
  }

  for (auto it = tensorsIt->begin(); it != tensorsIt->end(); ++it) {
    const std::string tensorName = it.key();
    const Json& tensorSpec = it.value();
    if (!tensorSpec.is_object()) {
      continue;
    }
    const bool isGltf = tensorSpec.value("is_gltf", false) ||
                        (tensorSpec.contains("tensor_type") && tensorSpec["tensor_type"].is_string() &&
                         tensorSpec["tensor_type"].get<std::string>() == "gltf");
    if (!isGltf) {
      continue;
    }

    const std::string assetPath = tensorSpec.value("asset", "");
    if (assetPath.empty()) {
      continue;
    }

    auto pipelineTensor = FindPackageTensor(package, tensorName);
    if (pipelineTensor == nullptr) {
      outError = Fmt("glTF tensor '%s' not found in pipeline '%s'", tensorName.c_str(), pipelineId.c_str());
      return false;
    }

    std::shared_ptr<GlobalTensor> globalTensor;
    if (const auto sharedIt = outBundle.globalTensorMap.find(tensorName); sharedIt != outBundle.globalTensorMap.end()) {
      globalTensor = sharedIt->second;
    } else {
      std::vector<char> gltfData;
      if (!SecureMrUtils::LoadAssetToBuffer(assetPath, gltfData, &outError)) {
        outError = Fmt("failed to load glTF tensor '%s' from '%s' in pipeline '%s': %s", tensorName.c_str(),
                       assetPath.c_str(), pipelineId.c_str(), outError.c_str());
        return false;
      }
      globalTensor = std::make_shared<GlobalTensor>(session, gltfData.data(), gltfData.size());
      outBundle.globalTensorMap[tensorName] = globalTensor;
    }

    package.globalTensorMap[tensorName] = globalTensor;
    package.submitBindings[pipelineTensor] = globalTensor;
  }
  return true;
}

bool BindPackageGltfFiles(const std::string& pipelineId, const std::shared_ptr<FrameworkSession>& session,
                          ModelPackagePipelineBundle& outBundle, ModelPackagePipeline& package,
                          std::string& outError) {
  auto tensorsIt = package.pipelineJson.find("tensors");
  if (tensorsIt == package.pipelineJson.end() || !tensorsIt->is_object()) {
    return true;
  }

  for (auto it = tensorsIt->begin(); it != tensorsIt->end(); ++it) {
    const std::string tensorName = it.key();
    const Json& tensorSpec = it.value();
    if (!tensorSpec.is_object()) {
      continue;
    }
    const bool isGltf = tensorSpec.value("is_gltf", false) ||
                        (tensorSpec.contains("tensor_type") && tensorSpec["tensor_type"].is_string() &&
                         tensorSpec["tensor_type"].get<std::string>() == "gltf");
    if (!isGltf) {
      continue;
    }

    const std::string assetPath = tensorSpec.value("asset", "");
    if (assetPath.empty()) {
      continue;
    }

    auto pipelineTensor = FindPackageTensor(package, tensorName);
    if (pipelineTensor == nullptr) {
      outError = Fmt("glTF tensor '%s' not found in pipeline '%s'", tensorName.c_str(), pipelineId.c_str());
      return false;
    }

    std::shared_ptr<GlobalTensor> globalTensor;
    if (const auto sharedIt = outBundle.globalTensorMap.find(tensorName); sharedIt != outBundle.globalTensorMap.end()) {
      globalTensor = sharedIt->second;
    } else {
      std::ifstream input(assetPath, std::ios::binary);
      if (!input) {
        outError = Fmt("failed to open glTF tensor '%s' from '%s' in pipeline '%s'", tensorName.c_str(),
                       assetPath.c_str(), pipelineId.c_str());
        return false;
      }
      std::vector<char> gltfData{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
      globalTensor = std::make_shared<GlobalTensor>(session, gltfData.data(), gltfData.size());
      outBundle.globalTensorMap[tensorName] = globalTensor;
    }

    package.globalTensorMap[tensorName] = globalTensor;
    package.submitBindings[pipelineTensor] = globalTensor;
  }
  return true;
}

}  // namespace

size_t SecureMrUtils::BytesPerElement(XrSecureMrTensorDataTypePICO dataType) {
  switch (dataType) {
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO:
      return 1;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO:
      return 2;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO:
      return 4;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO:
      return 8;
    default:
      return 0;
  }
}

size_t SecureMrUtils::ElementCount(const TensorAttribute& attr) {
  size_t count = 1;
  for (int dim : attr.dimensions) {
    count *= static_cast<size_t>(dim);
  }
  count *= static_cast<size_t>(attr.channels);
  return count;
}

std::optional<Json> SecureMrUtils::LoadModelJson(const std::filesystem::path& jsonPath) {
  try {
    return LoadJsonFromFile(jsonPath);
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error, Fmt("Failed to read %s: %s", jsonPath.string().c_str(), e.what()));
    return std::nullopt;
  }
}

bool SecureMrUtils::LoadAssetToBuffer(const std::string& assetPath, std::vector<char>& out, std::string* outError) {
  out.clear();
  if (assetPath.empty()) {
    if (outError != nullptr) {
      *outError = "asset path is empty";
    }
    return false;
  }

#ifdef XR_USE_PLATFORM_ANDROID
  if (g_assetManager == nullptr) {
    if (outError != nullptr) {
      *outError = "Android AssetManager is not available";
    }
    return false;
  }

  AAsset* asset = AAssetManager_open(g_assetManager, assetPath.c_str(), AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    if (outError != nullptr) {
      *outError = Fmt("unable to open asset '%s'", assetPath.c_str());
    }
    return false;
  }

  const off_t length = AAsset_getLength(asset);
  if (length < 0) {
    AAsset_close(asset);
    if (outError != nullptr) {
      *outError = Fmt("unable to read asset length for '%s'", assetPath.c_str());
    }
    return false;
  }
  out.resize(static_cast<size_t>(length));
  const int64_t read = AAsset_read(asset, out.data(), length);
  AAsset_close(asset);
  if (read != length) {
    out.clear();
    if (outError != nullptr) {
      *outError = Fmt("read %ld of %ld bytes from asset '%s'", static_cast<long>(read), static_cast<long>(length),
                      assetPath.c_str());
    }
    return false;
  }
  return true;
#else
  std::ifstream ifs(assetPath, std::ios::binary);
  if (!ifs) {
    if (outError != nullptr) {
      *outError = Fmt("unable to open file '%s'", assetPath.c_str());
    }
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  return true;
#endif
}

std::optional<Json> SecureMrUtils::LoadJsonAsset(const std::string& assetPath, std::string* outError) {
  std::vector<char> buffer;
  if (!LoadAssetToBuffer(assetPath, buffer, outError)) {
    return std::nullopt;
  }

  try {
    return Json::parse(buffer.begin(), buffer.end());
  } catch (const std::exception& e) {
    if (outError != nullptr) {
      *outError = Fmt("failed to parse JSON asset '%s': %s", assetPath.c_str(), e.what());
    }
    return std::nullopt;
  }
}

bool SecureMrUtils::PrepareBindings(const Json& jsonSpec,
                                    std::vector<TensorBinding>& inputBindings,
                                    std::vector<TensorBinding>& outputBindings,
                                    std::string& modelName) {
  inputBindings.clear();
  outputBindings.clear();

  auto infoIt = jsonSpec.find("info");
  if (infoIt == jsonSpec.end() || !infoIt->is_object()) {
    Log::Write(Log::Level::Error, "ModelInspect: model JSON missing top-level info");
    return false;
  }
  auto graphsIt = infoIt->find("graphs");
  if (graphsIt == infoIt->end() || !graphsIt->is_array() || graphsIt->empty()) {
    Log::Write(Log::Level::Error, "ModelInspect: model JSON missing graphs array");
    return false;
  }
  const auto& graph = (*graphsIt)[0];
  auto graphInfoIt = graph.find("info");
  if (graphInfoIt == graph.end() || !graphInfoIt->is_object()) {
    Log::Write(Log::Level::Error, "ModelInspect: model JSON graph missing info");
    return false;
  }
  if (auto nameIt = graphInfoIt->find("graphName"); nameIt != graphInfoIt->end() && nameIt->is_string()) {
    modelName = nameIt->get<std::string>();
  }

  if (!ParseBindings(*graphInfoIt, "graphInputs", inputBindings)) {
    return false;
  }
  if (!ParseBindings(*graphInfoIt, "graphOutputs", outputBindings)) {
    return false;
  }
  return true;
}

bool SecureMrUtils::LoadModelPackagePipelinesFromAssets(
    const std::string& packageAssetRoot, const std::shared_ptr<FrameworkSession>& session,
    const std::unordered_map<std::string, std::shared_ptr<GlobalTensor>>& externalGlobals,
    ModelPackagePipelineBundle& outBundle, std::string& outError) {
  outBundle = {};
  outError.clear();
  if (session == nullptr) {
    outError = "FrameworkSession is null";
    return false;
  }

  const std::string manifestAssetPath = JoinAssetPath(packageAssetRoot, "manifest.json");
  auto manifest = LoadJsonAsset(manifestAssetPath, &outError);
  if (!manifest.has_value()) {
    return false;
  }
  outBundle.manifest = *manifest;
  if (!ManifestSupportsXr(outBundle.manifest)) {
    outError = "model package manifest runtime.supported_modes must include xr";
    return false;
  }
  outBundle.detectionTensor = ReadStringValue(outBundle.manifest, {"runtime", "detection_tensor"});

  const auto pipelineSpecs = ReadManifestPipelineSpecs(outBundle.manifest);
  if (pipelineSpecs.empty()) {
    outError = "model package manifest missing pipelines";
    return false;
  }

  const std::string modelBinRelativePath = ReadStringValue(outBundle.manifest, {"model", "bin_path"});

  std::string modelJsonRelativePath = ReadStringValue(outBundle.manifest, {"model", "extra_json_path"});
  if (modelJsonRelativePath.empty()) {
    modelJsonRelativePath = ReadStringValue(outBundle.manifest, {"model", "json_path"});
  }
  std::string modelName;
  if (!modelJsonRelativePath.empty()) {
    const std::string modelJsonAssetPath = JoinAssetPath(packageAssetRoot, modelJsonRelativePath);
    auto modelJson = LoadJsonAsset(modelJsonAssetPath, &outError);
    if (!modelJson.has_value()) {
      return false;
    }
    outBundle.modelJson = *modelJson;
    modelName = SanitizeModelName(outBundle.modelJson.value("model_name", ""));
  }

  auto ensureSharedBinding = [&](const std::string& pipelineId, ModelPackagePipeline& package,
                                 const std::string& tensorName) -> bool {
    auto pipelineTensor = FindPackageTensor(package, tensorName);
    if (pipelineTensor == nullptr) {
      outError = Fmt("model package tensor '%s' not found in pipeline '%s'", tensorName.c_str(), pipelineId.c_str());
      return false;
    }

    std::shared_ptr<GlobalTensor> globalTensor;
    const auto externalIt = externalGlobals.find(tensorName);
    if (externalIt != externalGlobals.end()) {
      globalTensor = externalIt->second;
      if (globalTensor == nullptr) {
        outError = Fmt("external global for tensor '%s' is null", tensorName.c_str());
        return false;
      }
      if (!ValidateTensorAttributeMatch(tensorName, pipelineId, pipelineTensor, globalTensor, outError)) {
        return false;
      }
    } else if (const auto sharedIt = outBundle.globalTensorMap.find(tensorName);
               sharedIt != outBundle.globalTensorMap.end()) {
      globalTensor = sharedIt->second;
      if (!ValidateTensorAttributeMatch(tensorName, pipelineId, pipelineTensor, globalTensor, outError)) {
        return false;
      }
    } else {
      globalTensor = CreateGlobalLikePipelineTensor(session, pipelineTensor, outError);
      if (globalTensor == nullptr) {
        if (outError.empty()) {
          outError = Fmt("failed to create global tensor for '%s' in pipeline '%s'", tensorName.c_str(),
                         pipelineId.c_str());
        } else {
          outError = Fmt("failed to create global tensor for '%s' in pipeline '%s': %s", tensorName.c_str(),
                         pipelineId.c_str(), outError.c_str());
        }
        return false;
      }
    }

    outBundle.globalTensorMap[tensorName] = globalTensor;
    package.globalTensorMap[tensorName] = globalTensor;
    package.submitBindings[pipelineTensor] = globalTensor;
    return true;
  };

  for (const auto& spec : pipelineSpecs) {
    auto pipelineJson = LoadJsonAsset(JoinAssetPath(packageAssetRoot, spec.path), &outError);
    if (!pipelineJson.has_value()) {
      return false;
    }
    PatchModelOperators(*pipelineJson, packageAssetRoot, modelBinRelativePath, modelName);
    ResolvePackageAssetPaths(*pipelineJson, packageAssetRoot);

    PipelineDeserializationResult deserializeResult;
    if (!DeserializePipelineFromJson(*pipelineJson, session, deserializeResult, outError)) {
      outError = Fmt("failed to deserialize pipeline '%s': %s", spec.id.c_str(), outError.c_str());
      return false;
    }

    ModelPackagePipeline package;
    package.manifest = outBundle.manifest;
    package.modelJson = outBundle.modelJson;
    package.pipelineJson = *pipelineJson;
    package.pipeline = std::move(deserializeResult.pipeline);
    package.tensorMap = std::move(deserializeResult.tensorMap);
    package.inputs = ParseTensorList(package.pipelineJson.value("inputs", Json::array()));
    package.outputs = ParseTensorList(package.pipelineJson.value("outputs", Json::array()));
    package.detectionTensor = outBundle.detectionTensor;

    if (!BindPackageGltfAssets(spec.id, session, outBundle, package, outError)) {
      return false;
    }

    for (const auto& tensorName : package.inputs) {
      if (!ensureSharedBinding(spec.id, package, tensorName)) {
        return false;
      }
    }
    for (const auto& tensorName : package.outputs) {
      if (!ensureSharedBinding(spec.id, package, tensorName)) {
        return false;
      }
    }
    if (!package.detectionTensor.empty() && package.tensorMap.find(package.detectionTensor) != package.tensorMap.end() &&
        !ensureSharedBinding(spec.id, package, package.detectionTensor)) {
      return false;
    }

    outBundle.pipelines.emplace(spec.id, std::move(package));
  }

  return true;
}

bool SecureMrUtils::LoadModelPackagePipelinesFromFiles(
    const std::filesystem::path& packageRoot, const std::shared_ptr<FrameworkSession>& session,
    const std::unordered_map<std::string, std::shared_ptr<GlobalTensor>>& externalGlobals,
    ModelPackagePipelineBundle& outBundle, std::string& outError, const ModelPackageLoadOptions& options) {
  outBundle = {};
  outError.clear();
  if (session == nullptr) {
    outError = "FrameworkSession is null";
    return false;
  }

  const auto manifestPath = packageRoot / "manifest.json";
  try {
    outBundle.manifest = LoadJsonFromFile(manifestPath);
  } catch (const std::exception& e) {
    outError = e.what();
    return false;
  }
  if (!ManifestSupportsXr(outBundle.manifest)) {
    outError = "model package manifest runtime.supported_modes must include xr";
    return false;
  }
  outBundle.detectionTensor = ReadStringValue(outBundle.manifest, {"runtime", "detection_tensor"});

  const auto pipelineSpecs = ReadManifestPipelineSpecs(outBundle.manifest);
  if (pipelineSpecs.empty()) {
    outError = "model package manifest missing pipelines";
    return false;
  }

  const std::string defaultModelPath = ReadStringValue(outBundle.manifest, {"model", "bin_path"});
  std::string modelJsonRelativePath = ReadStringValue(outBundle.manifest, {"model", "extra_json_path"});
  if (modelJsonRelativePath.empty()) {
    modelJsonRelativePath = ReadStringValue(outBundle.manifest, {"model", "json_path"});
  }
  std::string modelName;
  if (!modelJsonRelativePath.empty()) {
    try {
      outBundle.modelJson = LoadJsonFromFile(JoinFilePath(packageRoot, modelJsonRelativePath));
      modelName = SanitizeModelName(outBundle.modelJson.value("model_name", ""));
    } catch (const std::exception& e) {
      outError = e.what();
      return false;
    }
  }

  auto ensureSharedBinding = [&](const std::string& pipelineId, ModelPackagePipeline& package,
                                 const std::string& tensorName) -> bool {
    auto pipelineTensor = FindPackageTensor(package, tensorName);
    if (pipelineTensor == nullptr) {
      outError = Fmt("model package tensor '%s' not found in pipeline '%s'", tensorName.c_str(), pipelineId.c_str());
      return false;
    }

    std::shared_ptr<GlobalTensor> globalTensor;
    const auto externalIt = externalGlobals.find(tensorName);
    if (externalIt != externalGlobals.end()) {
      globalTensor = externalIt->second;
      if (globalTensor == nullptr) {
        outError = Fmt("external global for tensor '%s' is null", tensorName.c_str());
        return false;
      }
      if (!ValidateTensorAttributeMatch(tensorName, pipelineId, pipelineTensor, globalTensor, outError)) {
        return false;
      }
    } else if (const auto sharedIt = outBundle.globalTensorMap.find(tensorName);
               sharedIt != outBundle.globalTensorMap.end()) {
      globalTensor = sharedIt->second;
      if (!ValidateTensorAttributeMatch(tensorName, pipelineId, pipelineTensor, globalTensor, outError)) {
        return false;
      }
    } else {
      globalTensor = CreateGlobalLikePipelineTensor(session, pipelineTensor, outError);
      if (globalTensor == nullptr) {
        if (outError.empty()) {
          outError = Fmt("failed to create global tensor for '%s' in pipeline '%s'", tensorName.c_str(),
                         pipelineId.c_str());
        } else {
          outError = Fmt("failed to create global tensor for '%s' in pipeline '%s': %s", tensorName.c_str(),
                         pipelineId.c_str(), outError.c_str());
        }
        return false;
      }
    }

    outBundle.globalTensorMap[tensorName] = globalTensor;
    package.globalTensorMap[tensorName] = globalTensor;
    package.submitBindings[pipelineTensor] = globalTensor;
    return true;
  };

  for (const auto& spec : pipelineSpecs) {
    Json pipelineJson;
    try {
      pipelineJson = LoadJsonFromFile(JoinFilePath(packageRoot, spec.path));
    } catch (const std::exception& e) {
      outError = e.what();
      return false;
    }

    PatchModelOperatorsForFiles(pipelineJson, packageRoot, defaultModelPath, modelName);
    ResolvePackageFileAssetPaths(pipelineJson, packageRoot);
    if (options.stripRectifiedVstAccess) {
      RemoveOperatorsByType(pipelineJson, {"rectified_vst_access", "camera_access"}, true);
    }

    PipelineDeserializationResult deserializeResult;
    if (!DeserializePipelineFromJson(pipelineJson, session, deserializeResult, outError)) {
      outError = Fmt("failed to deserialize pipeline '%s': %s", spec.id.c_str(), outError.c_str());
      return false;
    }

    ModelPackagePipeline package;
    package.manifest = outBundle.manifest;
    package.modelJson = outBundle.modelJson;
    package.pipelineJson = pipelineJson;
    package.pipeline = std::move(deserializeResult.pipeline);
    package.tensorMap = std::move(deserializeResult.tensorMap);
    package.inputs = ParseTensorList(package.pipelineJson.value("inputs", Json::array()));
    package.outputs = ParseTensorList(package.pipelineJson.value("outputs", Json::array()));
    package.detectionTensor = outBundle.detectionTensor;

    if (!BindPackageGltfFiles(spec.id, session, outBundle, package, outError)) {
      return false;
    }

    for (const auto& tensorName : package.inputs) {
      if (!ensureSharedBinding(spec.id, package, tensorName)) {
        return false;
      }
    }
    for (const auto& tensorName : package.outputs) {
      if (!ensureSharedBinding(spec.id, package, tensorName)) {
        return false;
      }
    }
    if (!package.detectionTensor.empty() && package.tensorMap.find(package.detectionTensor) != package.tensorMap.end() &&
        !ensureSharedBinding(spec.id, package, package.detectionTensor)) {
      return false;
    }

    outBundle.pipelines.emplace(spec.id, std::move(package));
  }

  return true;
}

}  // namespace SecureMR
