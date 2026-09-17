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

std::string JoinAssetPath(const std::string& root, const std::string& relativePath) {
  const std::string normalized = NormalizePackageRelativePath(relativePath, "package asset path");
  if (root.empty()) {
    return normalized;
  }
  if (root.back() == '/') {
    return root + normalized;
  }
  return root + "/" + normalized;
}

std::filesystem::path JoinFilePath(const std::filesystem::path& root, const std::string& relativePath) {
  const std::string normalized = NormalizePackageRelativePath(relativePath, "package path");
  const std::filesystem::path resolved = (root / std::filesystem::path(normalized)).lexically_normal();
  const std::filesystem::path normalizedRoot = root.lexically_normal();
  auto rootIt = normalizedRoot.begin();
  auto resolvedIt = resolved.begin();
  for (; rootIt != normalizedRoot.end() && resolvedIt != resolved.end(); ++rootIt, ++resolvedIt) {
    if (*rootIt != *resolvedIt) {
      throw std::runtime_error("package path escapes package root");
    }
  }
  if (rootIt != normalizedRoot.end()) {
    throw std::runtime_error("package path escapes package root");
  }
  return resolved;
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
  return {};
}

struct ManifestPipelineSpec {
  std::string id;
  std::string path;
};

void ValidatePackagePipelinePaths(const Json& pipelineJson) {
  const auto tensorsIt = pipelineJson.find("tensors");
  if (tensorsIt != pipelineJson.end() && tensorsIt->is_object()) {
    for (const auto& item : tensorsIt->items()) {
      if (!item.value().is_object() || !item.value().contains("asset")) {
        continue;
      }
      if (!item.value()["asset"].is_string()) {
        throw std::runtime_error(Fmt("tensor '%s' asset must be a string", item.key().c_str()));
      }
      NormalizePackageRelativePath(item.value()["asset"].get<std::string>(), "tensor asset");
    }
  }

  const auto operatorsIt = pipelineJson.find("operators");
  if (operatorsIt != pipelineJson.end() && operatorsIt->is_array()) {
    for (size_t index = 0; index < operatorsIt->size(); ++index) {
      const auto& op = (*operatorsIt)[index];
      if (!op.is_object() || !op.contains("model")) {
        continue;
      }
      const auto modelIt = op.find("model");
      if (!modelIt->is_object() || !modelIt->contains("bin_path") ||
          !(*modelIt)["bin_path"].is_string()) {
        throw std::runtime_error(Fmt("operator #%zu model.bin_path must be a string", index));
      }
      NormalizePackageRelativePath((*modelIt)["bin_path"].get<std::string>(),
                                  "run_algorithm model.bin_path");
    }
  }
}

std::vector<ManifestPipelineSpec> ReadManifestPipelineSpecs(const Json& manifest) {
  std::vector<ManifestPipelineSpec> specs;
  if (auto pipelinesIt = manifest.find("pipelines"); pipelinesIt != manifest.end() && pipelinesIt->is_array()) {
    for (size_t idx = 0; idx < pipelinesIt->size(); ++idx) {
      const auto& pipelineSpec = (*pipelinesIt)[idx];
      if (!pipelineSpec.is_object()) {
        throw std::runtime_error(Fmt("manifest pipelines[%zu] must be an object", idx));
      }
      ManifestPipelineSpec spec;
      if (!pipelineSpec.contains("id") || !pipelineSpec["id"].is_string() ||
          pipelineSpec["id"].get<std::string>().empty()) {
        throw std::runtime_error(Fmt("manifest pipelines[%zu].id must be a non-empty string", idx));
      }
      if (!pipelineSpec.contains("path") || !pipelineSpec["path"].is_string()) {
        throw std::runtime_error(Fmt("manifest pipelines[%zu].path must be a string", idx));
      }
      spec.id = pipelineSpec["id"].get<std::string>();
      spec.path = NormalizePackageRelativePath(pipelineSpec["path"].get<std::string>(), "manifest pipeline path");
      specs.emplace_back(std::move(spec));
    }
  } else {
    throw std::runtime_error("manifest pipelines must be an array");
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

bool IsSchemaV2(const Json& manifest) {
  return manifest.value("schema_version", std::string{}) == "2" && !manifest.contains("model") &&
         !manifest.contains("models");
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

void PatchModelOperators(Json& pipelineJson, const std::string& packageAssetRoot) {
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
      throw std::runtime_error("run_algorithm requires inline model.bin_path");
    }
    (*opSpec.find("model"))["bin_path"] = JoinAssetPath(packageAssetRoot, modelPath);
  }
}

void PatchModelOperatorsForFiles(Json& pipelineJson, const std::filesystem::path& packageRoot) {
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
      throw std::runtime_error("run_algorithm requires inline model.bin_path");
    }
    (*opSpec.find("model"))["bin_path"] = JoinFilePath(packageRoot, modelPath).string();
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
    spec["asset"] = JoinFilePath(packageRoot, assetPath).string();
  }
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

bool SecureMrUtils::LoadModelPackagePipelinesFromAssets(
    const std::string& packageAssetRoot, const std::shared_ptr<FrameworkSession>& session,
    const std::unordered_map<std::string, std::shared_ptr<GlobalTensor>>& externalGlobals,
    ModelPackagePipelineBundle& outBundle, std::string& outError, const ModelPackageLoadOptions& options) {
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
  if (!IsSchemaV2(outBundle.manifest)) {
    outError = "model package manifest schema_version must be 2";
    return false;
  }
  if (!ManifestSupportsXr(outBundle.manifest)) {
    outError = "model package manifest runtime.supported_modes must include xr";
    return false;
  }
  outBundle.detectionTensor = ReadStringValue(outBundle.manifest, {"runtime", "detection_tensor"});

  std::vector<ManifestPipelineSpec> pipelineSpecs;
  try {
    pipelineSpecs = ReadManifestPipelineSpecs(outBundle.manifest);
  } catch (const std::exception& e) {
    outError = e.what();
    return false;
  }
  if (pipelineSpecs.empty()) {
    outError = "model package manifest missing pipelines";
    return false;
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
    try {
      ValidatePackagePipelinePaths(*pipelineJson);
    } catch (const std::exception& e) {
      outError = Fmt("invalid package paths in pipeline '%s': %s", spec.id.c_str(), e.what());
      return false;
    }
    try {
      PatchModelOperators(*pipelineJson, packageAssetRoot);
    } catch (const std::exception& e) {
      outError = e.what();
      return false;
    }
    ResolvePackageAssetPaths(*pipelineJson, packageAssetRoot);

    PipelineDeserializationResult deserializeResult;
    if (!DeserializePipelineFromJson(*pipelineJson, session, deserializeResult, outError)) {
      outError = Fmt("failed to deserialize pipeline '%s': %s", spec.id.c_str(), outError.c_str());
      return false;
    }

    ModelPackagePipeline package;
    package.manifest = outBundle.manifest;
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
    for (const auto& [tensorName, tensorSpec] : package.pipelineJson["tensors"].items()) {
      if (!tensorSpec.is_object() || !tensorSpec.value("is_placeholder", false)) {
        continue;
      }
      const auto pipelineTensor = FindPackageTensor(package, tensorName);
      if (pipelineTensor == nullptr || package.submitBindings.find(pipelineTensor) == package.submitBindings.end()) {
        if (!ensureSharedBinding(spec.id, package, tensorName)) {
          return false;
        }
      }
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
    if (!IsSchemaV2(outBundle.manifest)) {
      outError = "model package manifest schema_version must be 2";
      return false;
    }
  } catch (const std::exception& e) {
    outError = e.what();
    return false;
  }
  if (!ManifestSupportsXr(outBundle.manifest)) {
    outError = "model package manifest runtime.supported_modes must include xr";
    return false;
  }
  outBundle.detectionTensor = ReadStringValue(outBundle.manifest, {"runtime", "detection_tensor"});

  std::vector<ManifestPipelineSpec> pipelineSpecs;
  try {
    pipelineSpecs = ReadManifestPipelineSpecs(outBundle.manifest);
  } catch (const std::exception& e) {
    outError = e.what();
    return false;
  }
  if (pipelineSpecs.empty()) {
    outError = "model package manifest missing pipelines";
    return false;
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
    try {
      ValidatePackagePipelinePaths(pipelineJson);
    } catch (const std::exception& e) {
      outError = Fmt("invalid package paths in pipeline '%s': %s", spec.id.c_str(), e.what());
      return false;
    }

    try {
      PatchModelOperatorsForFiles(pipelineJson, packageRoot);
    } catch (const std::exception& e) {
      outError = e.what();
      return false;
    }
    ResolvePackageFileAssetPaths(pipelineJson, packageRoot);
    if (options.stripRectifiedVstAccess) {
      RemoveOperatorsAndPromoteOutputsToInputs(pipelineJson, {"camera_access"});
    }

    PipelineDeserializationResult deserializeResult;
    if (!DeserializePipelineFromJson(pipelineJson, session, deserializeResult, outError)) {
      outError = Fmt("failed to deserialize pipeline '%s': %s", spec.id.c_str(), outError.c_str());
      return false;
    }

    ModelPackagePipeline package;
    package.manifest = outBundle.manifest;
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

    for (const auto& [tensorName, tensorSpec] : package.pipelineJson["tensors"].items()) {
      if (!tensorSpec.is_object() || !tensorSpec.value("is_placeholder", false)) {
        continue;
      }
      const auto pipelineTensor = FindPackageTensor(package, tensorName);
      if (pipelineTensor == nullptr || package.submitBindings.find(pipelineTensor) == package.submitBindings.end()) {
        if (!ensureSharedBinding(spec.id, package, tensorName)) {
          return false;
        }
      }
    }

    outBundle.pipelines.emplace(spec.id, std::move(package));
  }

  return true;
}

}  // namespace SecureMR
