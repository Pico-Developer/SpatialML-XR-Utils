# SpatialML XR Utils

Reusable NativeActivity/OpenXR base sources for SpatialML XR Android apps.

This repository packages the `SecureMR-Samples-Private/base` C++ app base as a
small Android Studio library project plus a CMake helper that consuming
NativeActivity apps can use.

## CMake Usage

Add this repository as a submodule in an Android app project, then include the
CMake integration file:

```cmake
include(path/to/SpatialML-XR-Utils/cmake/SpatialMLXRUtils.cmake)

spatialml_xr_add_native_activity(
  TARGET your_native_target
  SOURCES src/main/cpp/your_runner.cpp
)
```

The consuming app still owns its Android manifest and its app-specific
`CreateSecureMrProgram(...)` implementation. The shared helper owns the
NativeActivity entrypoint, OpenXR setup, platform/graphics code, shaders,
OpenXR loader target, and `securemr_utils` sources.

## Package Loading

The utility API includes:

- `SecureMrUtils::LoadModelPackagePipelinesFromAssets(...)`
- `SecureMrUtils::LoadModelPackagePipelinesFromFiles(...)`

The file loader supports pySpatialML schema v2 packages:

- top-level `schema_version: "2"`
- top-level `pipelines: [{id, path}]`
- inline LiteRT/TFLite model metadata on `run_algorithm` operators under
  `operator.model`
- package-relative model paths such as `model/face_detector.tflite`
- package-relative glTF assets such as `gltf/frame.gltf`

When running with supplied image tensors instead of device VST camera input,
pass `ModelPackageLoadOptions{.stripRectifiedVstAccess = true}` to remove
`RECTIFIED_VST_ACCESS` operators before deserialization.
