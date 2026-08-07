set(SPATIALML_XR_UTILS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "SpatialML XR Utils root")

function(spatialml_xr_add_native_activity)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES INCLUDE_DIRS LIBRARIES)
  cmake_parse_arguments(SPATIALML_XR "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT SPATIALML_XR_TARGET)
    message(FATAL_ERROR "spatialml_xr_add_native_activity requires TARGET")
  endif()
  if(NOT SPATIALML_XR_SOURCES)
    message(FATAL_ERROR "spatialml_xr_add_native_activity requires SOURCES")
  endif()

  include("${SPATIALML_XR_UTILS_ROOT}/external/openxr/openxr.cmake")

  if(ANDROID AND NOT Vulkan_LIBRARY)
    find_library(Vulkan_LIBRARY NAMES vulkan)
  endif()
  if(READBACK_USE_OPENGL)
    if(NOT ANDROID)
      message(FATAL_ERROR "The OpenGL ES readback path only supports Android builds.")
    endif()
    set(XR_USE_GRAPHICS_API_OPENGL_ES TRUE)
    add_definitions(-DXR_USE_GRAPHICS_API_OPENGL_ES)
    message(STATUS "Enabling Android OpenGL ES support")
    add_library("${SPATIALML_XR_TARGET}_common_lib"
      "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/gfxwrapper_opengl.c"
    )
    target_include_directories("${SPATIALML_XR_TARGET}_common_lib" PUBLIC
      "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils"
    )
    target_link_libraries("${SPATIALML_XR_TARGET}_common_lib" PUBLIC EGL GLESv3)
  else()
    find_package(Vulkan)
    if(Vulkan_FOUND)
      set(XR_USE_GRAPHICS_API_VULKAN TRUE)
      add_definitions(-DXR_USE_GRAPHICS_API_VULKAN)
      message(STATUS "Enabling Vulkan support")
    elseif(BUILD_ALL_EXTENSIONS)
      message(FATAL_ERROR "Vulkan headers not found")
    endif()
  endif()

  set(SPATIALML_XR_BASE_SRCS
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/d3d_common.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_d3d11.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_d3d12.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_factory.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_opengl.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_opengles.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_vulkan.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/graphicsplugin_metal.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/logger.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/platformplugin_android.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/platformplugin_factory.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/platformplugin_posix.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/platformplugin_win32.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils/pch.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/main.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/openxr_program.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/pipeline.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/rendercommand.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/serialization.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/session.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/tensor.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/readback_async.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/readback.cpp"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils/utils.cpp"
  )

  set(SPATIALML_XR_VULKAN_SHADERS
    "${SPATIALML_XR_UTILS_ROOT}/base/vulkan_shaders/frag.glsl"
    "${SPATIALML_XR_UTILS_ROOT}/base/vulkan_shaders/vert.glsl"
  )

  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
  )
  FetchContent_GetProperties(nlohmann_json)
  if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
  endif()

  add_library(
    "${SPATIALML_XR_TARGET}" MODULE
    ${SPATIALML_XR_BASE_SRCS}
    ${SPATIALML_XR_SOURCES}
    ${SPATIALML_XR_VULKAN_SHADERS}
    "${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c"
  )

  target_link_libraries(
    "${SPATIALML_XR_TARGET}" PRIVATE
    android
    log
    OpenXR::openxr_loader
    ${Vulkan_LIBRARY}
    ${SPATIALML_XR_LIBRARIES}
  )
  if(READBACK_USE_OPENGL)
    target_link_libraries("${SPATIALML_XR_TARGET}" PRIVATE "${SPATIALML_XR_TARGET}_common_lib")
  endif()

  target_include_directories("${SPATIALML_XR_TARGET}" PRIVATE
    "${ANDROID_NDK}/sources/android/native_app_glue"
    "${SPATIALML_XR_UTILS_ROOT}/base"
    "${SPATIALML_XR_UTILS_ROOT}/base/vulkan_shaders"
    "${SPATIALML_XR_UTILS_ROOT}/base/oxr_utils"
    "${SPATIALML_XR_UTILS_ROOT}/base/image_utils"
    "${SPATIALML_XR_UTILS_ROOT}/base/securemr_utils"
    "${SPATIALML_XR_UTILS_ROOT}"
    "${CMAKE_SOURCE_DIR}"
    ${Vulkan_INCLUDE_DIRS}
    "${nlohmann_json_SOURCE_DIR}/single_include"
    ${SPATIALML_XR_INCLUDE_DIRS}
  )

  include("${SPATIALML_XR_UTILS_ROOT}/scripts/compile_glsl.cmake")
  compile_glsl("${SPATIALML_XR_TARGET}_glsl_compiles" ${SPATIALML_XR_VULKAN_SHADERS})
  if(GLSLANG_VALIDATOR AND NOT GLSLC_COMMAND)
    target_compile_definitions("${SPATIALML_XR_TARGET}" PRIVATE USE_GLSLANGVALIDATOR)
  endif()
  add_dependencies("${SPATIALML_XR_TARGET}" "${SPATIALML_XR_TARGET}_glsl_compiles")

  if(NOT READBACK_USE_GPU)
    target_compile_definitions("${SPATIALML_XR_TARGET}" PRIVATE
      DEFAULT_GRAPHICS_PLUGIN_VULKAN
      XR_USE_PLATFORM_ANDROID
      XR_READBACK_USE_CPU
      XR_USE_GRAPHICS_API_VULKAN
    )
  endif()
  if(READBACK_USE_VULKAN)
    target_compile_definitions("${SPATIALML_XR_TARGET}" PRIVATE
      DEFAULT_GRAPHICS_PLUGIN_VULKAN
      XR_USE_PLATFORM_ANDROID
      XR_USE_GRAPHICS_API_VULKAN
    )
  endif()
  if(READBACK_USE_OPENGL)
    target_compile_definitions("${SPATIALML_XR_TARGET}" PRIVATE
      DEFAULT_GRAPHICS_PLUGIN_OPENGLES
      XR_USE_PLATFORM_ANDROID
      XR_USE_GRAPHICS_API_OPENGL_ES
    )
  endif()
endfunction()
