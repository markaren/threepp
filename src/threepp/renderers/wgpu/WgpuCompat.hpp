#pragma once

// WgpuCompat.hpp — Compatibility layer between wgpu-native's C API and
// Emscripten's WebGPU C API.

#include <webgpu/webgpu.h>

// ---------------------------------------------------------------------------
// Label helpers
// wgpu-native: labels are WGPUStringView {.data, .length}
// Emscripten:  labels are const char*
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
#  define WGPU_LABEL(s) (s)
#  define WGPU_STR(ptr, len) (ptr)
#  define WGPU_ENTRY(s) (s)
#else
#  define WGPU_LABEL(s) WGPUStringView{.data = (s), .length = sizeof(s) - 1}
#  define WGPU_STR(ptr, len) WGPUStringView{.data = (ptr), .length = static_cast<size_t>(len)}
#  define WGPU_ENTRY(s) WGPUStringView{.data = (s), .length = sizeof(s) - 1}
#endif

// ---------------------------------------------------------------------------
// Shader source structs
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
using WgpuShaderSourceWGSL = WGPUShaderModuleWGSLDescriptor;
#  define WGPU_STYPE_SHADER_SOURCE_WGSL WGPUSType_ShaderModuleWGSLDescriptor
#  define WGPU_SHADER_CODE(src, ptr, len) (src).code = (ptr)
#else
using WgpuShaderSourceWGSL = WGPUShaderSourceWGSL;
#  define WGPU_STYPE_SHADER_SOURCE_WGSL WGPUSType_ShaderSourceWGSL
#  define WGPU_SHADER_CODE(src, ptr, len) (src).code = {.data = (ptr), .length = static_cast<size_t>(len)}
#endif

// ---------------------------------------------------------------------------
// Optional bool (depth write)
// wgpu-native: WGPUOptionalBool enum
// Emscripten:  WGPUBool (uint32_t), true=1 false=0
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
#  define WGPU_OPTIONAL_BOOL_TRUE  1
#  define WGPU_OPTIONAL_BOOL_FALSE 0
#else
#  define WGPU_OPTIONAL_BOOL_TRUE  WGPUOptionalBool_True
#  define WGPU_OPTIONAL_BOOL_FALSE WGPUOptionalBool_False
#endif

// ---------------------------------------------------------------------------
// Texture/buffer copy info structs
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
using WgpuTexelCopyTextureInfo  = WGPUImageCopyTexture;
using WgpuTexelCopyBufferInfo   = WGPUImageCopyBuffer;
using WgpuTexelCopyBufferLayout = WGPUTextureDataLayout;
#else
using WgpuTexelCopyTextureInfo  = WGPUTexelCopyTextureInfo;
using WgpuTexelCopyBufferInfo   = WGPUTexelCopyBufferInfo;
using WgpuTexelCopyBufferLayout = WGPUTexelCopyBufferLayout;
#endif

// ---------------------------------------------------------------------------
// Buffer map async status
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
using WgpuMapAsyncStatus = WGPUBufferMapAsyncStatus;
#  define WGPU_MAP_ASYNC_STATUS_SUCCESS WGPUBufferMapAsyncStatus_Success
#else
using WgpuMapAsyncStatus = WGPUMapAsyncStatus;
#  define WGPU_MAP_ASYNC_STATUS_SUCCESS WGPUMapAsyncStatus_Success
#endif

// ---------------------------------------------------------------------------
// Buffer usage flags type
// wgpu-native: WGPUBufferUsage enum with operator|
// Emscripten:  WGPUBufferUsageFlags is uint32_t
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__
using WgpuBufferUsageFlags = WGPUBufferUsageFlags;
#else
using WgpuBufferUsageFlags = WGPUBufferUsage;
#endif