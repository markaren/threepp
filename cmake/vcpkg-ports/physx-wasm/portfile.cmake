# PhysX 5.5 for wasm32-emscripten, built from the same source tag as the vcpkg
# 'physx' port. PhysX's headers already know Emscripten (PX_EMSCRIPTEN, treated
# as 32-bit x86 Linux with PX_SUPPORT_GPU_PHYSX 0), so the build reuses the
# linux platform files. emscripten.patch holds what that path still needs:
#   - compiler flags: no -Weverything -Werror (emsdk's clang is newer than the
#     warning list), no -ffp-exception-behavior=maytrap (unsupported on wasm),
#     and -msimd128 -msse2, because the broadphase uses SSE intrinsics with no
#     scalar fallback; Emscripten compiles them to wasm SIMD.
#   - PxMath.h: no glibc .symver directives.
#   - PxHash.h, FdTempAllocator.cpp: size_t is unsigned long on wasm32, which
#     made two overload sets ambiguous.
#   - FdUnixThread.cpp: no x86 'pause' instruction; FdUnixMutex.cpp: Emscripten's
#     libc has no pthread_mutexattr_setprioceiling.
#   - NpPhysics.cpp, DyIslandManager.h: the PX_SUPPORT_GPU_PHYSX=0 paths did not
#     compile (a stale stub signature, a duplicate stub block, forward
#     declarations hidden behind the GPU guard).
# Static, release only, no threads: create the dispatcher with
# PxDefaultCpuDispatcherCreate(0).

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO NVIDIA-Omniverse/PhysX
    REF 106.4-physx-5.5.0
    SHA512 93ad438db81e9dc095741c837c0e797b56b35d6b77c7d1b1367b11bcbcb4ee1b8ff2affc27624d06829ac5e979f08d506fe727851fc383724e6633b775752d82
    HEAD_REF main
    PATCHES
        emscripten.patch
)

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
set(VCPKG_BUILD_TYPE release)

set(PHYSX_ROOT_DIR "${SOURCE_PATH}/physx")

vcpkg_cmake_configure(
    SOURCE_PATH "${PHYSX_ROOT_DIR}/compiler/public"
    OPTIONS
        -DCMAKE_TOOLCHAIN_FILE=${VCPKG_CHAINLOAD_TOOLCHAIN_FILE}
        -DTARGET_BUILD_PLATFORM=linux
        -DPHYSX_ROOT_DIR=${PHYSX_ROOT_DIR}
        -DPX_OUTPUT_LIB_DIR=${PHYSX_ROOT_DIR}
        -DPX_OUTPUT_BIN_DIR=${PHYSX_ROOT_DIR}
        -DCMAKE_INSTALL_PREFIX=${PHYSX_ROOT_DIR}/install/emscripten/PhysX
        -DPX_BUILDSNIPPETS=OFF
        -DPX_BUILDPVDRUNTIME=OFF
        -DPX_GENERATE_STATIC_LIBRARIES=ON
        # Skips the linux files' copy of the prebuilt CUDA library, which does
        # not exist for this target.
        -DGPU_LIB_COPIED=1
)

vcpkg_cmake_build()

# PhysX writes its archives to physx/bin/<platform>/release rather than the
# build tree's install prefix; collect them from there.
file(GLOB_RECURSE PHYSX_LIBS LIST_DIRECTORIES false "${PHYSX_ROOT_DIR}/bin/*/release/*.a")
if (NOT PHYSX_LIBS)
    message(FATAL_ERROR "physx-wasm: no PhysX archives under ${PHYSX_ROOT_DIR}/bin")
endif ()
file(INSTALL ${PHYSX_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")

file(INSTALL "${PHYSX_ROOT_DIR}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include/physx")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/unofficial-omniverse-physx-sdk-config.cmake"
        DESTINATION "${CURRENT_PACKAGES_DIR}/share/unofficial-omniverse-physx-sdk")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
