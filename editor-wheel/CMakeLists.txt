# The threepp-editor wheel: builds the editor out of the parent repository and
# installs it — executable, runtime DLLs, licenses and the union type stubs —
# into the `threepp_editor` Python package under one component, mirroring how
# the main wheel's python_module component works.
#
# This project only makes sense INSIDE the repository (see pyproject.toml: the
# distribution is wheel-only for exactly that reason).
cmake_minimum_required(VERSION 3.21)
project(threepp_editor_wheel LANGUAGES C CXX)

# The editor needs the library, GLFW and its own subdirectory; examples/tests
# stay off. Vulkan stays OFF for the wheel build — the exe would gain a
# load-time vulkan-1.dll import (the same brick-on-import problem the module
# wheel solved with /DELAYLOAD), and the editor's GL viewport is the product
# here. Audio stays on: the editor authors positional sound.
set(THREEPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(THREEPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(THREEPP_BUILD_EDITOR ON CACHE BOOL "" FORCE)
set(THREEPP_WITH_PYTHON OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." threepp)

install(TARGETS threepp_editor
        RUNTIME DESTINATION threepp_editor/bin COMPONENT editor_package)

# The same explicit-name discipline as the module wheel (python/CMakeLists.txt):
# the four CPU PhysX DLLs are the executable's load-time imports; the GPU pair
# (238 MB) is dlopen'd lazily and deliberately excluded — drop the two DLLs
# into threepp_editor/bin to enable GPU dynamics / soft-body play.
if (WIN32 AND unofficial-omniverse-physx-sdk_DIR)
    get_filename_component(_physx_bin "${unofficial-omniverse-physx-sdk_DIR}/../../bin" ABSOLUTE)
    foreach (_dll PhysX_64.dll PhysXCommon_64.dll PhysXCooking_64.dll PhysXFoundation_64.dll)
        install(FILES "${_physx_bin}/${_dll}"
                DESTINATION threepp_editor/bin COMPONENT editor_package)
    endforeach ()
    get_filename_component(_physx_share "${unofficial-omniverse-physx-sdk_DIR}/.." ABSOLUTE)
    if (EXISTS "${_physx_share}/physx/copyright")
        install(FILES "${_physx_share}/physx/copyright"
                DESTINATION threepp_editor COMPONENT editor_package RENAME "PHYSX_LICENSE.txt")
    endif ()
endif ()

# The UNION stubs — including the hand-maintained editor.pyi with the physics
# handles only the editor's embedded interpreter serves. This is the one
# distribution where the union is the TRUTH, not an over-promise: these stubs
# describe exactly what an editor script can call. The launcher points
# THREEPP_PYTHON_STUBS here so "Edit in VS Code" resolves them via Pylance.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../python/threepp/threepp"
        DESTINATION threepp_editor/stubs COMPONENT editor_package
        FILES_MATCHING PATTERN "*.pyi")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY.md"
        DESTINATION threepp_editor COMPONENT editor_package)
