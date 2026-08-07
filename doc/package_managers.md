# Consuming threepp from a package manager

`threepp` is primarily a CMake package, and [CMake `FetchContent` is the recommended
way to consume it](../README.md#cmake-fetchcontent-recommended) — it is the path the
project itself tests, and it lets you point at any tag, branch or commit.

This page covers the two package-manager routes instead. Both go through
[Conan Center](https://conan.io/center/recipes/threepp), so both are subject to the
caveat below.

> **These lag the repository.** The Conan Center recipe lives in
> [conan-center-index](https://github.com/conan-io/conan-center-index), not here, and is
> published separately from this repository's tags — so the newest version on Conan Center
> is normally older than the newest tag, sometimes by several releases. Check
> [the recipe page](https://conan.io/center/recipes/threepp) for the current version rather
> than trusting the number in the snippets below, and use `FetchContent` if you need
> something recent.

## Conan

Example `conanfile.py`:

```python
from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout


class ExampleRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("threepp/0.0.20260310")

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
```

## Xmake

[xmake](https://xmake.io/) consumes the same Conan package through its `conan::` bridge.

Example `xmake.lua`:

```lua
add_rules("mode.debug", "mode.release")
add_requires("imgui", {configs = {glfw_opengl3 = true}}) -- optional dependency for UI widgets
add_requires("assimp") -- optional dependency for importing assembly models (.glb/.dae)
add_requires("conan::threepp/0.0.20260310", {
    alias = "threepp",
    configs = {
        settings = {"compiler.cppstd=20"}
    }
})
target("example")
set_kind("binary")
add_files("src/*.cpp")
add_packages("imgui", "threepp", "assimp")
set_languages("c++20")
```

## What you do not get this way

The Conan package builds the library. The optional halves that need an external SDK or a
build-time choice are not part of it — the Vulkan renderer, PhysX physics, the FBX and USD
loaders, the Python bindings, and the editor and player applications. Those need a CMake
build of the repository; see [How to build](../README.md#how-to-build) and the preset table
there.
