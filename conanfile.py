from os import path
from conans import ConanFile, CMake, tools


class ThreeppConan(ConanFile):
    name = "threepp"
    author = "Lars Ivar Hatledal"
    license = "MIT"
    exports = "version.txt"
    scm = {
        "type": "git",
        "url": "auto",
        "revision": "auto"
    }
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"
    requires = (
        "glfw/3.3.8",
        "glad/0.1.36",
        "stb/cci.20210910"
    )

    options = {
        "with_bullet": [True, False],
        "with_assimp": [True, False],
        "with_imgui": [True, False],
    }

    default_options = (
        "with_bullet=False",
        "with_assimp=False",
        "glad:gl_version=4.1"
    )

    def set_version(self):
        self.version = tools.load(path.join(self.recipe_folder, "version.txt")).strip()

    def requirements(self):
        if self.options.with_bullet:
            self.requires("bullet3/3.24")
        if self.options.with_assimp:
            self.requires("assimp/5.2.2")
        if self.options.with_imgui:
            self.requires("imgui/cci.20220621+1.88.docking")

    def imports(self):
        self.copy("imgui_impl_glfw*", dst="_deps/imgui_glfw", src="res/bindings")
        self.copy("imgui_impl_opengl3*", dst="_deps/imgui_glfw", src="res/bindings")

    def configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["THREEPP_BUILD_EXAMPLES"] = "OFF"
        cmake.definitions["THREEPP_EXAMPLES_WITH_BULLET"] = "ON" if self.options.with_bullet else "OFF"
        cmake.configure()
        return cmake

    def build(self):
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):
        cmake = self.configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["threepp"]
