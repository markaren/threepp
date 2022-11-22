from os import path
from conans import ConanFile, CMake, tools
from conan.tools.cmake import CMakeDeps

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
        "with_bullet": [True, False]
    }

    default_options = (
        "with_bullet=False",
        "glad:gl_version=4.1"
    )

    def set_version(self):
        self.version = tools.load(path.join(self.recipe_folder, "version.txt")).strip()

    def requirements(self):
        if self.options.with_bullet:
            self.requires("bullet3/3.24")

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
