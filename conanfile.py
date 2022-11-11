from conans import ConanFile, CMake, tools
from os import path


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
    generators = "cmake"
    requires = (
        "glfw/3.3.4",
        "glad/0.1.34",
        "stb/20200203"
    )

    default_options = (
        "glad:gl_version=4.1"
    )

    def set_version(self):
        self.version = tools.load(path.join(self.recipe_folder, "version.txt")).strip()

    def configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["THREEPP_BUILD_EXAMPLES"] = "OFF"
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
