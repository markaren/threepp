from conans import ConanFile


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
    )

    options = {
        "with_bullet": [True, False],
        "with_assimp": [True, False],
        "with_imgui": [True, False]
    }

    default_options = (
        "with_bullet=False",
        "with_assimp=False",
        "with_imgui=False",
        "glad:gl_version=4.1"
    )

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
