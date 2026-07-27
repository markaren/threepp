// Single translation unit owning the stb_image_write implementation.
//
// It used to live in GLRenderer.cpp, which meant anything calling stbi_write_*
// pulled that object file out of the archive - and with it Canvas, which is
// only compiled when THREEPP_WITH_GLFW is on. ObjectExporter encodes embedded
// PNG textures, so a serialization-only test failed to link in a no-GLFW build
// against a renderer it never used.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
