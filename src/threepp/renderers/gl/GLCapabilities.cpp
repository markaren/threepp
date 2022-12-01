
#include "threepp/renderers/gl/GLCapabilities.hpp"

#include "threepp/renderers/gl/glHelper.hpp"

using namespace threepp::gl;

GLCapabilities::GLCapabilities()
    : maxAnisotropy(0),

      maxTextures(glGetParameter(GL_MAX_TEXTURE_IMAGE_UNITS)),
      maxVertexTextures(glGetParameter(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)),
      maxTextureSize(glGetParameter(GL_MAX_TEXTURE_SIZE)),
      maxCubemapSize(glGetParameter(GL_MAX_CUBE_MAP_TEXTURE_SIZE)),

      maxAttributes(glGetParameter(GL_MAX_VERTEX_ATTRIBS)),
      maxVertexUniforms(glGetParameter(GL_MAX_VERTEX_UNIFORM_VECTORS)),
      maxVaryings(glGetParameter(GL_MAX_VARYING_VECTORS)),
      maxFragmentUniforms(glGetParameter(GL_MAX_FRAGMENT_UNIFORM_VECTORS)),

      vertexTextures(maxVertexTextures > 0),

      maxSamples(glGetParameter(GL_MAX_SAMPLES)) {}
