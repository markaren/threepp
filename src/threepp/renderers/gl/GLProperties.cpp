
#include "GLProperties.hpp"

#include "GLUniforms.hpp"
#include "GLProgram.hpp"

using namespace threepp::gl;


void GLProperties::dispose() {

    textureProperties.dispose();
    materialProperties.dispose();
}
