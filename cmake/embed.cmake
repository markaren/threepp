
set(generatedSourcesDir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generatedSourcesDir}")

############################################
# shaders
############################################

set(THREEPP_SHADER_INCLUDES)
set(THREEPP_SHADERCHUNK_CODE)
set(THREEPP_SHADERLIB_CODE)

file(GLOB files "${PROJECT_SOURCE_DIR}/data/shaders/ShaderChunk/*.glsl")
foreach (shaderFile ${files})

    get_filename_component(fileName ${shaderFile} NAME_WLE)
    set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk/${fileName}.hpp")

    file(READ ${shaderFile} text)
    set(text "\
namespace threepp::shaders::shaderchunk {\n\n\
const char* ${fileName}=R\"(${text})\";\n\n\
}\n")

    file(WRITE "${header_file}"
            "#ifndef THREEPP_${fileName}_HPP\n"
            "#define THREEPP_${fileName}_HPP\n\n")
    file(APPEND "${header_file}" "${text}")
    file(APPEND "${header_file}" "\n\n#endif\n")

    set(THREEPP_SHADER_INCLUDES "${THREEPP_SHADER_INCLUDES}\n#include \"ShaderChunk/${fileName}.hpp\"")
    set(THREEPP_SHADERCHUNK_CODE "${THREEPP_SHADERCHUNK_CODE}\tdata_[\"${fileName}\"] = shaderchunk::${fileName};\n")

endforeach ()

file(GLOB files "${PROJECT_SOURCE_DIR}/data/shaders/ShaderLib/*.glsl")
foreach (shaderFile ${files})

    get_filename_component(fileName ${shaderFile} NAME_WLE)
    set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/ShaderLib/${fileName}.hpp")

    file(READ ${shaderFile} text)
    set(text "\
namespace threepp::shaders::shaderlib {\n\n\
const char* ${fileName}=R\"(${text})\";\n\n\
}\n")

    file(WRITE "${header_file}"
            "#ifndef THREEPP_${fileName}_HPP\n"
            "#define THREEPP_${fileName}_HPP\n\n")
    file(APPEND "${header_file}" "${text}")
    file(APPEND "${header_file}" "\n\n#endif\n")

    set(THREEPP_SHADER_INCLUDES "${THREEPP_SHADER_INCLUDES}\n#include \"ShaderLib/${fileName}.hpp\"")
    set(THREEPP_SHADERLIB_CODE "${THREEPP_SHADERLIB_CODE}\tdata_[\"${fileName}\"] = shaderlib::${fileName};\n")

endforeach ()

configure_file(
        "threepp/renderers/shaders/ShaderChunk.cpp.in"
        "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp"
        @ONLY
)

# ==============================================================================
# favicon
# ==============================================================================

set(favicon_in "${PROJECT_SOURCE_DIR}/data/favicon.bmp")
set(favicon_out "${generatedSourcesDir}/threepp/favicon.hpp")

#https://jonathanhamberg.com/post/cmake-file-embedding/
file(READ "${favicon_in}" content HEX)
string(REGEX MATCHALL "([A-Fa-f0-9][A-Fa-f0-9])" SEPARATED_HEX ${content})

set(counter 0)
foreach (hex IN LISTS SEPARATED_HEX)
    string(APPEND hex_data "0x${hex},")
    math(EXPR counter "${counter}+1")
    if (counter GREATER 16)
        # Write a newline so that all of the array initializer
        # gets spread across multiple lines.
        string(APPEND hex_data "\n    ")
        set(counter 0)
    endif ()
endforeach ()

set(header_content "\

#include <vector>\n\

namespace threepp {\n\n\

std::vector<unsigned char> faviconSource() {\n\
    return std::vector<unsigned char>{${hex_data}};\n\
}\n\n\

}\n")

file(WRITE "${favicon_out}"
        "#ifndef THREEPP_FAVICON_HPP\n"
        "#define THREEPP_FAVICON_HPP\n\n")
file(APPEND "${favicon_out}" "${header_content}")
file(APPEND "${favicon_out}" "\n\n#endif\n")
