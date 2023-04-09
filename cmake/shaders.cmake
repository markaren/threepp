
set(generatedSourcesDir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generatedSourcesDir}")

set(THREEPP_SHADER_INCLUDES)
set(THREEPP_SHADERCHUNK_CODE)
set(THREEPP_SHADERLIB_CODE)

file(GLOB files "${PROJECT_SOURCE_DIR}/data/shaders/ShaderChunk/*.glsl")
foreach(shaderFile ${files})
    get_filename_component(fileName ${shaderFile} NAME_WLE)
    set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk/${fileName}.hpp")
    file(READ ${shaderFile} text)
    set(text "\
namespace threepp::shaders::shaderchunk {\n\n\
const char* ${fileName}=R\"(${text})\";\n\n\
}\n"
            )
    file(WRITE "${header_file}"
            "#ifndef THREEPP_${fileName}_HPP\n"
            "#define THREEPP_${fileName}_HPP\n\n"
            )
    file(APPEND "${header_file}" "${text}")
    file(APPEND "${header_file}" "\n\n#endif")

    set(THREEPP_SHADER_INCLUDES "${THREEPP_SHADER_INCLUDES}\n#include \"ShaderChunk/${fileName}.hpp\"")

    set(THREEPP_SHADERCHUNK_CODE
            "${THREEPP_SHADERCHUNK_CODE}\tdata_[\"${fileName}\"] = shaderchunk::${fileName};\n"
            )

endforeach()

file(GLOB files "${PROJECT_SOURCE_DIR}/data/shaders/ShaderLib/*.glsl")
foreach(shaderFile ${files})
    get_filename_component(fileName ${shaderFile} NAME_WLE)
    set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/ShaderLib/${fileName}.hpp")
    file(READ ${shaderFile} text)
    set(text "\
namespace threepp::shaders::shaderlib {\n\n\
const char* ${fileName}=R\"(${text})\";\n\n\
}\n"
            )
    file(WRITE "${header_file}"
            "#ifndef THREEPP_${fileName}_HPP\n"
            "#define THREEPP_${fileName}_HPP\n\n"
            )
    file(APPEND "${header_file}" "${text}")
    file(APPEND "${header_file}" "\n\n#endif")

    set(THREEPP_SHADER_INCLUDES "${THREEPP_SHADER_INCLUDES}\n#include \"ShaderLib/${fileName}.hpp\"")

    set(THREEPP_SHADERLIB_CODE
            "${THREEPP_SHADERLIB_CODE}\tdata_[\"${fileName}\"] = shaderlib::${fileName};\n"
            )

endforeach()

configure_file(
        "threepp/renderers/shaders/ShaderChunk.cpp.in"
        "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp"
        @ONLY
)
