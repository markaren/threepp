
set(generatedSourcesDir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generatedSourcesDir}")

############################################
# shaders
############################################

set(THREEPP_SHADER_INCLUDES)
set(THREEPP_SHADERCHUNK_CODE)
set(THREEPP_SHADERLIB_CODE)

# Filled with the generated headers; src/CMakeLists.txt hands them to the
# threepp target so the custom commands below are part of the build graph.
set(THREEPP_GENERATED_SHADER_HEADERS)

# embed_glsl_dir(<source subdir> <namespace> <code accumulator var>)
#
# One build-time custom command per shader, mirroring how the Vulkan backend
# treats its .comp/.glsl (see cmake/CompileVulkanShaders.cmake). Editing a
# shader then re-embeds exactly that one header and recompiles ShaderChunk.cpp.
#
# This used to be a plain file(READ)/file(WRITE) here at configure time, which
# meant nothing in the build graph depended on the .glsl files: `cmake --build`
# after a shader edit rebuilt nothing and the binary silently kept running the
# previously embedded text. It cost a debugging session — a probe edit produced
# output bit-identical to the unedited shader, which looked like a finding.
#
# The registration list below (the #includes and data_[] lines baked into
# ShaderChunk.cpp) stays a configure-time product, which is right: it changes
# only when a shader is ADDED or REMOVED, and CONFIGURE_DEPENDS on the globs
# makes CMake re-configure in exactly that case.
function(embed_glsl_dir subdir ns codeVar)

    file(GLOB files CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/shaders/${subdir}/*.glsl")

    set(_includes "${THREEPP_SHADER_INCLUDES}")
    set(_code "${${codeVar}}")
    set(_headers "${THREEPP_GENERATED_SHADER_HEADERS}")

    foreach (shaderFile ${files})

        get_filename_component(fileName ${shaderFile} NAME_WLE)
        set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/${subdir}/${fileName}.hpp")

        add_custom_command(
                OUTPUT "${header_file}"
                COMMAND "${CMAKE_COMMAND}"
                        "-DIN=${shaderFile}"
                        "-DOUT=${header_file}"
                        "-DNS=${ns}"
                        "-DNAME=${fileName}"
                        -P "${PROJECT_SOURCE_DIR}/cmake/EmbedGlsl.cmake"
                DEPENDS "${shaderFile}" "${PROJECT_SOURCE_DIR}/cmake/EmbedGlsl.cmake"
                COMMENT "Embedding GL shader ${subdir}/${fileName}.glsl"
                VERBATIM)

        list(APPEND _headers "${header_file}")
        set(_includes "${_includes}\n#include \"${subdir}/${fileName}.hpp\"")
        set(_code "${_code}\tdata_[\"${fileName}\"] = ${ns}::${fileName};\n")

    endforeach ()

    set(THREEPP_SHADER_INCLUDES "${_includes}" PARENT_SCOPE)
    set(${codeVar} "${_code}" PARENT_SCOPE)
    set(THREEPP_GENERATED_SHADER_HEADERS "${_headers}" PARENT_SCOPE)

endfunction()

embed_glsl_dir(ShaderChunk shaderchunk THREEPP_SHADERCHUNK_CODE)
embed_glsl_dir(ShaderLib shaderlib THREEPP_SHADERLIB_CODE)

configure_file(
        "threepp/renderers/shaders/ShaderChunk.cpp.in"
        "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp"
        @ONLY
)

# ==============================================================================
# favicon
# ==============================================================================

set(favicon_in "${PROJECT_SOURCE_DIR}/src/resources/favicon.bmp")
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


############################################
# fonts
############################################

set(embeddedFonts_out "${generatedSourcesDir}/threepp/EmbeddedFonts.cpp")

set(fontName "helvetiker_bold")
set(fontFile "${PROJECT_SOURCE_DIR}/src/resources/${fontName}.typeface.json")
file(READ ${fontFile} FILE_CONTENTS HEX)
string(REGEX REPLACE "(..)" "0x\\0," FILE_CONTENTS "${FILE_CONTENTS}")
get_filename_component(fontName ${fontFile} NAME)

file(WRITE ${embeddedFonts_out}
        "#include \"threepp/loaders/FontLoader.hpp\"\n\n"

        "#include <vector>\n\n"

        "using namespace threepp;\n\n"

        "// ${fontName}\n"
        "const std::vector<unsigned char> data{${FILE_CONTENTS}};\n\n"

        "Font FontLoader::defaultFont() {\n"
        "\treturn *load(data);\n"
        "}\n"
)
