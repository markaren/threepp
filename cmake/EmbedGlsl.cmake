# Embed one GLSL file into a C++ header as a raw string literal.
#
# Script mode — invoked per shader by a build-time custom command, not included:
#
#   cmake -DIN=<file.glsl> -DOUT=<file.hpp> -DNS=<namespace> -DNAME=<identifier>
#         -P EmbedGlsl.cmake
#
# Doing this from a custom command rather than inline at configure time is the
# whole point: a configure-time file(READ) has nothing in the build graph
# depending on it, so editing a shader and running `cmake --build` silently
# keeps the previously embedded text. That fails quietly — the build succeeds
# and the binary runs the old shader — which is the worst way for it to fail.

if (NOT IN OR NOT OUT OR NOT NS OR NOT NAME)
    message(FATAL_ERROR "EmbedGlsl.cmake: IN, OUT, NS and NAME are all required")
endif ()

file(READ "${IN}" text)

file(WRITE "${OUT}"
        "#ifndef THREEPP_${NAME}_HPP\n"
        "#define THREEPP_${NAME}_HPP\n\n"
        "namespace threepp::shaders::${NS} {\n\n"
        "const char* ${NAME}=R\"(${text})\";\n\n"
        "}\n"
        "\n\n#endif\n")
