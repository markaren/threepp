# Embed one SPIR-V module into a C++ header as a `const uint32_t NAME[]` array,
# the shape glslangValidator's --vn used to write.
#
# Script mode, invoked per shader by compile_vulkan_shader's custom command:
#
#   cmake -DIN=<module.spv> -DOUT=<header.spv.h> -DVAR=<identifier> -P EmbedSpirv.cmake
#
# SPIR-V is a stream of little-endian 32-bit words, so every 4 bytes of the
# file become one word, low byte first. About 4 s for the 5 MB deferred
# shade, milliseconds for everything else.

if (NOT IN OR NOT OUT OR NOT VAR)
    message(FATAL_ERROR "EmbedSpirv.cmake: IN, OUT and VAR are all required")
endif ()

file(READ "${IN}" _hex HEX)
string(LENGTH "${_hex}" _len)
math(EXPR _rem "${_len} % 8")
if (NOT _rem EQUAL 0)
    message(FATAL_ERROR "EmbedSpirv.cmake: ${IN} is not a whole number of 32-bit words")
endif ()
math(EXPR _words "${_len} / 8")

# bytes b0 b1 b2 b3 -> 0xb3b2b1b0
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1," _body "${_hex}")
# eight words per line
string(REGEX REPLACE "((0x[0-9a-f]+,){8})" "\\1\n\t" _body "${_body}")

file(WRITE "${OUT}"
        "\t// ${_words} words\n"
        "\t#pragma once\n"
        "const uint32_t ${VAR}[] = {\n"
        "\t${_body}\n"
        "};\n")
