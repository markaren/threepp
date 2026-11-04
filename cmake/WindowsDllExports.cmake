# The export list of a shared threepp on MSVC: everything CMake's
# WINDOWS_EXPORT_ALL_SYMBOLS would export, minus what is defined inside
# namespace std.
#
# Why not WINDOWS_EXPORT_ALL_SYMBOLS as it is: it exports every external symbol
# in the library's object files, and most of those are the COMDAT instantiations
# of std templates (std::vector<Vector3>::push_back and its kin) that every
# translation unit instantiates for itself. On 2026-11-04 they were 67 000 of
# the 77 000 exports, and an import library holds at most 65 535 members
# (LNK1189). No consumer resolves a std template through threepp.lib: its own
# object files carry the same COMDATs, and the linker takes a definition it
# already has before it opens any import library. What threepp itself defines
# is under 10 000 symbols.
#
# How: the target keeps WINDOWS_EXPORT_ALL_SYMBOLS off. A PRE_LINK step runs the
# `cmake -E __create_def` the generator would have put into the link rule (the
# same internal command, over the same object files, so the same symbol
# selection), then the small host tool built from WindowsDllExports.cpp rewrites
# that .def without the std:: entries, and the link gets the result as /DEF:.
# PRE_LINK is the one slot after the objects are compiled and before they are
# linked; a custom command with the .def as its OUTPUT cannot take it, because
# Ninja gives every object of a target an order-only edge to the target's custom
# outputs, and the .def needs the objects.
#
# The rest of the export story is unchanged: static data members still need the
# explicit THREEPP_API (include/threepp/export.hpp), as before.

set(_THREEPP_DLL_EXPORTS_TOOL_SOURCE "${CMAKE_CURRENT_LIST_DIR}/WindowsDllExports.cpp")

function(threepp_windows_dll_exports target)
    if (NOT MSVC)
        return()
    endif ()
    get_target_property(_type ${target} TYPE)
    if (NOT _type STREQUAL "SHARED_LIBRARY")
        return()
    endif ()

    if (NOT TARGET threepp_dll_exports)
        add_executable(threepp_dll_exports "${_THREEPP_DLL_EXPORTS_TOOL_SOURCE}")
        target_compile_features(threepp_dll_exports PRIVATE cxx_std_20)
        target_link_libraries(threepp_dll_exports PRIVATE dbghelp)
        set_target_properties(threepp_dll_exports PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dll-exports"
                FOLDER "tools")
    endif ()

    set(_dir "${CMAKE_CURRENT_BINARY_DIR}/dll-exports/${target}/$<CONFIG>")
    # One object path per line, the input format of `cmake -E __create_def`.
    file(GENERATE OUTPUT "${_dir}/objects.txt"
            CONTENT "$<JOIN:$<TARGET_OBJECTS:${target}>,\n>\n")

    set_property(TARGET ${target} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    add_custom_command(TARGET ${target} PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E __create_def "${_dir}/all.def" "${_dir}/objects.txt"
            COMMAND threepp_dll_exports "${_dir}/all.def" "${_dir}/exports.def"
            VERBATIM)
    add_dependencies(${target} threepp_dll_exports)
    target_link_options(${target} PRIVATE "/DEF:${_dir}/exports.def")
endfunction()
