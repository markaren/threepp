# the Snake program built in "A threepp lesson, Part 0c: threepp is the view"

# the "before" pictures: everything in main(), then the inheritance trap
add_example(NAME "lesson_snake_tangle" SOURCES tangle/main.cpp)
add_example(NAME "lesson_snake_trap" SOURCES trap/main.cpp)

if (NOT DEFINED EMSCRIPTEN)
# The film's capture builds (not built by default): headless, a fixed 1/60 s clock and
# scripted keys. ../capture.hpp is force-included, so the sources stay as shown on screen.
set(LESSON_CAPTURE_HPP "${CMAKE_CURRENT_SOURCE_DIR}/../capture.hpp")
foreach (name tangle trap)
    add_example(NAME "lesson_snake_${name}_capture" SOURCES ${name}/main.cpp)
    target_compile_definitions("lesson_snake_${name}_capture" PRIVATE LESSON_CAPTURE)
    if (MSVC)
        target_compile_options("lesson_snake_${name}_capture" PRIVATE "/FI${LESSON_CAPTURE_HPP}")
    else ()
        target_compile_options("lesson_snake_${name}_capture" PRIVATE "SHELL:-include ${LESSON_CAPTURE_HPP}")
    endif ()
    set_target_properties("lesson_snake_${name}_capture" PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach ()
endif ()

# The rules are a library of their own. It doesn't link threepp, so a threepp
# #include in the model fails to compile: the build keeps the design honest.
add_library(snake_model STATIC model/Snake.cpp model/Game.cpp)
target_include_directories(snake_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(snake_model PUBLIC cxx_std_20)

# Catch2 is fetched after the examples are added; the name resolves when CMake generates.
if (THREEPP_BUILD_TESTS AND NOT DEFINED EMSCRIPTEN)
    add_executable(test_snake_model tests/test_game.cpp)
    target_link_libraries(test_snake_model PRIVATE snake_model Catch2::Catch2WithMain)
    add_test(NAME test_snake_model COMMAND test_snake_model)
endif ()
