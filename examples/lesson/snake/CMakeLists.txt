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
# the trap with its Group moved by one line elsewhere (-DLESSON_MOVE_GROUP): drawn and rules part ways
add_example(NAME "lesson_snake_trap_moved_capture" SOURCES trap/main.cpp)
target_compile_definitions(lesson_snake_trap_moved_capture PRIVATE LESSON_CAPTURE LESSON_MOVE_GROUP)
if (MSVC)
    target_compile_options(lesson_snake_trap_moved_capture PRIVATE "/FI${LESSON_CAPTURE_HPP}")
else ()
    target_compile_options(lesson_snake_trap_moved_capture PRIVATE "SHELL:-include ${LESSON_CAPTURE_HPP}")
endif ()
set_target_properties(lesson_snake_trap_moved_capture PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()

# The rules are a library of their own. It doesn't link threepp, so a threepp
# #include in the model fails to compile: the build keeps the design honest.
add_library(snake_model STATIC model/Snake.cpp model/Game.cpp)
target_include_directories(snake_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(snake_model PUBLIC cxx_std_20)

# the finished app: rules in snake_model, looks in view/, players in control/
set(SNAKE_APP_SOURCES main.cpp
        view/SnakeView.cpp view/AsciiView.cpp view/MinimapView.cpp
        control/KeyboardController.cpp control/GreedyAI.cpp)
add_example(NAME "lesson_snake" SOURCES ${SNAKE_APP_SOURCES})
# add_example makes no target under Emscripten (not a WEB example), hence the guard
if (TARGET lesson_snake)
    target_link_libraries(lesson_snake PRIVATE snake_model)
endif ()

# build lesson_snake as it stood after one step: cmake -DLESSON_SNAKE_STEP=3 (1..6; empty = the whole app)
set(LESSON_SNAKE_STEP "" CACHE STRING "lesson_snake as it stood after this step, 1..6 (empty: the whole app)")
if (TARGET lesson_snake AND NOT LESSON_SNAKE_STEP STREQUAL "")
    target_compile_definitions(lesson_snake PRIVATE STEP=${LESSON_SNAKE_STEP})
endif ()

if (NOT DEFINED EMSCRIPTEN)
foreach (step RANGE 1 6)
    add_example(NAME "lesson_snake_capture_${step}" SOURCES ${SNAKE_APP_SOURCES})
    target_link_libraries("lesson_snake_capture_${step}" PRIVATE snake_model)
    target_compile_definitions("lesson_snake_capture_${step}" PRIVATE STEP=${step} LESSON_CAPTURE)
    if (MSVC)
        target_compile_options("lesson_snake_capture_${step}" PRIVATE "/FI${LESSON_CAPTURE_HPP}")
    else ()
        target_compile_options("lesson_snake_capture_${step}" PRIVATE "SHELL:-include ${LESSON_CAPTURE_HPP}")
    endif ()
    set_target_properties("lesson_snake_capture_${step}" PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach ()
endif ()

# Catch2 is fetched after the examples are added; the name resolves when CMake generates.
# The tests don't link threepp either, and neither do the two classes they bring
# along: the text view and the AI build without it, so they can be tested without it.
if (THREEPP_BUILD_TESTS AND NOT DEFINED EMSCRIPTEN)
    add_executable(test_snake
            tests/test_game.cpp tests/test_ascii_view.cpp tests/test_greedy_ai.cpp
            view/AsciiView.cpp control/GreedyAI.cpp)
    target_link_libraries(test_snake PRIVATE snake_model Catch2::Catch2WithMain)
    add_test(NAME test_snake COMMAND test_snake)
endif ()
