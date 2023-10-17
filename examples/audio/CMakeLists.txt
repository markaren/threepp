
add_executable(audio audio.cpp)
target_link_libraries(audio PRIVATE threepp imgui::imgui)
target_include_directories(audio PRIVATE "${CMAKE_SOURCE_DIR}/examples/libs" "${MINIAUDIO_INCLUDE_DIRS}")
