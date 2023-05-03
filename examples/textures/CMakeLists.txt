
add_executable(texture2d texture2d.cpp)
target_link_libraries(texture2d PUBLIC threepp)
if (CURL_FOUND)
    target_link_libraries(texture2d PUBLIC CURL::libcurl)
endif ()

add_executable(texture3d texture3d.cpp)
target_link_libraries(texture3d PUBLIC threepp)

add_executable(data_texture data_texture.cpp)
target_link_libraries(data_texture PUBLIC threepp)
