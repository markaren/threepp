
add_executable(geometries geometries.cpp)
target_link_libraries(geometries PUBLIC threepp glfw::glfw)

add_executable(lathe_geometry lathe_geometry.cpp)
target_link_libraries(lathe_geometry PUBLIC threepp glfw::glfw)

add_executable(tube_geometry tube_geometry.cpp)
target_link_libraries(tube_geometry PUBLIC threepp glfw::glfw)

add_executable(shape_geometry shape_geometry.cpp)
target_link_libraries(shape_geometry PUBLIC threepp glfw::glfw)

add_executable(convex_geometry convex_geometry.cpp)
target_link_libraries(convex_geometry PUBLIC threepp glfw::glfw)

if (imgui_FOUND)
   add_executable(box_geometry box_geometry.cpp)
   target_link_libraries(box_geometry PUBLIC threepp imgui::imgui glfw::glfw)

   add_executable(cylinder_geometry cylinder_geometry.cpp)
   target_link_libraries(cylinder_geometry PUBLIC threepp imgui::imgui glfw::glfw)

   add_executable(plane_geometry plane_geometry.cpp)
   target_link_libraries(plane_geometry PUBLIC threepp imgui::imgui glfw::glfw)

   add_executable(sphere_geometry sphere_geometry.cpp)
   target_link_libraries(sphere_geometry PUBLIC threepp imgui::imgui glfw::glfw)
endif()
