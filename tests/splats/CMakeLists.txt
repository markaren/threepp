
add_test_executable(SplatData_test)

# None of these needs a window: SplatCloud is fully built and sorted on the
# CPU, and SplatSH_test only reads the shader source out of it. They link
# without a GLFW frontend because SplatCloud reaches the viewport through the
# virtual Renderer interface rather than a concrete GLRenderer cast.
add_test_executable(SplatCloudSort_test)
add_test_executable(SplatCloudRaycast_test)
add_test_executable(SplatSH_test)
# The direct point-cloud collider is CPU code: no window, no GPU.
add_test_executable(PointSurface_test)
