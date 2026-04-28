add_cross_renderer_test(rtdetr_test rtdetr_test.cpp)
target_sources(rtdetr_test PRIVATE
        "${PROJECT_SOURCE_DIR}/examples/wgpu/rtdetr/RtDetr.cpp"
        "${PROJECT_SOURCE_DIR}/examples/wgpu/rtdetr/WeightLoader.cpp"
)
target_include_directories(rtdetr_test PRIVATE
        "${PROJECT_SOURCE_DIR}/examples/wgpu"
)
