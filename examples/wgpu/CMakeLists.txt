
add_example(NAME "wgpu_triangle")
add_example(NAME "wgpu_postprocess")

if (THREEPP_WITH_INFERENCE)
    add_example(NAME "yolov8_inference"
            SOURCES yolov8_inference.cpp
            yolov8/YoloV8n.cpp
            yolov8/WeightLoader.cpp)
    target_include_directories(yolov8_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${WGPU_INCLUDE_DIR})

    add_example(NAME "rtdetr_inference"
            SOURCES rtdetr_inference.cpp
            rtdetr/RtDetr.cpp
            rtdetr/WeightLoader.cpp)
    target_include_directories(rtdetr_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${WGPU_INCLUDE_DIR})
endif ()
