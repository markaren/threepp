
add_example(NAME "vulkan_gltf_samples" LINK_IMGUI)
add_example(NAME "vulkan_showcase" LINK_IMGUI)
add_example(NAME "vulkan_lights" LINK_IMGUI)
add_example(NAME "vulkan_fog" LINK_IMGUI)
add_example(NAME "vulkan_ocean" LINK_IMGUI)
add_example(NAME "vulkan_denoise" LINK_IMGUI)
add_example(NAME "vulkan_animation" LINK_IMGUI)
add_example(NAME "vulkan_gallery" LINK_IMGUI)
add_example(NAME "vulkan_restir_test" LINK_IMGUI)
add_example(NAME "vulkan_lidar" LINK_IMGUI)
add_example(NAME "vulkan_event_camera" LINK_IMGUI)

if (THREEPP_WITH_FBX)
    add_example(NAME "vulkan_bistro" LINK_IMGUI)
endif ()

if (THREEPP_WITH_INFERENCE)
    include(${PROJECT_SOURCE_DIR}/cmake/CompileVulkanShaders.cmake)
    find_package(Vulkan REQUIRED)

    add_example(NAME "vulkan_yolov8_inference"
            SOURCES vulkan_yolov8_inference.cpp
            yolov8/YoloV8nVk.cpp
            yolov8/VkInfer.cpp
            yolov8/WeightLoader.cpp)

    if (TARGET vulkan_yolov8_inference)
        target_include_directories(vulkan_yolov8_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(vulkan_yolov8_inference PRIVATE Vulkan::Vulkan)

        set(_yolo_sh "${CMAKE_CURRENT_SOURCE_DIR}/yolov8/shaders")
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv2d.comp"       kConv2dSpv      _yh0)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv1x1.comp"      kConv1x1Spv     _yh1)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv3x3s1.comp"    kConv3x3s1Spv   _yh2)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv3x3s2.comp"    kConv3x3s2Spv   _yh3)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/preprocess.comp"   kPreprocessSpv  _yh4)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/bn_silu.comp"      kBnSiluSpv      _yh5)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/add.comp"          kAddSpv         _yh6)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/maxpool.comp"      kMaxpoolSpv     _yh7)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/upsample.comp"     kUpsampleSpv    _yh8)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/concat.comp"       kConcatSpv      _yh9)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/slice.comp"        kSliceSpv       _yh10)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/detect.comp"       kDetectSpv      _yh11)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/detect_fused.comp" kDetectFusedSpv _yh12)
    endif ()
endif ()