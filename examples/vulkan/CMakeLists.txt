
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

    add_example(NAME "vulkan_rtdetr_inference"
            SOURCES vulkan_rtdetr_inference.cpp
            rtdetr/RtDetrVk.cpp
            rtdetr/VkInfer.cpp
            rtdetr/WeightLoader.cpp)

    if (TARGET vulkan_rtdetr_inference)
        target_include_directories(vulkan_rtdetr_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(vulkan_rtdetr_inference PRIVATE Vulkan::Vulkan)

        # VARIANT_SUFFIX=rtdetr keeps the generated headers from colliding with
        # the YOLO kernels that share a basename (conv2d/add/maxpool/concat/upsample).
        set(_rt_sh "${CMAKE_CURRENT_SOURCE_DIR}/rtdetr/shaders")
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/conv2d.comp"   kRtConv2dSpv   _rh0 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/conv3x3.comp"  kRtConv3x3Spv  _rh0b VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/conv1x1.comp"  kRtConv1x1Spv  _rh0c VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/im2col.comp"   kRtIm2colSpv   _rh0d VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/dwconv.comp"   kRtDwConvSpv   _rh1 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/maxpool.comp"  kRtMaxPoolSpv  _rh2 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/concat.comp"   kRtConcatSpv   _rh3 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/add.comp"      kRtAddSpv      _rh4 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/add_silu.comp" kRtAddSiluSpv  _rh5 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/upsample.comp" kRtUpsampleSpv _rh6 VARIANT_SUFFIX rtdetr)
        # Phase B: AIFI transformer encoder.
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/linear.comp"               kRtLinearSpv     _rh7  VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/layernorm.comp"            kRtLayerNormSpv  _rh8  VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/softmax.comp"              kRtSoftmaxSpv    _rh9  VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/gelu.comp"                 kRtGeluSpv       _rh10 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/transpose_chw2tokens.comp" kRtT2tSpv        _rh11 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/transpose_tokens2chw.comp" kRtT2cSpv        _rh12 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/qkv_splice.comp"           kRtQkvSpliceSpv  _rh13 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/attn_scores.comp"          kRtAttnScoresSpv _rh14 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/attn_apply.comp"           kRtAttnApplySpv  _rh15 VARIANT_SUFFIX rtdetr)
        # Phase D: deformable decoder.
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/relu.comp"             kRtReluSpv     _rh16 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/transpose_mask.comp"   kRtTMaskSpv    _rh17 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/offset_preprocess.comp" kRtOffPreSpv  _rh18 VARIANT_SUFFIX rtdetr)
        compile_vulkan_shader(vulkan_rtdetr_inference "${_rt_sh}/msdeformattn.comp"     kRtMsDeformSpv _rh19 VARIANT_SUFFIX rtdetr)
    endif ()

    add_example(NAME "vulkan_rfdetr_inference"
            SOURCES vulkan_rfdetr_inference.cpp
            rfdetr/RfDetrVk.cpp
            rfdetr/VkInfer.cpp
            rfdetr/WeightLoader.cpp)

    if (TARGET vulkan_rfdetr_inference)
        target_include_directories(vulkan_rfdetr_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(vulkan_rfdetr_inference PRIVATE Vulkan::Vulkan)

        # VARIANT_SUFFIX=rfdetr — distinct generated headers from the rtdetr set.
        set(_rf_sh "${CMAKE_CURRENT_SOURCE_DIR}/rfdetr/shaders")
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/conv2d.comp"     kRfConv2dSpv     _rf0  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/conv1x1.comp"    kRfConv1x1Spv    _rf1  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/conv3x3.comp"    kRfConv3x3Spv    _rf2  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/im2col.comp"     kRfIm2colSpv     _rf3  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/linear.comp"     kRfLinearSpv     _rf4  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/layernorm.comp"  kRfLayerNormSpv  _rf5  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/softmax.comp"    kRfSoftmaxSpv    _rf6  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/gelu.comp"       kRfGeluSpv       _rf7  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/layerscale.comp" kRfLayerScaleSpv _rf8  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_scores.comp" kRfAttnScoresSpv _rf9 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_apply.comp" kRfAttnApplySpv  _rf10 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/add.comp"        kRfAddSpv        _rf11 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/concat.comp"     kRfConcatSpv     _rf12 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/build_embeddings.comp" kRfBuildEmbedSpv _rf13 VARIANT_SUFFIX rfdetr)
        # Batched separate-Q/K/V ViT attention (DINOv2 backbone).
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_vit_scores.comp" kRfAttnVitScoresSpv _rf14 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_vit_apply.comp"  kRfAttnVitApplySpv  _rf15 VARIANT_SUFFIX rfdetr)
        # C2f projector: de-window taps, channel-wise LayerNorm (+SiLU), channel slice/concat.
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/unwindow.comp"      kRfUnwindowSpv  _rf16 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/layernorm_chw.comp" kRfLnChwSpv     _rf17 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/chcopy.comp"        kRfChCopySpv    _rf18 VARIANT_SUFFIX rfdetr)
        # Two-stage deformable decoder.
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/transpose_chw2tokens.comp" kRfT2tSpv       _rf19 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/qkv_splice.comp"           kRfQkvSpliceSpv _rf20 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/relu.comp"                 kRfReluSpv      _rf21 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/offset_preprocess.comp"    kRfOffPreSpv    _rf22 VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/msdeformattn.comp"         kRfMsDeformSpv  _rf23 VARIANT_SUFFIX rfdetr)
        # Fused DINOv2 residual (LayerScale + add).
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/scale_add.comp"            kRfScaleAddSpv  _rf24 VARIANT_SUFFIX rfdetr)
    endif ()
endif ()