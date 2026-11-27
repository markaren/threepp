
add_example(NAME "vulkan_gltf_samples" LINK_IMGUI)
add_example(NAME "vulkan_showcase" LINK_IMGUI)
add_example(NAME "vulkan_lights" LINK_IMGUI)
add_example(NAME "vulkan_physical_camera" LINK_IMGUI)
add_example(NAME "vulkan_fog" LINK_IMGUI)
add_example(NAME "vulkan_ridge_fog" LINK_IMGUI)
add_example(NAME "vulkan_fire" LINK_IMGUI)
add_example(NAME "vulkan_snow" LINK_IMGUI)
add_example(NAME "vulkan_smoke")
add_example(NAME "vulkan_smoke_drone")
add_example(NAME "vulkan_fjord" LINK_IMGUI)
add_example(NAME "vulkan_drone_clouds" LINK_IMGUI)
add_example(NAME "vulkan_ocean_minimal")
add_example(NAME "vulkan_denoise" LINK_IMGUI)
add_example(NAME "vulkan_animation" LINK_IMGUI)
add_example(NAME "vulkan_gallery" LINK_IMGUI)
add_example(NAME "vulkan_restir_test" LINK_IMGUI)
add_example(NAME "vulkan_lidar" LINK_IMGUI)
add_example(NAME "vulkan_sonar" LINK_IMGUI)
add_example(NAME "vulkan_event_camera" LINK_IMGUI)

# audits/: harnesses that measure the renderer rather than show it. Same
# targets as any example; kept apart so the files beside this one read as
# the examples.
add_example(NAME "vulkan_furnace_test" SOURCES "audits/vulkan_furnace_test.cpp" LINK_IMGUI)
add_example(NAME "vulkan_aov_audit" SOURCES "audits/vulkan_aov_audit.cpp")
add_example(NAME "vulkan_lidar_audit" SOURCES "audits/vulkan_lidar_audit.cpp")
# Render-graph A/B capture harness (see the file comment and
# plans/render-graph.md, "Verification method"). Run by hand; --selfcheck
# fails on any render-graph diagnostic in a scene.
add_example(NAME "rg_ab" SOURCES "audits/rg_ab.cpp")
# Ocean sea-state audit: Hs, Tp, dispersion, slopes and crests of the FFT
# ocean against published measurements (see the file comment).
add_example(NAME "ocean_metrics" SOURCES "audits/ocean_metrics.cpp")

# Local scratch harnesses (capture / repro / profiling tools) are not tracked —
# see scratch/README.md and .gitignore. Auto-register every scratch/*.cpp as its
# own target so this tracked build file never churns as harnesses come and go.
# CONFIGURE_DEPENDS re-globs on each build; an empty/absent scratch/ is a no-op.
file(GLOB _scratch_examples CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scratch/*.cpp")
foreach (_scratch_src ${_scratch_examples})
    get_filename_component(_scratch_name "${_scratch_src}" NAME_WE)
    add_example(NAME "${_scratch_name}" SOURCES "scratch/${_scratch_name}.cpp" LINK_IMGUI)
endforeach ()

add_example(NAME "vulkan_bistro" LINK_IMGUI)

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
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv2d.comp"       kConv2dSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv1x1.comp"      kConv1x1Spv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv3x3s1.comp"    kConv3x3s1Spv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/conv3x3s2.comp"    kConv3x3s2Spv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/preprocess.comp"   kPreprocessSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/bn_silu.comp"      kBnSiluSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/add.comp"          kAddSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/maxpool.comp"      kMaxpoolSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/upsample.comp"     kUpsampleSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/concat.comp"       kConcatSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/slice.comp"        kSliceSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/detect.comp"       kDetectSpv)
        compile_vulkan_shader(vulkan_yolov8_inference "${_yolo_sh}/detect_fused.comp" kDetectFusedSpv)
    endif ()

    # Synthetic perception loop: render an animated scene with the deferred
    # path, detect in the rendered frames with the Vulkan RF-DETR-Nano port,
    # overlay the detections live. No image files involved.
    add_example(NAME "vulkan_synthetic_inference"
            SOURCES vulkan_synthetic_inference.cpp
            rfdetr/RfDetrVk.cpp
            rfdetr/VkInfer.cpp
            rfdetr/WeightLoader.cpp)

    if (TARGET vulkan_synthetic_inference)
        target_include_directories(vulkan_synthetic_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(vulkan_synthetic_inference PRIVATE Vulkan::Vulkan)

        set(_rf_sh "${CMAKE_CURRENT_SOURCE_DIR}/rfdetr/shaders")
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/conv2d.comp"     kRfConv2dSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/conv1x1.comp"    kRfConv1x1Spv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/conv3x3.comp"    kRfConv3x3Spv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/im2col.comp"     kRfIm2colSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/linear.comp"     kRfLinearSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/layernorm.comp"  kRfLayerNormSpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/softmax.comp"    kRfSoftmaxSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/gelu.comp"       kRfGeluSpv       VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/layerscale.comp" kRfLayerScaleSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/attn_scores.comp" kRfAttnScoresSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/attn_apply.comp" kRfAttnApplySpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/add.comp"        kRfAddSpv        VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/concat.comp"     kRfConcatSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/build_embeddings.comp" kRfBuildEmbedSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/attn_vit_scores.comp" kRfAttnVitScoresSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/attn_vit_apply.comp"  kRfAttnVitApplySpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/unwindow.comp"      kRfUnwindowSpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/layernorm_chw.comp" kRfLnChwSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/chcopy.comp"        kRfChCopySpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/transpose_chw2tokens.comp" kRfT2tSpv       VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/qkv_splice.comp"           kRfQkvSpliceSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/relu.comp"                 kRfReluSpv      VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/offset_preprocess.comp"    kRfOffPreSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/msdeformattn.comp"         kRfMsDeformSpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/scale_add.comp"            kRfScaleAddSpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/linear_splitk.comp"        kRfLinearSplitKSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/reduce_splitk.comp"        kRfReduceSplitKSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/preprocess.comp"           kRfPreprocessSpv   VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/proposals.comp"            kRfProposalsSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/topk_rank.comp"            kRfTopkRankSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_synthetic_inference "${_rf_sh}/query_refs.comp"           kRfQueryRefsSpv    VARIANT_SUFFIX rfdetr)
    endif ()

    add_example(NAME "vulkan_rfdetr_inference"
            SOURCES vulkan_rfdetr_inference.cpp
            rfdetr/RfDetrVk.cpp
            rfdetr/VkInfer.cpp
            rfdetr/WeightLoader.cpp)

    if (TARGET vulkan_rfdetr_inference)
        target_include_directories(vulkan_rfdetr_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(vulkan_rfdetr_inference PRIVATE Vulkan::Vulkan)

        # VARIANT_SUFFIX=rfdetr keeps the generated headers from colliding with
        # the YOLO kernels that share a basename (conv2d/conv1x1/add/concat).
        set(_rf_sh "${CMAKE_CURRENT_SOURCE_DIR}/rfdetr/shaders")
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/conv2d.comp"     kRfConv2dSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/conv1x1.comp"    kRfConv1x1Spv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/conv3x3.comp"    kRfConv3x3Spv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/im2col.comp"     kRfIm2colSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/linear.comp"     kRfLinearSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/layernorm.comp"  kRfLayerNormSpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/softmax.comp"    kRfSoftmaxSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/gelu.comp"       kRfGeluSpv       VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/layerscale.comp" kRfLayerScaleSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_scores.comp" kRfAttnScoresSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_apply.comp" kRfAttnApplySpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/add.comp"        kRfAddSpv        VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/concat.comp"     kRfConcatSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/build_embeddings.comp" kRfBuildEmbedSpv VARIANT_SUFFIX rfdetr)
        # Batched separate-Q/K/V ViT attention (DINOv2 backbone).
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_vit_scores.comp" kRfAttnVitScoresSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/attn_vit_apply.comp"  kRfAttnVitApplySpv  VARIANT_SUFFIX rfdetr)
        # C2f projector: de-window taps, channel-wise LayerNorm (+SiLU), channel slice/concat.
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/unwindow.comp"      kRfUnwindowSpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/layernorm_chw.comp" kRfLnChwSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/chcopy.comp"        kRfChCopySpv    VARIANT_SUFFIX rfdetr)
        # Two-stage deformable decoder.
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/transpose_chw2tokens.comp" kRfT2tSpv       VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/qkv_splice.comp"           kRfQkvSpliceSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/relu.comp"                 kRfReluSpv      VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/offset_preprocess.comp"    kRfOffPreSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/msdeformattn.comp"         kRfMsDeformSpv  VARIANT_SUFFIX rfdetr)
        # Fused DINOv2 residual (LayerScale + add).
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/scale_add.comp"            kRfScaleAddSpv  VARIANT_SUFFIX rfdetr)
        # Split-K GEMM (partial accumulation + reduction) for low-N occupancy-bound linears.
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/linear_splitk.comp"        kRfLinearSplitKSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/reduce_splitk.comp"        kRfReduceSplitKSpv VARIANT_SUFFIX rfdetr)
        # GPU preprocess and two-stage query selection: one submit per inference.
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/preprocess.comp"           kRfPreprocessSpv   VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/proposals.comp"            kRfProposalsSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/topk_rank.comp"            kRfTopkRankSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(vulkan_rfdetr_inference "${_rf_sh}/query_refs.comp"           kRfQueryRefsSpv    VARIANT_SUFFIX rfdetr)
    endif ()
endif ()
