
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

    # The RF-DETR port's kernels, for each target that builds rfdetr/RfDetrVk.cpp.
    # VARIANT_SUFFIX rfdetr keeps the generated headers from colliding with the
    # YOLO kernels that share a basename (add/preprocess). gemm.comp is built once
    # per B layout (nn.Linear weight, 1x1-conv input, implicit im2col) and tile.
    function(rfdetr_shaders target)
        set(_rf_sh "${CMAKE_CURRENT_SOURCE_DIR}/rfdetr/shaders")
        # gemm.comp's tiles: 128x64 (8x8 per invocation, BK 32, no register prefetch)
        # and 64x64 (4x4, BK 16, prefetch); RfDetrVk::gemm_ picks one per GEMM.
        foreach (_layout NK KN CONV)
            string(TOLOWER ${_layout} _l)
            compile_vulkan_shader(${target} "${_rf_sh}/gemm.comp" kRfGemm${_layout}128x64Spv
                    VARIANT_SUFFIX rfdetr_${_l}128x64 DEFINES B_${_layout} BM=128 BN=64 BKD=32 NOPREFETCH)
            compile_vulkan_shader(${target} "${_rf_sh}/gemm.comp" kRfGemm${_layout}64x64Spv
                    VARIANT_SUFFIX rfdetr_${_l}64x64 DEFINES B_${_layout} BM=64 BN=64 TM=4 TN=4)
        endforeach ()
        # gemm_cm.comp: the tensor-core GEMM (VK_KHR_cooperative_matrix, fp16 products,
        # fp32 sums), 64x64 tiles, built per B layout like gemm.comp; used where the
        # device has it.
        compile_vulkan_shader(${target} "${_rf_sh}/gemm_cm.comp" kRfGemmCm64x64Spv
                VARIANT_SUFFIX rfdetr_cm64x64 DEFINES BM=64 BN=64)
        compile_vulkan_shader(${target} "${_rf_sh}/gemm_cm.comp" kRfGemmCmKnSpv
                VARIANT_SUFFIX rfdetr_cmkn DEFINES B_KN BM=64 BN=64)
        compile_vulkan_shader(${target} "${_rf_sh}/gemm_cm.comp" kRfGemmCmConvSpv
                VARIANT_SUFFIX rfdetr_cmconv DEFINES B_CONV BM=64 BN=64)
        compile_vulkan_shader(${target} "${_rf_sh}/gemm_reduce.comp"         kRfGemmReduceSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/layernorm.comp"           kRfLayerNormSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/softmax.comp"             kRfSoftmaxSpv       VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/attn_scores.comp"         kRfAttnScoresSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/attn_apply.comp"          kRfAttnApplySpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/add.comp"                 kRfAddSpv           VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/build_embeddings.comp"    kRfBuildEmbedSpv    VARIANT_SUFFIX rfdetr)
        # ViT attention: scores + softmax + apply, or fused (online softmax) where the
        # device has clustered subgroup ops.
        compile_vulkan_shader(${target} "${_rf_sh}/attn_vit_scores.comp"     kRfAttnVitScoresSpv VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/attn_vit_apply.comp"      kRfAttnVitApplySpv  VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/attn_vit_flash.comp"      kRfAttnFlashSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/attn_vit_flash_cm.comp"   kRfAttnFlashCmSpv   VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/attn_vit_flash_cm.comp"   kRfAttnFlashCm32Spv
                VARIANT_SUFFIX rfdetr_hd32 DEFINES HD=32)
        # C2f projector: de-window taps, channel-wise LayerNorm (+SiLU).
        compile_vulkan_shader(${target} "${_rf_sh}/unwindow.comp"            kRfUnwindowSpv      VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/layernorm_chw.comp"       kRfLnChwSpv         VARIANT_SUFFIX rfdetr)
        # Two-stage deformable decoder.
        compile_vulkan_shader(${target} "${_rf_sh}/transpose_chw2tokens.comp" kRfT2tSpv          VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/offset_preprocess.comp"   kRfOffPreSpv        VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/msdeformattn.comp"        kRfMsDeformSpv      VARIANT_SUFFIX rfdetr)
        # GPU preprocess and two-stage query selection: one submit per inference.
        compile_vulkan_shader(${target} "${_rf_sh}/preprocess.comp"          kRfPreprocessSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/proposals.comp"           kRfProposalsSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/topk_rank.comp"           kRfTopkRankSpv      VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/query_refs.comp"          kRfQueryRefsSpv     VARIANT_SUFFIX rfdetr)
        # The Keypoint variant's group self-attention.
        compile_vulkan_shader(${target} "${_rf_sh}/repeat_rows.comp"         kRfRepeatRowsSpv    VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/group_pack.comp"          kRfGroupPackSpv     VARIANT_SUFFIX rfdetr)
        compile_vulkan_shader(${target} "${_rf_sh}/group_unpack.comp"        kRfGroupUnpackSpv   VARIANT_SUFFIX rfdetr)
    endfunction ()

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

        rfdetr_shaders(vulkan_synthetic_inference)
    endif ()

    add_example(NAME "vulkan_rfdetr_inference"
            SOURCES vulkan_rfdetr_inference.cpp
            rfdetr/RfDetrVk.cpp
            rfdetr/VkInfer.cpp
            rfdetr/WeightLoader.cpp)

    if (TARGET vulkan_rfdetr_inference)
        target_include_directories(vulkan_rfdetr_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(vulkan_rfdetr_inference PRIVATE Vulkan::Vulkan)

        rfdetr_shaders(vulkan_rfdetr_inference)
    endif ()
endif ()
