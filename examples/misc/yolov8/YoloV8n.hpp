#pragma once

#include "GPUTensor.hpp"
#include "WeightLoader.hpp"

#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace yolo {

    struct Detection {
        float x1, y1, x2, y2;  ///< Box in input image pixels (640×640 space)
        float conf;             ///< Object confidence (sigmoid of best class logit)
        int   cls_id;           ///< COCO class index (0-based)
    };

    /// YOLOv8n object detector running entirely on the WebGPU compute pipeline.
    /// Load pre-trained weights exported by scripts/export_yolov8n_weights.py,
    /// then call infer() with any RGBA image.
    class YoloV8n {
    public:
        static constexpr int INPUT_SIZE    = 640;
        static constexpr int NUM_CLASSES   = 80;
        static constexpr int REG_MAX       = 16;

        explicit YoloV8n(threepp::WgpuRenderer& renderer);
        ~YoloV8n();

        YoloV8n(const YoloV8n&) = delete;
        YoloV8n& operator=(const YoloV8n&) = delete;

        /// Upload weights from the .bin file into GPU buffers.
        void loadWeights(const std::string& path);

        /// Run inference on an RGBA image (channels = 4, uint8).
        /// The image is bilinearly resized to 640×640 on CPU before upload.
        /// Returns detections clipped to the 640×640 model space.
        std::vector<Detection> infer(const unsigned char* rgba,
                                     int width, int height,
                                     float confThresh = 0.25f,
                                     float iouThresh  = 0.45f);

    private:
        threepp::WgpuRenderer& renderer_;

        // One pipeline per operation type (reused for every layer of that type)
        threepp::WgpuComputePipeline convPipe_;
        threepp::WgpuComputePipeline conv3x3s1Pipe_;
        threepp::WgpuComputePipeline conv3x3s2Pipe_;
        threepp::WgpuComputePipeline conv1x1Pipe_;
        threepp::WgpuComputePipeline bnSiluPipe_;
        threepp::WgpuComputePipeline addPipe_;
        threepp::WgpuComputePipeline maxpoolPipe_;
        threepp::WgpuComputePipeline upsamplePipe_;
        threepp::WgpuComputePipeline concatPipe_;
        threepp::WgpuComputePipeline slicePipe_;
        threepp::WgpuComputePipeline detectPipe_;
        threepp::WgpuComputePipeline detectFusedPipe_;
        threepp::WgpuComputePipeline preprocessPipe_;

        // Uniform buffers for per-layer params (written fresh before each dispatch)
        threepp::WgpuBuffer convParamBuf_;
        threepp::WgpuBuffer bnParamBuf_;
        threepp::WgpuBuffer addParamBuf_;
        threepp::WgpuBuffer poolParamBuf_;
        threepp::WgpuBuffer upParamBuf_;
        threepp::WgpuBuffer catParamBuf_;
        threepp::WgpuBuffer sliceParamBuf_;
        threepp::WgpuBuffer detectParamBuf_;
        threepp::WgpuBuffer prepParamBuf_;

        // Dummy 1-float zero bias (used when a layer has no bias)
        threepp::WgpuBuffer dummyBias_;

        // Letterbox state (written by preprocess_, consumed to un-letterbox boxes).
        float lbScale_ = 1.f;
        float lbPadX_  = 0.f;
        float lbPadY_  = 0.f;

        // GPU detection pre-filter: shared counter + dets across all 3 scales.
        // The decode kernel atomically appends only anchors with conf >= thresh,
        // so readback is bounded by survivors (typically <100) instead of 8400.
        static constexpr uint32_t MAX_DETS = 1024;
        std::unique_ptr<threepp::WgpuBuffer> detCounterBuf_;
        std::unique_ptr<threepp::WgpuBuffer> detDetsBuf_;

        // Weight tensors uploaded to GPU (keyed by PyTorch state_dict name)
        std::unordered_map<std::string, GPUTensor> weights_;

        // ---------------------------------------------------------------
        // Low-level GPU operation helpers
        // ---------------------------------------------------------------

        /// Convolution: weight/bias keys are looked up in weights_.
        /// kH/kW are derived from the weight tensor shape.
        /// If biasKey is empty, no bias is added.
        GPUTensor conv_(const GPUTensor& x,
                        const std::string& weightKey,
                        const std::string& biasKey,
                        int strideH = 1, int strideW = 1,
                        bool silu = false,
                        const GPUTensor* residual = nullptr);

        GPUTensor bnSilu_(const GPUTensor& x, const std::string& bnPrefix);

        GPUTensor add_(const GPUTensor& a, const GPUTensor& b);

        GPUTensor maxpool_(const GPUTensor& x, int k, int stride, int pad);

        GPUTensor upsample2x_(const GPUTensor& x);

        GPUTensor concat_(const GPUTensor& a, const GPUTensor& b);

        GPUTensor sliceChannels_(const GPUTensor& x, uint32_t offsetC, uint32_t countC);

        // ---------------------------------------------------------------
        // Compound block helpers
        // ---------------------------------------------------------------

        /// Conv block: conv + BN + SiLU.
        /// Kernel size derived from weight shape (prefix+".conv.weight").
        GPUTensor convBnSilu_(const GPUTensor& x, const std::string& prefix, int stride = 1,
                              const GPUTensor* residual = nullptr);

        /// Plain Conv2d (no BN, with bias). Weight/bias at prefix+".weight" / prefix+".bias".
        GPUTensor plainConv_(const GPUTensor& x, const std::string& prefix);

        /// C2f block with n bottleneck sub-layers.
        GPUTensor c2f_(const GPUTensor& x, const std::string& prefix, int n, bool shortcut);

        /// SPPF block.
        GPUTensor sppf_(const GPUTensor& x, const std::string& prefix);

        /// Concatenate a list of tensors along the channel axis pairwise.
        GPUTensor concatMany_(std::vector<GPUTensor>& parts);

        /// A sub-channel-range view into a source buffer — used to concat
        /// slices without materialising each slice as its own tensor.
        struct SourceView {
            threepp::WgpuBuffer* buf;
            uint32_t srcTotalC;   ///< total channel count of source tensor
            uint32_t srcOffsetC;  ///< first channel to read
            uint32_t countC;      ///< number of channels to copy
        };

        /// Concat arbitrary sub-ranges of source buffers into a new tensor.
        /// All views must share spatial dims (H, W).
        GPUTensor concatViews_(const std::vector<SourceView>& views,
                               uint32_t H, uint32_t W);

        // ---------------------------------------------------------------
        // Detect head + NMS
        // ---------------------------------------------------------------

        /// Decode one detection scale on the GPU. Atomically appends anchors
        /// with conf >= confThresh into the shared (counter, dets) buffers.
        void decodeScaleGpu_(const GPUTensor& feat, int scaleIdx, float stride,
                             float confThresh);

        /// CPU-side IoU NMS.
        std::vector<Detection> nms_(std::vector<Detection>& dets, float iouThresh);

        // ---------------------------------------------------------------
        // GPU readback
        // ---------------------------------------------------------------

        /// Copy a StorageReadback buffer to CPU. Uses raw WebGPU commands.
        std::vector<float> readback_(threepp::WgpuBuffer& buf, size_t floatCount);

        /// Batched readback: copies all source buffers in one encoder submit
        /// and maps each, returning a concatenated result (in order).
        std::vector<std::vector<float>> readbackMany_(
                std::vector<std::pair<threepp::WgpuBuffer*, size_t>> sources);

        // ---------------------------------------------------------------
        // Preprocessing
        // ---------------------------------------------------------------

        /// Bilinear resize + normalise RGBA→float [0,1] RGB, upload to GPU.
        GPUTensor preprocess_(const unsigned char* rgba, int width, int height);
    };

}// namespace yolo
