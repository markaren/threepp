#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace yolo {

    /// Maps PyTorch state_dict tensor names to flat float32 vectors.
    /// Tensor data is stored in PyTorch's native NCHW layout.
    using WeightMap = std::unordered_map<std::string, std::vector<float>>;

    /// Shape info stored alongside weight data.
    using ShapeMap = std::unordered_map<std::string, std::vector<uint32_t>>;

    struct Weights {
        WeightMap data;   ///< name → float32 values
        ShapeMap shapes;  ///< name → shape dims
    };

    /// Load weights from the custom binary format exported by export_yolov8n_weights.py.
    /// Format: magic "YOLO", version u32, num_tensors u32,
    ///   per tensor: name_len u32, name bytes, ndim u32, shape[ndim] u32[], data_bytes u32, float32[].
    /// Throws std::runtime_error on any read failure.
    Weights parseWeightBinary(const std::string& path);

}// namespace yolo
