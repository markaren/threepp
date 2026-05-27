#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace rfdetr {

    /// Maps PyTorch state_dict tensor names to flat float32 vectors.
    using WeightMap = std::unordered_map<std::string, std::vector<float>>;
    using ShapeMap  = std::unordered_map<std::string, std::vector<uint32_t>>;

    struct Weights {
        WeightMap data;    ///< name → float32 values
        ShapeMap  shapes;  ///< name → shape dims
    };

    /// Reads the binary format emitted by scripts/export_rtdetr_weights.py.
    /// Same format as the YoloV8 exporter (magic "YOLO"), re-parsed here to keep
    /// the rtdetr example self-contained. Throws std::runtime_error on failure.
    Weights parseWeightBinary(const std::string& path);

}// namespace rfdetr
