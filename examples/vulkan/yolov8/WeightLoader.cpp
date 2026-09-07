#include "WeightLoader.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace yolo {

namespace {

    template<typename T>
    T readLE(std::ifstream& f) {
        T v{};
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!f) throw std::runtime_error("yolo::loadWeights: unexpected end of file");
        return v;
    }

}// namespace

Weights parseWeightBinary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("yolo::loadWeights: cannot open '" + path + "'");

    // Magic
    char magic[4]{};
    f.read(magic, 4);
    if (!f || std::strncmp(magic, "YOLO", 4) != 0)
        throw std::runtime_error("yolo::loadWeights: bad magic in '" + path + "'");

    uint32_t version = readLE<uint32_t>(f);
    if (version != 1)
        throw std::runtime_error("yolo::loadWeights: unsupported version " + std::to_string(version));

    uint32_t numTensors = readLE<uint32_t>(f);

    Weights w;
    w.data.reserve(numTensors);
    w.shapes.reserve(numTensors);

    for (uint32_t t = 0; t < numTensors; ++t) {
        // Name
        uint32_t nameLen = readLE<uint32_t>(f);
        std::string name(nameLen, '\0');
        f.read(name.data(), nameLen);
        if (!f) throw std::runtime_error("yolo::loadWeights: failed reading tensor name");

        // Shape
        uint32_t ndim = readLE<uint32_t>(f);
        std::vector<uint32_t> shape(ndim);
        for (auto& s : shape) s = readLE<uint32_t>(f);

        // Data
        uint32_t dataBytes = readLE<uint32_t>(f);
        if (dataBytes % sizeof(float) != 0)
            throw std::runtime_error("yolo::loadWeights: data size not multiple of 4 for '" + name + "'");

        std::vector<float> data(dataBytes / sizeof(float));
        f.read(reinterpret_cast<char*>(data.data()), dataBytes);
        if (!f) throw std::runtime_error("yolo::loadWeights: failed reading data for '" + name + "'");

        w.data[name] = std::move(data);
        w.shapes[name] = std::move(shape);
    }

    return w;
}

}// namespace yolo
