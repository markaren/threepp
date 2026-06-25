
#include "threepp/loaders/STLLoader.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // A binary STL has a fixed layout: 80-byte header + uint32 face count + 50 bytes/face
    // (12-byte normal + 3*12-byte vertices + 2-byte attribute). An ASCII STL is text
    // starting with "solid " followed by "facet"/"vertex" keywords. Some binary writers
    // also start the header with "solid", so the size check is the primary discriminator
    // (matching three.js' STLLoader heuristic).
    bool isBinary(const std::vector<char>& data) {

        constexpr size_t headerLength = 80;
        constexpr size_t faceLength = 12 * 4 + 2;// 50

        if (data.size() < headerLength + sizeof(uint32_t)) {
            // Too small to hold a binary header — must be (truncated) ASCII.
            return false;
        }

        uint32_t faces;
        std::memcpy(&faces, data.data() + headerLength, sizeof(uint32_t));

        const size_t expect = headerLength + sizeof(uint32_t) + static_cast<size_t>(faces) * faceLength;
        if (expect == data.size()) {
            return true;
        }

        // Size didn't match: if the file does not begin with "solid", assume binary.
        static const char solid[] = {'s', 'o', 'l', 'i', 'd'};
        for (size_t i = 0; i < sizeof(solid); ++i) {
            if (data[i] != solid[i]) {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<BufferGeometry> parseBinary(const std::vector<char>& data) {

        uint32_t faces;
        std::memcpy(&faces, data.data() + 80, sizeof(uint32_t));

        constexpr size_t dataOffset = 84;
        constexpr size_t faceLength = 12 * 4 + 2;

        const size_t expect = dataOffset + static_cast<size_t>(faces) * faceLength;
        if (data.size() < expect) {
            std::cerr << "[STLLoader] Truncated binary STL: header claims " << faces
                      << " faces but file is too small!" << std::endl;
            return nullptr;
        }

        std::vector<float> vertices(static_cast<size_t>(faces) * 3 * 3);
        std::vector<float> normals(static_cast<size_t>(faces) * 3 * 3);

        for (uint32_t face = 0; face < faces; ++face) {
            const size_t start = dataOffset + static_cast<size_t>(face) * faceLength;

            float normalX, normalY, normalZ;
            std::memcpy(&normalX, data.data() + start + 0, 4);
            std::memcpy(&normalY, data.data() + start + 4, 4);
            std::memcpy(&normalZ, data.data() + start + 8, 4);

            for (int i = 1; i <= 3; ++i) {
                const size_t vertexStart = start + static_cast<size_t>(i) * 12;
                const size_t componentIdx = static_cast<size_t>(face) * 3 * 3 + static_cast<size_t>(i - 1) * 3;

                std::memcpy(&vertices[componentIdx + 0], data.data() + vertexStart + 0, 4);
                std::memcpy(&vertices[componentIdx + 1], data.data() + vertexStart + 4, 4);
                std::memcpy(&vertices[componentIdx + 2], data.data() + vertexStart + 8, 4);

                normals[componentIdx + 0] = normalX;
                normals[componentIdx + 1] = normalY;
                normals[componentIdx + 2] = normalZ;
            }
        }

        auto geometry = std::make_shared<BufferGeometry>();
        geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));

        return geometry;
    }

    std::shared_ptr<BufferGeometry> parseASCII(const std::vector<char>& data) {

        std::vector<float> vertices;
        std::vector<float> normals;

        std::istringstream stream(std::string(data.begin(), data.end()));
        std::string token;

        float nx = 0.f, ny = 0.f, nz = 0.f;

        while (stream >> token) {
            if (token == "facet") {
                std::string keyword;
                stream >> keyword;// "normal"
                stream >> nx >> ny >> nz;
            } else if (token == "vertex") {
                float x, y, z;
                stream >> x >> y >> z;
                vertices.insert(vertices.end(), {x, y, z});
                normals.insert(normals.end(), {nx, ny, nz});
            }
        }

        auto geometry = std::make_shared<BufferGeometry>();
        geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));

        return geometry;
    }

}// namespace


std::shared_ptr<BufferGeometry> STLLoader::load(const std::filesystem::path& path) const {

    if (!exists(path)) {
        std::cerr << "[STLLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }

    std::ifstream reader(path, std::ios::binary | std::ios::ate);
    if (!reader) {
        std::cerr << "[STLLoader] Failed to open file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }

    const auto fileSize = static_cast<std::streamoff>(reader.tellg());
    if (fileSize <= 0) {
        std::cerr << "[STLLoader] Empty file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }
    reader.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<size_t>(fileSize));
    reader.read(data.data(), fileSize);

    return isBinary(data) ? parseBinary(data) : parseASCII(data);
}
