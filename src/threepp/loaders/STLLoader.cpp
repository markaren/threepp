
#include "threepp/loaders/STLLoader.hpp"

#include <fstream>
#include <iostream>
#include <string>

using namespace threepp;


std::shared_ptr<BufferGeometry> STLLoader::load(const std::filesystem::path& path) const {

    if (!exists(path)) {
        std::cerr << "[STLLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }

    std::ifstream reader(path, std::ios::binary);
    reader.seekg(80, std::ios::beg);

    uint32_t faces;
    reader.read(reinterpret_cast<char*>(&faces), sizeof(uint32_t));

    const int dataOffset = 84;
    const int faceLength = 12 * 4 + 2;

    auto geometry = std::make_shared<BufferGeometry>();

    std::vector<float> vertices(faces * 3 * 3);
    std::vector<float> normals(faces * 3 * 3);

    float normalX;
    float normalY;
    float normalZ;

    for (uint32_t face = 0; face < faces; face++) {
        int start = dataOffset + face * faceLength;
        reader.seekg(start, std::ios::beg);

        reader.read(reinterpret_cast<char*>(&normalX), sizeof(float));
        reader.read(reinterpret_cast<char*>(&normalY), sizeof(float));
        reader.read(reinterpret_cast<char*>(&normalZ), sizeof(float));


        for (int i = 1; i <= 3; i++) {
            int vertexStart = start + i * 12;
            int componentIdx = face * 3 * 3 + (i - 1) * 3;
            reader.seekg(vertexStart, std::ios::beg);

            vertices.emplace_back();
            vertices.emplace_back();
            vertices.emplace_back();

            reader.read(reinterpret_cast<char*>(&vertices[componentIdx]), 4);
            reader.read(reinterpret_cast<char*>(&vertices[componentIdx + 1]), 4);
            reader.read(reinterpret_cast<char*>(&vertices[componentIdx + 2]), 4);

            normals[componentIdx] = normalX;
            normals[componentIdx + 1] = normalY;
            normals[componentIdx + 2] = normalZ;
        }
    }

    geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));

    return geometry;
}
