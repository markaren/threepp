#include "threepp/loaders/FBXLoader.hpp"

#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include "ofbx.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace threepp {

    namespace {

        // OpenFBX uses row-vector convention (translation in last row).
        // Transposing gives the column-vector Matrix4 expected by threepp.
        // Matrix4::set takes elements in row-major reading order.
        Matrix4 toMatrix4(const ofbx::DMatrix& m) {
            return Matrix4().set(
                (float)m.m[0],  (float)m.m[4],  (float)m.m[8],  (float)m.m[12],
                (float)m.m[1],  (float)m.m[5],  (float)m.m[9],  (float)m.m[13],
                (float)m.m[2],  (float)m.m[6],  (float)m.m[10], (float)m.m[14],
                (float)m.m[3],  (float)m.m[7],  (float)m.m[11], (float)m.m[15]);
        }

        // Build a non-indexed BufferGeometry from one GeometryPartition.
        // ofbx::triangulate() converts each polygon in the partition to triangles
        // and returns face-vertex indices; Vec*Attributes::get() resolves
        // both indexed and non-indexed storage transparently.
        std::shared_ptr<BufferGeometry> buildPartitionGeometry(
                const ofbx::GeometryData& geomData,
                const ofbx::GeometryPartition& partition) {

            const auto positions = geomData.getPositions();
            const auto normals   = geomData.getNormals();
            const auto uvs       = geomData.getUVs(0);
            const bool hasNormals = normals.values != nullptr;
            const bool hasUVs     = uvs.values != nullptr;

            std::vector<float> posData;
            std::vector<float> normData;
            std::vector<float> uvData;
            posData.reserve(partition.triangles_count * 9);
            if (hasNormals) normData.reserve(partition.triangles_count * 9);
            if (hasUVs)     uvData.reserve(partition.triangles_count * 6);

            // Scratch buffer for triangulate() — sized for the largest polygon.
            std::vector<int> triIndices(partition.max_polygon_triangles * 3);

            for (int pi = 0; pi < partition.polygon_count; ++pi) {
                const auto& polygon = partition.polygons[pi];
                // Returns number of resulting indices (numTriangles * 3).
                const int indexCount = (int)ofbx::triangulate(geomData, polygon, triIndices.data());
                for (int ti = 0; ti < indexCount; ++ti) {
                    const int fv = triIndices[ti];
                    const auto p = positions.get(fv);
                    posData.push_back((float)p.x);
                    posData.push_back((float)p.y);
                    posData.push_back((float)p.z);
                    if (hasNormals) {
                        const auto n = normals.get(fv);
                        normData.push_back((float)n.x);
                        normData.push_back((float)n.y);
                        normData.push_back((float)n.z);
                    }
                    if (hasUVs) {
                        const auto uv = uvs.get(fv);
                        uvData.push_back((float)uv.x);
                        uvData.push_back((float)uv.y);
                    }
                }
            }

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position",
                    FloatBufferAttribute::create(std::move(posData), 3));
            if (!normData.empty())
                geometry->setAttribute("normal",
                        FloatBufferAttribute::create(std::move(normData), 3));
            if (!uvData.empty())
                geometry->setAttribute("uv",
                        FloatBufferAttribute::create(std::move(uvData), 2));
            if (normData.empty())
                geometry->computeVertexNormals();

            return geometry;
        }

        std::filesystem::path resolveTexturePath(
                const ofbx::Texture* tex,
                const std::filesystem::path& baseDir) {
            if (!tex) return {};
            char relBuf[512] = {};
            char absBuf[512] = {};
            tex->getRelativeFileName().toString(relBuf);
            tex->getFileName().toString(absBuf);

            for (char* p = relBuf; *p; ++p) if (*p == '\\') *p = '/';
            for (char* p = absBuf; *p; ++p) if (*p == '\\') *p = '/';

            // Strip leading "./" so filesystem::path joining works correctly.
            auto stripDotSlash = [](const char* s) -> const char* {
                return (s[0] == '.' && s[1] == '/') ? s + 2 : s;
            };
            const char* rel = stripDotSlash(relBuf);
            const char* abs = absBuf;

            // 1. baseDir + relative path (most common for well-packaged FBX files)
            if (rel[0]) {
                auto candidate = baseDir / rel;
                if (std::filesystem::exists(candidate)) return candidate;
            }
            // 2. absolute path as stored (works when running on the original machine)
            if (abs[0] && std::filesystem::exists(abs)) return abs;
            // 3. filename-only from relative path, next to the FBX file
            if (rel[0]) {
                auto candidate = baseDir / std::filesystem::path(rel).filename();
                if (std::filesystem::exists(candidate)) return candidate;
            }
            // 4. filename-only from absolute path, next to the FBX file
            if (abs[0]) {
                auto candidate = baseDir / std::filesystem::path(abs).filename();
                if (std::filesystem::exists(candidate)) return candidate;
            }
            return {};
        }

        bool isSupportedImageFormat(const std::filesystem::path& p) {
            const auto ext = p.extension().string();
            // stb_image supports these; DDS and other GPU-compressed formats are not supported.
            for (const auto* e : {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".hdr", ".pic", ".pnm", ".dds"})
                if (ext == e) return true;
            return false;
        }

        std::shared_ptr<MeshPhongMaterial> buildMaterial(
                const ofbx::Material* mat,
                const std::filesystem::path& baseDir,
                TextureLoader& texLoader) {
            auto material = MeshPhongMaterial::create();
            if (!mat) return material;
            const auto dc = mat->getDiffuseColor();
            material->color.setRGB(dc.r, dc.g, dc.b);
            const auto sc = mat->getSpecularColor();
            material->specular.setRGB(sc.r, sc.g, sc.b);
            if (const auto* diffTex = mat->getTexture(ofbx::Texture::DIFFUSE)) {
                auto texPath = resolveTexturePath(diffTex, baseDir);
                if (!texPath.empty() && isSupportedImageFormat(texPath)) {
                    auto tex = texLoader.load(texPath, true);
                    if (tex) {
                        tex->wrapS = TextureWrapping::Repeat;
                        tex->wrapT = TextureWrapping::Repeat;
                        tex->needsUpdate();
                    }
                    material->map = tex;
                    material->color.setHex(0xffffff);
                }
            }
            return material;
        }

    }// namespace

    struct FBXLoader::Impl {
        TextureLoader texLoader;
    };

    FBXLoader::FBXLoader() : pimpl_(std::make_unique<Impl>()) {}
    FBXLoader::~FBXLoader() = default;

    std::shared_ptr<Group> FBXLoader::load(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "[FBXLoader] File does not exist: "
                      << std::filesystem::absolute(path) << std::endl;
            return nullptr;
        }

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "[FBXLoader] Cannot open: " << path << std::endl;
            return nullptr;
        }
        const auto fileSize = file.tellg();
        file.seekg(0);
        std::vector<ofbx::u8> data(static_cast<size_t>(fileSize));
        file.read(reinterpret_cast<char*>(data.data()), fileSize);
        file.close();

        ofbx::IScene* scene = ofbx::load(
                data.data(),
                static_cast<ofbx::usize>(data.size()),
                static_cast<ofbx::u16>(ofbx::LoadFlags::NONE));

        if (!scene) {
            std::cerr << "[FBXLoader] Parse error: " << ofbx::getError() << std::endl;
            return nullptr;
        }

        const std::filesystem::path baseDir = path.parent_path();
        auto root = Group::create();
        root->name = path.stem().string();

        const int meshCount = scene->getMeshCount();
        for (int mi = 0; mi < meshCount; ++mi) {
            const ofbx::Mesh* fbxMesh = scene->getMesh(mi);
            const ofbx::GeometryData& geomData = fbxMesh->getGeometryData();
            if (!geomData.hasVertices()) continue;

            const int partCount = geomData.getPartitionCount();
            if (partCount == 0) continue;

            Matrix4 worldMatrix;
            worldMatrix.multiplyMatrices(
                    toMatrix4(fbxMesh->getGlobalTransform()),
                    toMatrix4(fbxMesh->getGeometricMatrix()));

            if (partCount == 1) {
                const auto part = geomData.getPartition(0);
                if (part.triangles_count == 0) continue;

                auto geometry = buildPartitionGeometry(geomData, part);
                const ofbx::Material* mat = fbxMesh->getMaterialCount() > 0
                        ? fbxMesh->getMaterial(0) : nullptr;
                auto material = buildMaterial(mat, baseDir, pimpl_->texLoader);

                auto mesh = Mesh::create(geometry, material);
                mesh->name = fbxMesh->name;
                mesh->applyMatrix4(worldMatrix);
                root->add(mesh);
            } else {
                // One sub-mesh per partition (material group).
                auto meshGroup = Group::create();
                meshGroup->name = fbxMesh->name;
                meshGroup->applyMatrix4(worldMatrix);

                for (int pi = 0; pi < partCount; ++pi) {
                    const auto part = geomData.getPartition(pi);
                    if (part.triangles_count == 0) continue;
                    auto geometry = buildPartitionGeometry(geomData, part);
                    const ofbx::Material* mat = pi < fbxMesh->getMaterialCount()
                            ? fbxMesh->getMaterial(pi) : nullptr;
                    auto material = buildMaterial(mat, baseDir, pimpl_->texLoader);
                    meshGroup->add(Mesh::create(geometry, material));
                }
                root->add(meshGroup);
            }
        }

        scene->destroy();
        return root;
    }

}// namespace threepp
