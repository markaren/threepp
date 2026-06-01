#include "threepp/loaders/FBXLoader.hpp"

#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include "ofbx.h"

#include <algorithm>
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

        // Returns true when the filename suggests a PBR ORM-packed texture
        // (R=AO, G=Roughness, B=Metalness).  This heuristic covers Unreal Engine
        // exports (e.g. "_Specular") and common explicit ORM naming.
        // Used only in MaterialMode::Auto; Phong/PBR override it via the loader flag.
        bool looksLikeORM(const std::filesystem::path& p) {
            const auto stem = p.stem().string();
            // Case-insensitive search for known ORM suffixes/substrings.
            auto ci = [&](const char* needle) {
                auto it = std::search(stem.begin(), stem.end(),
                                      needle, needle + std::strlen(needle),
                                      [](char a, char b){ return std::tolower(a) == std::tolower(b); });
                return it != stem.end();
            };
            return ci("_specular") || ci("_orm") || ci("_rma") || ci("_occlusionroughnessmetallic")
                || ci("_roughness") || ci("_metalrough") || ci("_metallicrough");
        }

        // Load a texture from the given FBX texture slot, configure wrapping,
        // and return it.  Returns nullptr if the slot is empty or unsupported.
        std::shared_ptr<Texture> loadTex(const ofbx::Texture* slot,
                                         const std::filesystem::path& baseDir,
                                         TextureLoader& texLoader,
                                         ColorSpace cs = ColorSpace::sRGB) {
            if (!slot) return nullptr;
            auto p = resolveTexturePath(slot, baseDir);
            if (p.empty() || !isSupportedImageFormat(p)) return nullptr;
            auto tex = texLoader.load(p, cs);
            if (tex) {
                tex->wrapS = TextureWrapping::Repeat;
                tex->wrapT = TextureWrapping::Repeat;
                tex->needsUpdate();
            }
            return tex;
        }

        // Returns true when the material name or diffuse texture name suggests glass.
        // Also covers the FBX opacity property for materials that set it explicitly.
        bool looksLikeGlass(const ofbx::Material* mat, const std::filesystem::path& diffPath) {
            // Explicit FBX opacity property
            if (static_cast<float>(mat->getOpacity()) < 0.99f) return true;
            // Name-based heuristic: material name or diffuse texture stem
            auto ciContains = [](const std::string& s, const char* needle) {
                return std::search(s.begin(), s.end(), needle, needle + std::strlen(needle),
                                   [](char a, char b){ return std::tolower(a) == std::tolower(b); }) != s.end();
            };
            const std::string matName(mat->name);
            for (const char* kw : {"glass", "window", "crystal", "transparent"}) {
                if (ciContains(matName, kw)) return true;
            }
            if (!diffPath.empty()) {
                const std::string stem = diffPath.stem().string();
                for (const char* kw : {"glass", "window", "crystal"}) {
                    if (ciContains(stem, kw)) return true;
                }
            }
            return false;
        }

        // Apply normal map + emissive — shared between all material types.
        // Opacity/transmission is handled per-branch.
        template<typename M>
        void applyCommon(M& m, const ofbx::Material* mat,
                         const std::filesystem::path& baseDir,
                         TextureLoader& texLoader,
                         float emissiveScale) {
            if (auto tex = loadTex(mat->getTexture(ofbx::Texture::NORMAL), baseDir, texLoader, ColorSpace::Linear)) {
                m.normalMap   = tex;
                m.normalScale = {1.0f, -1.0f};  // DirectX → OpenGL Y-flip
            }

            // Emissive. OpenFBX returns its struct default of white (1,1,1) /
            // factor 1 when the FBX omits these properties — it can't report
            // "absent" — so reading them unconditionally would make every
            // non-emissive material glow. Treat a material as emissive only when
            // it has an emissive texture or an explicit, non-default emissive
            // colour. (A genuine white emitter with no texture is therefore
            // skipped, which is rare; real assets either tint it or use a map.)
            const auto ec = mat->getEmissiveColor();
            const auto emissiveTex = loadTex(mat->getTexture(ofbx::Texture::EMISSIVE), baseDir, texLoader);
            const bool ecBlack = ec.r <= 0.f && ec.g <= 0.f && ec.b <= 0.f;
            const bool ecDefaultWhite = ec.r == 1.f && ec.g == 1.f && ec.b == 1.f;
            if (emissiveTex || (!ecBlack && !ecDefaultWhite)) {
                if (emissiveTex) {
                    m.emissiveMap = emissiveTex;
                    // A map modulates against the emissive colour; fall back to
                    // white when the colour was left unset so the map stays visible.
                    if (ecBlack) m.emissive.setHex(0xffffff);
                    else         m.emissive.setRGB(ec.r, ec.g, ec.b);
                } else {
                    m.emissive.setRGB(ec.r, ec.g, ec.b);
                }
                m.emissiveIntensity = static_cast<float>(mat->getEmissiveFactor()) * emissiveScale;
            }
        }

        std::shared_ptr<Material> buildMaterial(
                const ofbx::Material* mat,
                const std::filesystem::path& baseDir,
                TextureLoader& texLoader,
                FBXLoader::MaterialMode materialMode,
                float emissiveScale) {
            if (!mat) return MeshStandardMaterial::create();

            // Decide whether the SPECULAR slot is an ORM-packed PBR texture or a
            // traditional specular map. The loader flag overrides the filename guess.
            const ofbx::Texture* specSlot = mat->getTexture(ofbx::Texture::SPECULAR);
            const auto specPath = resolveTexturePath(specSlot, baseDir);
            const bool hasSpecTex = !specPath.empty() && isSupportedImageFormat(specPath);
            bool isPBR;
            switch (materialMode) {
                case FBXLoader::MaterialMode::Phong:
                    isPBR = false;
                    break;
                case FBXLoader::MaterialMode::PBR:
                    isPBR = hasSpecTex;
                    break;
                case FBXLoader::MaterialMode::Auto:
                default:
                    isPBR = hasSpecTex && looksLikeORM(specPath);
                    break;
            }

            const auto dc = mat->getDiffuseColor();
            const float opacity = static_cast<float>(mat->getOpacity());

            // Resolve diffuse path for glass heuristic.
            const auto diffPath = resolveTexturePath(mat->getTexture(ofbx::Texture::DIFFUSE), baseDir);
            const bool isGlass = looksLikeGlass(mat, diffPath);

            if (isPBR) {
                // ---- PBR / MeshPhysicalMaterial --------------------------------
                auto m = MeshPhysicalMaterial::create();
                m->color.setRGB(dc.r, dc.g, dc.b);
                if (auto tex = loadTex(mat->getTexture(ofbx::Texture::DIFFUSE), baseDir, texLoader)) {
                    m->map = tex;
                    m->color.setHex(0xffffff);
                }
                if (auto tex = texLoader.load(specPath, ColorSpace::Linear)) {
                    tex->wrapS = TextureWrapping::Repeat;
                    tex->wrapT = TextureWrapping::Repeat;
                    tex->needsUpdate();
                    m->roughnessMap = tex;
                    m->metalnessMap = tex;
                    m->roughness    = 1.0f;  // texture drives; keep metalness default 0
                }
                applyCommon(*m, mat, baseDir, texLoader, emissiveScale);
                if (isGlass) {
                    m->transmission = opacity < 0.99f ? std::max(0.01f, 1.0f - opacity) : 1.0f;
                    m->setIor(1.5f);
                    m->side = Side::Double;
                }
                return m;
            } else if (isGlass) {
                // ---- Glass (no ORM) / MeshPhysicalMaterial ---------------------
                auto m = MeshPhysicalMaterial::create();
                m->color.setRGB(dc.r, dc.g, dc.b);
                if (auto tex = loadTex(mat->getTexture(ofbx::Texture::DIFFUSE), baseDir, texLoader)) {
                    m->map = tex;
                    m->color.setHex(0xffffff);
                }
                m->transmission = opacity < 0.99f ? std::max(0.01f, 1.0f - opacity) : 1.0f;
                m->setIor(1.5f);
                m->side = Side::Double;
                applyCommon(*m, mat, baseDir, texLoader, emissiveScale);
                return m;
            } else {
                // ---- Phong / MeshPhongMaterial ----------------------------------
                auto m = MeshPhongMaterial::create();
                m->color.setRGB(dc.r, dc.g, dc.b);
                if (auto tex = loadTex(mat->getTexture(ofbx::Texture::DIFFUSE), baseDir, texLoader)) {
                    m->map = tex;
                    m->color.setHex(0xffffff);
                }
                // Specular color + shininess from FBX material properties.
                const auto sc = mat->getSpecularColor();
                m->specular.setRGB(sc.r, sc.g, sc.b);
                const double shin = mat->getShininess();
                if (shin > 0.0) m->shininess = static_cast<float>(shin);
                if (auto tex = loadTex(specSlot, baseDir, texLoader, ColorSpace::Linear))
                    m->specularMap = tex;
                applyCommon(*m, mat, baseDir, texLoader, emissiveScale);
                if (opacity < 0.99f) {
                    m->opacity     = std::max(0.01f, opacity);
                    m->transparent = true;
                    m->side        = Side::Double;
                }
                return m;
            }
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
                auto material = buildMaterial(mat, baseDir, pimpl_->texLoader, materialMode, emissiveScale);

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
                    auto material = buildMaterial(mat, baseDir, pimpl_->texLoader, materialMode, emissiveScale);
                    meshGroup->add(Mesh::create(geometry, material));
                }
                root->add(meshGroup);
            }
        }

        // -----------------------------------------------------------------------
        // Lights
        // -----------------------------------------------------------------------
        const int lightCount = scene->getLightCount();
        for (int li = 0; li < lightCount; ++li) {
            const ofbx::Light* fbxLight = scene->getLight(li);
            if (!fbxLight) continue;

            const auto fc = fbxLight->getColor();
            const Color color(fc.r, fc.g, fc.b);
            // FBX intensity is in percent (0-100+). Normalize to 0-1 range.
            const float intensity = static_cast<float>(fbxLight->getIntensity()) / 100.0f;

            std::shared_ptr<Object3D> lightNode;

            switch (fbxLight->getLightType()) {
                case ofbx::Light::LightType::POINT: {
                    auto light = PointLight::create(color, intensity);
                    lightNode = light;
                    break;
                }
                case ofbx::Light::LightType::DIRECTIONAL: {
                    auto light = DirectionalLight::create(color, intensity);
                    lightNode = light;
                    break;
                }
                case ofbx::Light::LightType::SPOT: {
                    const float outerAngle = static_cast<float>(fbxLight->getOuterAngle())
                                           * math::DEG2RAD;
                    const float innerAngle = static_cast<float>(fbxLight->getInnerAngle())
                                           * math::DEG2RAD;
                    // penumbra = 1 - (inner/outer), clamped to [0,1]
                    const float penumbra = (outerAngle > 0.0f)
                            ? std::max(0.0f, std::min(1.0f, 1.0f - innerAngle / outerAngle))
                            : 0.0f;
                    auto light = SpotLight::create(color, intensity, 0.0f, outerAngle, penumbra);
                    lightNode = light;
                    break;
                }
                default:
                    break;
            }

            if (lightNode) {
                lightNode->name = fbxLight->name;
                lightNode->applyMatrix4(toMatrix4(fbxLight->getGlobalTransform()));
                root->add(lightNode);
            }
        }

        scene->destroy();
        return root;
    }

}// namespace threepp
