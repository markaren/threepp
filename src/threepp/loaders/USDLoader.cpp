#include "threepp/loaders/USDLoader.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/Texture.hpp"

#include "tinyusdz.hh"
#include "tydra/scene-access.hh"
#include "tydra/shader-network.hh"
#include "usdGeom.hh"
#include "usdShade.hh"
#include "value-types.hh"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace threepp {

    namespace {

        // -----------------------------------------------------------------------
        // Geometry helpers
        // -----------------------------------------------------------------------

        void triangulateFace(const std::vector<int32_t>& fvi, int faceStart, int count,
                             std::vector<unsigned int>& triIndices) {
            for (int i = 1; i + 1 < count; ++i) {
                triIndices.push_back(static_cast<unsigned int>(fvi[faceStart]));
                triIndices.push_back(static_cast<unsigned int>(fvi[faceStart + i]));
                triIndices.push_back(static_cast<unsigned int>(fvi[faceStart + i + 1]));
            }
        }

        Matrix4 toMatrix4(const tinyusdz::value::matrix4d& mat) {
            // USD row-vector convention → transpose for threepp column-vector
            const auto& r = mat.m;
            return Matrix4().set(
                static_cast<float>(r[0][0]), static_cast<float>(r[1][0]),
                static_cast<float>(r[2][0]), static_cast<float>(r[3][0]),
                static_cast<float>(r[0][1]), static_cast<float>(r[1][1]),
                static_cast<float>(r[2][1]), static_cast<float>(r[3][1]),
                static_cast<float>(r[0][2]), static_cast<float>(r[1][2]),
                static_cast<float>(r[2][2]), static_cast<float>(r[3][2]),
                static_cast<float>(r[0][3]), static_cast<float>(r[1][3]),
                static_cast<float>(r[2][3]), static_cast<float>(r[3][3]));
        }

        std::shared_ptr<BufferGeometry> geometryFromGeomMesh(const tinyusdz::GeomMesh& geomMesh) {
            const auto points = geomMesh.get_points();
            const auto fvc = geomMesh.get_faceVertexCounts();
            const auto fvi = geomMesh.get_faceVertexIndices();
            if (points.empty() || fvc.empty() || fvi.empty()) return nullptr;

            // Triangulate
            std::vector<unsigned int> triIndices;
            triIndices.reserve(fvi.size());
            int faceStart = 0;
            for (int count : fvc) {
                if (count >= 3) triangulateFace(fvi, faceStart, count, triIndices);
                faceStart += count;
            }
            if (triIndices.empty()) return nullptr;

            // Positions
            std::vector<float> positions;
            positions.reserve(points.size() * 3);
            for (const auto& p : points) {
                positions.push_back(p[0]);
                positions.push_back(p[1]);
                positions.push_back(p[2]);
            }

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position",
                    FloatBufferAttribute::create(std::move(positions), 3));
            geometry->setIndex(triIndices);

            // Normals — vertex-interp accepted directly; face-varying falls back to computed
            bool hasNormals = false;
            const auto normals = geomMesh.get_normals();
            if (!normals.empty() &&
                geomMesh.get_normalsInterpolation() == tinyusdz::Interpolation::Vertex) {
                std::vector<float> normData;
                normData.reserve(normals.size() * 3);
                for (const auto& n : normals) {
                    normData.push_back(n[0]);
                    normData.push_back(n[1]);
                    normData.push_back(n[2]);
                }
                geometry->setAttribute("normal",
                        FloatBufferAttribute::create(std::move(normData), 3));
                hasNormals = true;
            }
            if (!hasNormals) geometry->computeVertexNormals();

            // UV: try "st" then "st0"
            for (const char* uvName : {"st", "st0"}) {
                if (!geomMesh.has_primvar(uvName)) continue;
                tinyusdz::GeomPrimvar pv;
                std::string pvErr;
                if (!geomMesh.get_primvar(uvName, &pv, &pvErr)) continue;
                std::vector<tinyusdz::value::texcoord2f> uvs;
                if (pv.flatten_with_indices(&uvs)) {
                    std::vector<float> uvData;
                    uvData.reserve(uvs.size() * 2);
                    for (const auto& uv : uvs) {
                        uvData.push_back(uv[0]);
                        uvData.push_back(uv[1]);
                    }
                    geometry->setAttribute("uv",
                            FloatBufferAttribute::create(std::move(uvData), 2));
                }
                break;
            }

            return geometry;
        }

        // -----------------------------------------------------------------------
        // Material / texture helpers
        // -----------------------------------------------------------------------

        // Resolve the file path from a UsdUVTexture, relative to baseDir.
        std::filesystem::path resolveTexturePath(const tinyusdz::UsdUVTexture& uvtex,
                                                 const std::filesystem::path& baseDir) {
            if (!uvtex.file.authored()) return {};
            auto maybeAnim = uvtex.file.get_value();
            if (!maybeAnim) return {};
            tinyusdz::value::AssetPath assetPath;
            if (!maybeAnim->get_scalar(&assetPath)) return {};

            // Prefer already-resolved path, fall back to relative
            const std::string resolved = assetPath.GetResolvedPath();
            if (!resolved.empty()) return std::filesystem::path(resolved);

            const std::string raw = assetPath.GetAssetPath();
            if (raw.empty()) return {};
            return baseDir / raw;
        }

        // Follow a connected attribute to its UsdUVTexture and load the texture.
        // AttrT must have is_connection() / get_connections().
        template<typename AttrT>
        std::shared_ptr<Texture> loadConnectedTexture(
                const tinyusdz::Stage& stage,
                const AttrT& attr,
                const std::filesystem::path& baseDir,
                TextureLoader& texLoader,
                bool flipY = true) {
            if (!attr.is_connection()) return nullptr;
            const auto& conns = attr.get_connections();
            if (conns.empty()) return nullptr;

            // Connection path e.g. "/Materials/Mat/DiffTex.outputs:rgb"
            // Get only the prim part: "/Materials/Mat/DiffTex"
            tinyusdz::Path texPrimPath(conns[0].prim_part(), "");
            auto primResult = stage.GetPrimAtPath(texPrimPath);
            if (!primResult) return nullptr;

            const tinyusdz::Shader* shader = primResult.value()->as<tinyusdz::Shader>();
            if (!shader) return nullptr;

            const tinyusdz::UsdUVTexture* uvtex = shader->value.as<tinyusdz::UsdUVTexture>();
            if (!uvtex) return nullptr;

            const auto texPath = resolveTexturePath(*uvtex, baseDir);
            if (texPath.empty() || !std::filesystem::exists(texPath)) return nullptr;

            return texLoader.load(texPath, flipY);
        }

        // Build a MeshStandardMaterial from a bound UsdPreviewSurface.
        std::shared_ptr<MeshStandardMaterial> materialFromPreviewSurface(
                const tinyusdz::UsdPreviewSurface& ps,
                const tinyusdz::Stage& stage,
                const std::filesystem::path& baseDir,
                TextureLoader& texLoader) {
            auto mat = MeshStandardMaterial::create();

            // --- diffuseColor / map ---
            if (ps.diffuseColor.is_connection()) {
                auto tex = loadConnectedTexture(stage, ps.diffuseColor, baseDir, texLoader, true);
                if (tex) mat->map = tex;
            } else {
                tinyusdz::value::color3f c{0.18f, 0.18f, 0.18f};
                ps.diffuseColor.get_value().get_scalar(&c);
                mat->color.setRGB(c[0], c[1], c[2]);
            }

            // --- roughness / roughnessMap ---
            if (ps.roughness.is_connection()) {
                auto tex = loadConnectedTexture(stage, ps.roughness, baseDir, texLoader, true);
                if (tex) mat->roughnessMap = tex;
            } else {
                float r = 0.5f;
                ps.roughness.get_value().get_scalar(&r);
                mat->roughness = r;
            }

            // --- metallic / metalnessMap ---
            if (ps.metallic.is_connection()) {
                auto tex = loadConnectedTexture(stage, ps.metallic, baseDir, texLoader, true);
                if (tex) mat->metalnessMap = tex;
            } else {
                float m = 0.0f;
                ps.metallic.get_value().get_scalar(&m);
                mat->metalness = m;
            }

            // --- emissiveColor / emissiveMap ---
            if (ps.emissiveColor.is_connection()) {
                auto tex = loadConnectedTexture(stage, ps.emissiveColor, baseDir, texLoader, true);
                if (tex) mat->emissiveMap = tex;
            } else {
                tinyusdz::value::color3f e{0.f, 0.f, 0.f};
                ps.emissiveColor.get_value().get_scalar(&e);
                mat->emissive.setRGB(e[0], e[1], e[2]);
            }

            // --- normal map (always a texture in UsdPreviewSurface) ---
            if (ps.normal.is_connection()) {
                // Normal maps are linear — don't flip Y (USD tangent space)
                auto tex = loadConnectedTexture(stage, ps.normal, baseDir, texLoader, false);
                if (tex) {
                    tex->encoding = Encoding::Linear;
                    mat->normalMap = tex;
                }
            }

            // --- occlusion → aoMap ---
            if (ps.occlusion.is_connection()) {
                auto tex = loadConnectedTexture(stage, ps.occlusion, baseDir, texLoader, true);
                if (tex) mat->aoMap = tex;
            }

            // --- opacity ---
            if (!ps.opacity.is_connection()) {
                float op = 1.0f;
                ps.opacity.get_value().get_scalar(&op);
                if (op < 1.0f) {
                    mat->opacity = op;
                    mat->transparent = true;
                }
            }

            return mat;
        }

        // Resolve a material for a given mesh prim path.
        std::shared_ptr<MeshStandardMaterial> resolveMaterial(
                const tinyusdz::Stage& stage,
                const std::string& primPathStr,
                const std::filesystem::path& baseDir,
                TextureLoader& texLoader) {
            tinyusdz::Path primPath(primPathStr, "");
            tinyusdz::Path matPath;
            const tinyusdz::Material* material = nullptr;
            std::string err;

            if (!tinyusdz::tydra::GetBoundMaterial(stage, primPath, "", &matPath, &material, &err)
                    || !material) {
                return MeshStandardMaterial::create();
            }

            // Follow surface output connection → shader prim
            if (!material->surface.authored()) return MeshStandardMaterial::create();
            const auto& surfConns = material->surface.get_connections();
            if (surfConns.empty()) return MeshStandardMaterial::create();

            tinyusdz::Path shaderPrimPath(surfConns[0].prim_part(), "");
            auto primResult = stage.GetPrimAtPath(shaderPrimPath);
            if (!primResult) return MeshStandardMaterial::create();

            const tinyusdz::Shader* shader = primResult.value()->as<tinyusdz::Shader>();
            if (!shader) return MeshStandardMaterial::create();

            const tinyusdz::UsdPreviewSurface* ps =
                    shader->value.as<tinyusdz::UsdPreviewSurface>();
            if (!ps) return MeshStandardMaterial::create();

            return materialFromPreviewSurface(*ps, stage, baseDir, texLoader);
        }

    }// namespace

    // -----------------------------------------------------------------------
    // Impl
    // -----------------------------------------------------------------------

    struct USDLoader::Impl {
        TextureLoader texLoader;

        std::shared_ptr<Group> load(const std::filesystem::path& path) {
            const std::string pathStr = path.string();
            const std::string ext = [&] {
                std::string e = path.extension().string();
                for (auto& c : e)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                return e;
            }();

            tinyusdz::Stage stage;
            std::string warn, err;
            bool ok = (ext == ".usdz")
                ? tinyusdz::LoadUSDZFromFile(pathStr, &stage, &warn, &err)
                : tinyusdz::LoadUSDFromFile(pathStr, &stage, &warn, &err);

            if (!warn.empty()) std::cerr << "[USDLoader] Warning: " << warn << "\n";
            if (!ok) {
                std::cerr << "[USDLoader] Failed to load '" << pathStr << "': " << err << "\n";
                return nullptr;
            }

            const std::filesystem::path baseDir = path.parent_path();

            tinyusdz::tydra::PathPrimMap<tinyusdz::GeomMesh> meshMap;
            tinyusdz::tydra::ListPrims(stage, meshMap);
            if (meshMap.empty()) {
                std::cerr << "[USDLoader] No GeomMesh prims found in '" << pathStr << "'\n";
                return nullptr;
            }

            // Z-up → Y-up root correction
            auto root = Group::create();
            if (stage.metas().upAxis.get_value() == tinyusdz::Axis::Z) {
                root->rotation.x = -math::PI / 2.0f;
            }

            for (const auto& [primPath, geomMeshPtr] : meshMap) {
                if (!geomMeshPtr) continue;
                auto geometry = geometryFromGeomMesh(*geomMeshPtr);
                if (!geometry) continue;

                auto material = resolveMaterial(stage, primPath, baseDir, texLoader);
                auto mesh = Mesh::create(geometry, material);

                if (auto xform = geomMeshPtr->GetLocalMatrix()) {
                    mesh->matrix->copy(toMatrix4(*xform));
                    mesh->matrix->decompose(mesh->position, mesh->quaternion, mesh->scale);
                    mesh->matrixAutoUpdate = false;
                }

                root->add(mesh);
            }

            if (root->children.empty()) {
                std::cerr << "[USDLoader] No usable geometry in '" << pathStr << "'\n";
                return nullptr;
            }

            return root;
        }
    };

    USDLoader::USDLoader() : pimpl_(std::make_unique<Impl>()) {}
    USDLoader::~USDLoader() = default;

    std::shared_ptr<Group> USDLoader::load(const std::filesystem::path& path) {
        return pimpl_->load(path);
    }

}// namespace threepp
