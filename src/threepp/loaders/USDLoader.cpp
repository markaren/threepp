#include "threepp/loaders/USDLoader.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include "tinyusdz.hh"
#include "composition.hh"
#include "asset-resolution.hh"
#include "stage.hh"
#include "tydra/render-data.hh"
#include "tydra/scene-access.hh"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    namespace {

        // -----------------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------------

        // ----------------------------------------------------------------------
        // Strip list-edit qualifiers that tinyusdz's CompositeReferences /
        // CompositePayload do not support (Delete, Add, Order).
        // In a single-layer flat composition those arcs are no-ops anyway:
        // "delete references" removes items from a weaker opinion that simply
        // doesn't exist here.  Stripping them lets composition proceed without
        // bailing out on the first such prim.
        // ----------------------------------------------------------------------

        void stripUnsupportedArcs(tinyusdz::PrimSpec& ps) {
            using tinyusdz::ListEditQual;
            auto isUnsupported = [](ListEditQual q) {
                return q == ListEditQual::Delete ||
                       q == ListEditQual::Add    ||
                       q == ListEditQual::Order;
            };
            if (ps.metas().references && isUnsupported(ps.metas().references->first))
                ps.metas().references.reset();
            if (ps.metas().payload && isUnsupported(ps.metas().payload->first))
                ps.metas().payload.reset();
            if (ps.metas().inherits && isUnsupported(ps.metas().inherits->first))
                ps.metas().inherits.reset();
            for (auto& child : ps.children())
                stripUnsupportedArcs(child);
        }

        void stripUnsupportedArcsInLayer(tinyusdz::Layer& layer) {
            for (auto& [name, ps] : layer.primspecs())
                stripUnsupportedArcs(ps);
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

        // -----------------------------------------------------------------------
        // Geometry
        // -----------------------------------------------------------------------

        std::shared_ptr<BufferGeometry> geometryFromRenderMesh(
                const tinyusdz::tydra::RenderMesh& rm) {

            if (rm.points.empty()) return nullptr;

            // Prefer triangulated index buffer; fall back to raw (should not happen
            // with triangulate=true but be defensive).
            const std::vector<uint32_t>& idxBuf = rm.is_triangulated()
                ? rm.triangulatedFaceVertexIndices
                : rm.usdFaceVertexIndices;
            if (idxBuf.empty()) return nullptr;

            // Positions — vec3 = std::array<float,3>
            std::vector<float> pos;
            pos.reserve(rm.points.size() * 3);
            for (const auto& p : rm.points) {
                pos.push_back(p[0]);
                pos.push_back(p[1]);
                pos.push_back(p[2]);
            }

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position",
                    FloatBufferAttribute::create(std::move(pos), 3));

            // Copy indices (uint32 → unsigned int, same size on all modern targets)
            std::vector<unsigned int> indices(idxBuf.begin(), idxBuf.end());
            geometry->setIndex(indices);

            // Normals
            const auto& nattr = rm.normals;
            const size_t nVerts = nattr.vertex_count();
            if (nVerts > 0 &&
                nattr.format == tinyusdz::tydra::VertexAttributeFormat::Vec3) {
                const float* nptr =
                    reinterpret_cast<const float*>(nattr.data.data());
                geometry->setAttribute("normal",
                        FloatBufferAttribute::create(
                            std::vector<float>(nptr, nptr + nVerts * 3), 3));
            } else {
                geometry->computeVertexNormals();
            }

            // UV (slot 0)
            if (!rm.texcoords.empty()) {
                const auto& tcattr = rm.texcoords.begin()->second;
                const size_t nUVs = tcattr.vertex_count();
                if (nUVs > 0 &&
                    tcattr.format == tinyusdz::tydra::VertexAttributeFormat::Vec2) {
                    const float* uvptr =
                        reinterpret_cast<const float*>(tcattr.data.data());
                    geometry->setAttribute("uv",
                            FloatBufferAttribute::create(
                                std::vector<float>(uvptr, uvptr + nUVs * 2), 2));
                }
            }

            return geometry;
        }

        // -----------------------------------------------------------------------
        // Textures — build a cache indexed in parallel with RenderScene::textures
        // -----------------------------------------------------------------------

        std::vector<std::shared_ptr<Texture>> buildTextureCache(
                const tinyusdz::tydra::RenderScene& scene,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver,
                TextureLoader& texLoader) {

            std::vector<std::shared_ptr<Texture>> cache(scene.textures.size());

            for (size_t ti = 0; ti < scene.textures.size(); ++ti) {
                const auto& uvtex = scene.textures[ti];
                if (uvtex.texture_image_id < 0 ||
                    uvtex.texture_image_id >= static_cast<int64_t>(scene.images.size()))
                    continue;

                const auto& img = scene.images[uvtex.texture_image_id];
                if (img.asset_identifier.empty()) continue;

                // asset_identifier is the raw USD path string (from
                // assetPath.GetAssetPath()).  Normalise backslashes first.
                std::string rawId = img.asset_identifier;
                std::replace(rawId.begin(), rawId.end(), '\\', '/');

                // Strip leading "./" for filesystem join
                std::string rel = rawId;
                if (rel.size() >= 2 && rel[0] == '.' && rel[1] == '/') rel = rel.substr(2);

                // Try candidates in priority order:
                //  1. baseDir-relative (handles the common ./textures/foo case)
                //  2. resolver (knows referenced-file sub-directories)
                //  3. raw path as-is (already absolute)
                std::filesystem::path texPath;
                auto candidate = baseDir / rel;
                if (std::filesystem::exists(candidate)) {
                    texPath = candidate;
                } else {
                    const std::string resolved = resolver.resolve(rawId);
                    if (!resolved.empty() && std::filesystem::exists(resolved)) {
                        texPath = resolved;
                    } else {
                        texPath = rawId;
                    }
                }
                if (!std::filesystem::exists(texPath))
                    continue;

                const bool flipY = true;
                cache[ti] = texLoader.load(texPath, flipY);
            }

            return cache;
        }

        // -----------------------------------------------------------------------
        // MDL shader fallback — NVIDIA Omniverse materials use MDL (OmniPBR,
        // OmniGlass, …) rather than UsdPreviewSurface, so ConvertToRenderScene
        // produces a default (grey) material.  We fall back by reading
        // inputs:diffuse_texture / inputs:normalmap_texture / inputs:ORM_texture
        // directly from the Shader child PrimSpec in the composed Layer.
        // PrimSpecs from referenced files have their CWP set by
        // PropagateAssetResolverState during CompositeReferences, so we can
        // resolve "./Textures/foo.png" relative to the correct source directory.
        // -----------------------------------------------------------------------

        // Replace <UDIM> token with the first tile index so we get a usable path.
        std::string resolveUDIM(const std::string& s) {
            std::string out = s;
            const std::string tok = "<UDIM>";
            auto pos = out.find(tok);
            if (pos != std::string::npos) out.replace(pos, tok.size(), "1001");
            return out;
        }

        // Given a raw asset path string from a Shader prop, resolve it to an
        // existing file path.  'cwp' is the directory of the USD file that
        // defined the prim (from PrimSpec::get_current_working_path()).
        std::filesystem::path resolveAssetPath(
                const std::string& rawIn,
                const std::filesystem::path& cwp,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver) {

            std::string raw = resolveUDIM(rawIn);
            // Normalise separators
            std::replace(raw.begin(), raw.end(), '\\', '/');
            // Strip leading "./"
            std::string rel = raw;
            if (rel.size() >= 2 && rel[0] == '.' && rel[1] == '/') rel = rel.substr(2);

            // Priority: cwp (source file dir) > baseDir (root stage dir) > resolver
            for (const auto& base : {cwp, baseDir}) {
                auto cand = base / rel;
                if (std::filesystem::exists(cand)) return cand;
            }
            const std::string resolved = resolver.resolve(raw);
            if (!resolved.empty() && std::filesystem::exists(resolved))
                return std::filesystem::path(resolved);
            return {};  // not found
        }

        // Try to build a textured material from an MDL Shader PrimSpec.
        // Returns nullptr if the PrimSpec is not a recognisable MDL shader.
        std::shared_ptr<MeshStandardMaterial> materialFromMDLPrimSpec(
                const tinyusdz::PrimSpec& shaderSpec,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver,
                TextureLoader& texLoader) {

            // Only handle MDL shaders (sourceAsset implementation, or no info:id)
            auto infoIdIt = shaderSpec.props().find("info:id");
            if (infoIdIt != shaderSpec.props().end()) {
                // Has an info:id → if it's UsdPreviewSurface, ConvertToRenderScene
                // already handled it; skip the MDL fallback.
                return nullptr;
            }

            const std::filesystem::path cwp =
                shaderSpec.get_current_working_path().empty()
                    ? baseDir
                    : std::filesystem::path(shaderSpec.get_current_working_path());

            auto mat = MeshStandardMaterial::create();
            bool anyTex = false;

            auto tryLoad = [&](const std::string& propName, bool flipY = true)
                    -> std::shared_ptr<Texture> {
                auto it = shaderSpec.props().find(propName);
                if (it == shaderSpec.props().end()) return nullptr;
                const auto& prop = it->second;
                if (!prop.is_attribute()) return nullptr;
                auto pv = prop.get_attribute().get_value<tinyusdz::value::AssetPath>();
                if (!pv) return nullptr;
                const std::string raw = pv.value().GetAssetPath();
                if (raw.empty()) return nullptr;
                auto texPath = resolveAssetPath(raw, cwp, baseDir, resolver);
                if (texPath.empty()) return nullptr;
                return texLoader.load(texPath, flipY);
            };

            // Try a list of candidate names in order; return the first hit.
            auto tryFirst = [&](std::initializer_list<const char*> names, bool flipY = true)
                    -> std::shared_ptr<Texture> {
                for (auto* n : names) {
                    if (auto t = tryLoad(n, flipY)) return t;
                }
                return nullptr;
            };

            // Albedo / diffuse
            // Covers: snake_case (OmniPBR style) + PascalCase (OmniUe4Base /
            // Omniverse Unreal bridge, used by Lightwheel exports).
            if (auto t = tryFirst({"inputs:diffuse_texture",
                                   "inputs:albedo_texture",
                                   "inputs:Diffuse_Texture",
                                   "inputs:Albedo_Texture",
                                   "inputs:BaseColor_Texture"})) {
                mat->map = t; anyTex = true;
            }
            // Normal map (linear, no flip needed for tangent-space normals)
            if (auto t = tryFirst({"inputs:normalmap_texture",
                                   "inputs:normal_texture",
                                   "inputs:Normal_Texture",
                                   "inputs:NormalMap_Texture",
                                   "inputs:Bump_Texture"}, false)) {
                t->colorSpace = ColorSpace::Linear;
                mat->normalMap = t; anyTex = true;
            }
            // Roughness — try ORM-packed first, then dedicated roughness map.
            if (auto t = tryLoad("inputs:ORM_texture", false)) {
                t->colorSpace = ColorSpace::Linear;
                mat->aoMap         = t;
                mat->roughnessMap  = t;
                mat->metalnessMap  = t;
                anyTex = true;
            } else {
                if (auto r = tryFirst({"inputs:roughness_texture",
                                       "inputs:Roughness_Texture"}, false)) {
                    r->colorSpace = ColorSpace::Linear;
                    mat->roughnessMap = r; anyTex = true;
                }
                if (auto m = tryFirst({"inputs:metallic_texture",
                                       "inputs:Metallic_Texture"}, false)) {
                    m->colorSpace = ColorSpace::Linear;
                    mat->metalnessMap = m; anyTex = true;
                }
            }
            // Emissive
            if (auto t = tryFirst({"inputs:emissive_texture",
                                   "inputs:Emissive_Texture",
                                   "inputs:Emission_Texture"})) {
                mat->emissiveMap = t; anyTex = true;
            }

            // Constant-color fallbacks (used when there's no texture but we
            // still want a non-grey material).
            auto tryFloatTuple = [&](const std::string& propName)
                    -> std::optional<std::array<float, 3>> {
                auto it = shaderSpec.props().find(propName);
                if (it == shaderSpec.props().end()) return std::nullopt;
                const auto& prop = it->second;
                if (!prop.is_attribute()) return std::nullopt;
                const auto& attr = prop.get_attribute();
                if (auto pv = attr.get_value<tinyusdz::value::color3f>()) {
                    return std::array<float, 3>{pv->r, pv->g, pv->b};
                }
                if (auto pv = attr.get_value<tinyusdz::value::float3>()) {
                    return std::array<float, 3>{(*pv)[0], (*pv)[1], (*pv)[2]};
                }
                return std::nullopt;
            };
            auto tryFloat = [&](const std::string& propName) -> std::optional<float> {
                auto it = shaderSpec.props().find(propName);
                if (it == shaderSpec.props().end()) return std::nullopt;
                const auto& prop = it->second;
                if (!prop.is_attribute()) return std::nullopt;
                const auto& attr = prop.get_attribute();
                if (auto pv = attr.get_value<float>()) return *pv;
                if (auto pv = attr.get_value<double>()) return static_cast<float>(*pv);
                return std::nullopt;
            };

            if (!mat->map) {
                if (auto c = tryFloatTuple("inputs:Diffuse_Color")) {
                    mat->color.setRGB((*c)[0], (*c)[1], (*c)[2]);
                    anyTex = true;  // count constant color as "real"
                } else if (auto c2 = tryFloatTuple("inputs:diffuse_color_constant")) {
                    mat->color.setRGB((*c2)[0], (*c2)[1], (*c2)[2]);
                    anyTex = true;
                }
            }
            if (!mat->roughnessMap) {
                if (auto v = tryFloat("inputs:Roughness_Color")) {
                    mat->roughness = *v;
                } else if (auto v2 = tryFloat("inputs:reflection_roughness_constant")) {
                    mat->roughness = *v2;
                }
            }
            if (!mat->metalnessMap) {
                if (auto v = tryFloat("inputs:Metallic_Color")) {
                    mat->metalness = *v;
                } else if (auto v2 = tryFloat("inputs:metallic_constant")) {
                    mat->metalness = *v2;
                }
            }

            return anyTex ? mat : nullptr;
        }

        // Try to locate the material at `target`. If it's not directly
        // resolvable in `layer`, fall back to searching for a child with the
        // same leaf name inside any `Looks` scope along the ancestor chain of
        // `meshAbsPath`. This handles Lightwheel/Omniverse scenes whose
        // `material:binding` rels point at scene-level paths that no longer
        // exist after composition (the actual material was authored inside a
        // referenced prop's local /Looks scope).
        std::string resolveMaterialPath(const tinyusdz::Layer& layer,
                                        const std::string& target,
                                        const std::string& meshAbsPath) {
            if (target.empty()) return {};
            const tinyusdz::PrimSpec* matSpec = nullptr;
            std::string err;
            if (layer.find_primspec_at(tinyusdz::Path(target, ""), &matSpec, &err) && matSpec) {
                return target;  // direct hit
            }
            // Fallback: leaf-name search within ancestor /Looks scopes.
            const auto leafSep = target.rfind('/');
            if (leafSep == std::string::npos) return {};
            const std::string leaf = target.substr(leafSep + 1);
            if (leaf.empty()) return {};

            std::string anc = meshAbsPath;
            while (!anc.empty()) {
                const tinyusdz::PrimSpec* aspec = nullptr;
                std::string ae;
                if (layer.find_primspec_at(tinyusdz::Path(anc, ""), &aspec, &ae) && aspec) {
                    for (const auto& child : aspec->children()) {
                        if (child.name() == "Looks") {
                            for (const auto& mat : child.children()) {
                                if (mat.name() == leaf) {
                                    return anc + "/Looks/" + leaf;
                                }
                            }
                        }
                    }
                }
                auto sp = anc.rfind('/');
                if (sp == 0 || sp == std::string::npos) break;
                anc = anc.substr(0, sp);
            }
            return {};
        }

        // Given a prim abs_path, walk up the ancestor chain in the composed Layer
        // to find the nearest 'material:binding' relationship.
        // Returns the bound material's prim path string, or empty if not found.
        // This is the fallback for composite/instanceable scenes where
        // ConvertToRenderScene's GetBoundMaterial fails because instanceable
        // expanded paths don't exist in the Stage tree.
        std::string findMaterialBindingInLayer(
                const tinyusdz::Layer& layer,
                const std::string& primAbsPath) {

            std::string path = primAbsPath;
            while (!path.empty()) {
                const tinyusdz::PrimSpec* spec = nullptr;
                std::string err;
                if (layer.find_primspec_at(tinyusdz::Path(path, ""), &spec, &err) && spec) {
                    auto it = spec->props().find("material:binding");
                    if (it != spec->props().end() && it->second.is_relationship()) {
                        const auto& rel = it->second.get_relationship();
                        std::string raw;
                        if (rel.is_path()) {
                            raw = rel.targetPath.prim_part();
                        } else if (rel.is_pathvector() && !rel.targetPathVector.empty()) {
                            raw = rel.targetPathVector[0].prim_part();
                        }
                        if (auto resolved = resolveMaterialPath(layer, raw, primAbsPath); !resolved.empty()) {
                            return resolved;
                        }
                    }
                }
                // Strip the last path component to walk up
                auto slashPos = path.rfind('/');
                if (slashPos == 0 || slashPos == std::string::npos) break;
                path = path.substr(0, slashPos);
            }
            return {};
        }

        // Build a material from a UsdPreviewSurface-style Material prim. The
        // standard Lightwheel/Omniverse-exported pattern places sibling Shader
        // nodes named `<MatName>DiffuseColorTex`, `<MatName>NormalTex`, etc.
        // alongside a `<MatName>PreviewSurface` shader. Each tex shader holds
        // an `inputs:file` asset path. We classify by name suffix and build a
        // MeshStandardMaterial.
        std::shared_ptr<MeshStandardMaterial> materialFromUsdPreviewLayer(
                const tinyusdz::PrimSpec& matSpec,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver,
                TextureLoader& texLoader) {

            auto extractFile = [&](const tinyusdz::PrimSpec& shader) -> std::filesystem::path {
                auto it = shader.props().find("inputs:file");
                if (it == shader.props().end()) return {};
                if (!it->second.is_attribute()) return {};
                auto pv = it->second.get_attribute().get_value<tinyusdz::value::AssetPath>();
                if (!pv) return {};
                const std::string raw = pv.value().GetAssetPath();
                if (raw.empty()) return {};
                std::filesystem::path cwp = shader.get_current_working_path().empty()
                    ? baseDir
                    : std::filesystem::path(shader.get_current_working_path());
                return resolveAssetPath(raw, cwp, baseDir, resolver);
            };

            auto containsCI = [](const std::string& s, const std::string& sub) {
                if (sub.size() > s.size()) return false;
                for (size_t i = 0; i + sub.size() <= s.size(); ++i) {
                    bool match = true;
                    for (size_t j = 0; j < sub.size(); ++j) {
                        if (std::tolower(static_cast<unsigned char>(s[i + j])) !=
                            std::tolower(static_cast<unsigned char>(sub[j]))) {
                            match = false; break;
                        }
                    }
                    if (match) return true;
                }
                return false;
            };

            auto mat = MeshStandardMaterial::create();
            bool anyTex = false;

            for (const auto& shader : matSpec.children()) {
                const std::string& sname = shader.name();
                // Skip the PreviewSurface root and the UV reader; we only want
                // the sibling texture-reader shaders.
                if (containsCI(sname, "PreviewSurface")) continue;
                if (sname == "PrimST" || sname == "UnrealShader") continue;

                auto path = extractFile(shader);
                if (path.empty() || !std::filesystem::exists(path)) continue;

                // Color-space defaults: textures in normal/roughness/metallic
                // channels are linear; baseColor + emissive are sRGB.
                const bool isLinear =
                    containsCI(sname, "Normal") ||
                    containsCI(sname, "Roughness") ||
                    containsCI(sname, "Metallic") ||
                    containsCI(sname, "Specular") ||
                    containsCI(sname, "OpacityMask") ||
                    containsCI(sname, "ORM");

                if (containsCI(sname, "DiffuseColorTex") ||
                    containsCI(sname, "BaseColorTex") ||
                    containsCI(sname, "AlbedoTex")) {
                    if (!mat->map) {
                        mat->map = texLoader.load(path, true);
                        anyTex = true;
                    }
                } else if (containsCI(sname, "NormalTex")) {
                    if (!mat->normalMap) {
                        auto t = texLoader.load(path, false);
                        if (t) t->colorSpace = ColorSpace::Linear;
                        mat->normalMap = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "RoughnessTex")) {
                    if (!mat->roughnessMap) {
                        auto t = texLoader.load(path, false);
                        if (t) t->colorSpace = ColorSpace::Linear;
                        mat->roughnessMap = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "MetallicTex")) {
                    if (!mat->metalnessMap) {
                        auto t = texLoader.load(path, false);
                        if (t) t->colorSpace = ColorSpace::Linear;
                        mat->metalnessMap = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "EmissiveColorTex") ||
                           containsCI(sname, "EmissionTex")) {
                    if (!mat->emissiveMap) {
                        mat->emissiveMap = texLoader.load(path, true);
                        anyTex = true;
                    }
                }
                // OpacityMask, Specular: skipped (three.js MeshStandardMaterial
                // doesn't have a direct equivalent for either).
                (void)isLinear;
            }

            return anyTex ? mat : nullptr;
        }

        // Walk the composed layer to find the MDL Shader child of a material prim
        // at 'matAbsPath', then build a material from its MDL inputs.
        // Falls back to UsdPreviewSurface-style scanning when the MDL path
        // produces nothing.
        std::shared_ptr<MeshStandardMaterial> materialFromMDLLayer(
                const tinyusdz::Layer& layer,
                const std::string& matAbsPath,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver,
                TextureLoader& texLoader) {

            const tinyusdz::PrimSpec* matSpec = nullptr;
            std::string err;
            if (!layer.find_primspec_at(tinyusdz::Path(matAbsPath, ""), &matSpec, &err) || !matSpec)
                return nullptr;

            for (const auto& child : matSpec->children()) {
                if (auto mat = materialFromMDLPrimSpec(child, baseDir, resolver, texLoader))
                    return mat;
            }
            // UsdPreviewSurface fallback: scan sibling tex shaders inside the
            // Material directly. Lightwheel exports place these alongside the
            // `*PreviewSurface` shader rather than under a single MDL shader.
            return materialFromUsdPreviewLayer(*matSpec, baseDir, resolver, texLoader);
        }

        // -----------------------------------------------------------------------
        // Material
        // -----------------------------------------------------------------------

        std::shared_ptr<MeshStandardMaterial> materialFromRenderMaterial(
                const tinyusdz::tydra::RenderScene& scene,
                int materialId,
                const std::vector<std::shared_ptr<Texture>>& texCache) {

            auto mat = MeshStandardMaterial::create();
            if (materialId < 0 || materialId >= static_cast<int>(scene.materials.size()))
                return mat;

            const auto& shader = scene.materials[materialId].surfaceShader;

            // Helper: return cached texture or nullptr
            auto getTex = [&](int texId) -> std::shared_ptr<Texture> {
                if (texId < 0 || texId >= static_cast<int>(texCache.size())) return nullptr;
                return texCache[texId];
            };

            // diffuseColor / map
            if (shader.diffuseColor.is_texture()) {
                mat->map = getTex(shader.diffuseColor.texture_id);
            } else {
                const auto& c = shader.diffuseColor.value;
                mat->color.setRGB(c[0], c[1], c[2]);
            }

            // roughness
            if (shader.roughness.is_texture()) {
                mat->roughnessMap = getTex(shader.roughness.texture_id);
            } else {
                mat->roughness = shader.roughness.value;
            }

            // metallic
            if (shader.metallic.is_texture()) {
                mat->metalnessMap = getTex(shader.metallic.texture_id);
            } else {
                mat->metalness = shader.metallic.value;
            }

            // emissiveColor
            if (shader.emissiveColor.is_texture()) {
                mat->emissiveMap = getTex(shader.emissiveColor.texture_id);
            } else {
                const auto& e = shader.emissiveColor.value;
                mat->emissive.setRGB(e[0], e[1], e[2]);
            }

            // normal map
            if (shader.normal.is_texture()) {
                auto tex = getTex(shader.normal.texture_id);
                if (tex) {
                    tex->colorSpace = ColorSpace::Linear;
                    mat->normalMap = tex;
                }
            }

            // occlusion → aoMap
            if (shader.occlusion.is_texture()) {
                mat->aoMap = getTex(shader.occlusion.texture_id);
            }

            // opacity
            if (!shader.opacity.is_texture()) {
                const float op = shader.opacity.value;
                if (op < 1.0f) {
                    mat->opacity = op;
                    mat->transparent = true;
                }
            }

            return mat;
        }

        // -----------------------------------------------------------------------
        // Node traversal — flattened: each mesh goes straight under root so that
        // global_matrix can be used directly (no re-composition needed).
        // -----------------------------------------------------------------------

        void visitNode(const tinyusdz::tydra::RenderScene& scene,
                       const tinyusdz::tydra::Node& node,
                       Group& root,
                       const std::vector<std::shared_ptr<Texture>>& texCache,
                       const std::unordered_map<int, std::shared_ptr<MeshStandardMaterial>>& mdlOverrides,
                       const std::unordered_map<std::string, std::shared_ptr<MeshStandardMaterial>>& meshPathMdlMap) {

            if (node.nodeType == tinyusdz::tydra::NodeType::Mesh &&
                node.id >= 0 &&
                node.id < static_cast<int>(scene.meshes.size())) {

                const auto& rm = scene.meshes[node.id];
                auto geometry = geometryFromRenderMesh(rm);
                if (geometry) {
                    // Pick the first sub-mesh material; fall back to whole-mesh
                    int matId = rm.material_id;
                    if (!rm.material_subsetMap.empty()) {
                        const int subMatId = rm.material_subsetMap.begin()->second.material_id;
                        if (subMatId >= 0) matId = subMatId;
                    }

                    // Priority:
                    // 1. mdlOverrides by material_id (non-composite MDL files)
                    // 2. meshPathMdlMap by node.abs_path (composite/instanceable scenes)
                    // 3. materialFromRenderMaterial (UsdPreviewSurface or default grey)
                    std::shared_ptr<MeshStandardMaterial> material;
                    auto mdlIt = mdlOverrides.find(matId);
                    if (mdlIt != mdlOverrides.end() && mdlIt->second) {
                        material = mdlIt->second;
                    } else {
                        auto pathIt = meshPathMdlMap.find(node.abs_path);
                        if (pathIt != meshPathMdlMap.end() && pathIt->second) {
                            material = pathIt->second;
                        } else {
                            material = materialFromRenderMaterial(scene, matId, texCache);
                        }
                    }

                    auto mesh = Mesh::create(geometry, material);

                    // Apply world-space transform from RenderScene
                    mesh->matrix->copy(toMatrix4(node.global_matrix));
                    mesh->matrix->decompose(mesh->position, mesh->quaternion, mesh->scale);
                    mesh->matrixAutoUpdate = false;

                    root.add(mesh);
                }
            }

            for (const auto& child : node.children) {
                visitNode(scene, child, root, texCache, mdlOverrides, meshPathMdlMap);
            }
        }

    }// namespace

    // -----------------------------------------------------------------------
    // Impl
    // -----------------------------------------------------------------------

    struct USDLoader::Impl {
        TextureLoader texLoader;
        bool ignoreUpDirection = false;

        std::shared_ptr<Group> load(const std::filesystem::path& path) {
            const std::string pathStr = path.string();
            const std::string ext = [&] {
                std::string e = path.extension().string();
                for (auto& c : e)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                return e;
            }();

            const std::filesystem::path baseDir = path.parent_path();

            // --- Set up asset resolver with the file's directory ---
            tinyusdz::AssetResolutionResolver resolver;
            resolver.set_current_working_path(baseDir.string());
            resolver.set_search_paths({baseDir.string()});

            // --- Load raw Layer (no composition yet) ---
            tinyusdz::Layer layer;
            std::string warn, err;
            bool ok;
            if (ext == ".usdz") {
                // USDZ is a zip; load directly to Stage (self-contained, no external refs)
                tinyusdz::Stage stageDirect;
                ok = tinyusdz::LoadUSDZFromFile(pathStr, &stageDirect, &warn, &err);
                if (!ok) {
                    std::cerr << "[USDLoader] Failed to load '" << pathStr << "': " << err << "\n";
                    return nullptr;
                }
                return buildFromStage(stageDirect, pathStr, baseDir, resolver);
            } else {
                ok = tinyusdz::LoadLayerFromFile(pathStr, &layer, &warn, &err);
            }

            if (!ok) {
                std::cerr << "[USDLoader] Failed to load layer '" << pathStr << "': " << err << "\n";
                return nullptr;
            }

            // --- Composition pipeline (LIVRPS order) ---
            // 1. Sublayers first (strongest opinion)
            {
                tinyusdz::Layer composed;
                std::string cw, ce;
                if (tinyusdz::CompositeSublayers(resolver, layer, &composed, &cw, &ce))
                    layer = std::move(composed);
            }

            // 2. References and payloads — iterate until stable (max 8 passes).
            // Strip unsupported list-edit qualifiers (Delete/Add/Order) once
            // per expansion: the original layer first, then again after each
            // successful composition pass so arcs introduced by referenced files
            // don't trip up subsequent passes.
            // NOTE: do NOT call stripUnsupportedArcsInLayer inside the
            // check_unresolved_references() predicate — the layer may be large
            // and calling it every iteration would stall on big scenes.
            stripUnsupportedArcsInLayer(layer);

            for (int pass = 0; pass < 8; ++pass) {
                bool progressed = false;

                if (layer.check_unresolved_references()) {
                    tinyusdz::Layer composed;
                    std::string cw, ce;
                    if (!tinyusdz::CompositeReferences(resolver, layer, &composed, &cw, &ce))
                        break;
                    layer = std::move(composed);
                    stripUnsupportedArcsInLayer(layer); // strip arcs from newly expanded files
                    progressed = true;
                }

                if (layer.check_unresolved_payload()) {
                    tinyusdz::Layer composed;
                    std::string cw, ce;
                    if (!tinyusdz::CompositePayload(resolver, layer, &composed, &cw, &ce))
                        break;
                    layer = std::move(composed);
                    stripUnsupportedArcsInLayer(layer);
                    progressed = true;
                }

                if (!progressed) break;
            }

            // --- Build Stage from composed Layer ---
            tinyusdz::Stage stage;
            // Copy layer metadata (upAxis, defaultPrim, etc.) before LayerToStage
            // may reset them.
            stage.metas() = layer.metas();
            {
                std::string sw, se;
                if (!tinyusdz::LayerToStage(layer, &stage, &sw, &se)) {
                    std::cerr << "[USDLoader] LayerToStage failed for '" << pathStr << "': " << se << "\n";
                    return nullptr;
                }
            }

            return buildFromStage(stage, pathStr, baseDir, resolver, &layer);
        }

        std::shared_ptr<Group> buildFromStage(const tinyusdz::Stage& stage,
                                               const std::string& pathStr,
                                               const std::filesystem::path& baseDir,
                                               const tinyusdz::AssetResolutionResolver& resolver,
                                               const tinyusdz::Layer* layer = nullptr) {
            // --- Convert to RenderScene (triangulates, builds vertex indices) ---
            tinyusdz::tydra::RenderSceneConverterEnv env(stage);
            env.usd_filename = pathStr;
            // Use the resolver that accumulated all referenced-file search paths
            // during composition so texture/asset paths resolve correctly.
            // NOTE: AssetResolutionResolver copy operator does NOT copy
            // _current_working_path (tinyusdz bug), so restore it explicitly.
            env.asset_resolver = resolver;
            env.asset_resolver.set_current_working_path(baseDir.string());

            // triangulate and build single-indexable vertex buffers
            env.mesh_config.triangulate           = true;
            env.mesh_config.build_vertex_indices  = true;
            env.mesh_config.compute_normals       = true;

            // load_texture_assets = true ensures asset_identifier is always
            // populated with the raw USD path string (assetPath.GetAssetPath()),
            // even if tinyusdz's own decoder fails.  We reload the file ourselves
            // below via buildTextureCache so the decoded buffer is not used.
            env.scene_config.load_texture_assets  = true;

            tinyusdz::tydra::RenderScene renderScene;
            tinyusdz::tydra::RenderSceneConverter converter;

            if (!converter.ConvertToRenderScene(env, &renderScene))
                return nullptr;

            if (renderScene.meshes.empty()) {
                std::cerr << "[USDLoader] No meshes after conversion of '" << pathStr << "'\n";
                return nullptr;
            }


            // --- Textures via RenderScene (UsdPreviewSurface path) ---
            auto texCache = buildTextureCache(renderScene, baseDir, env.asset_resolver, texLoader);

            // --- MDL material overrides (NVIDIA Omniverse OmniPBR etc.) ---
            // ConvertToRenderScene yields default grey materials for MDL shaders.
            // We fall back by reading inputs:diffuse/normal/ORM textures directly
            // from Shader PrimSpecs in the composed Layer.
            //
            // Two cases:
            //  A) renderScene.materials is populated (individual / non-instanceable):
            //     build mdlOverrides[material_id] for materials lacking textures.
            //  B) renderScene.materials is empty (composite / instanceable prims):
            //     GetBoundMaterial in tinyusdz fails because instanceable-expanded
            //     prim paths don't exist in the Stage tree.  Resolve material:binding
            //     directly from the Layer for each mesh node abs_path instead.
            std::unordered_map<int, std::shared_ptr<MeshStandardMaterial>> mdlOverrides;
            std::unordered_map<std::string, std::shared_ptr<MeshStandardMaterial>> meshPathMdlMap;

            if (layer) {
                // Case A — per-material-id overrides
                for (int mi = 0; mi < static_cast<int>(renderScene.materials.size()); ++mi) {
                    const auto& rmat = renderScene.materials[mi];
                    const bool hasTextures =
                        rmat.surfaceShader.diffuseColor.is_texture() ||
                        rmat.surfaceShader.roughness.is_texture()    ||
                        rmat.surfaceShader.metallic.is_texture()     ||
                        rmat.surfaceShader.normal.is_texture();
                    if (hasTextures) continue;
                    if (auto mdlMat = materialFromMDLLayer(
                            *layer, rmat.abs_path, baseDir, env.asset_resolver, texLoader))
                        mdlOverrides[mi] = std::move(mdlMat);
                }

                // Case B — per-mesh-path overrides for instanceable / composite scenes.
                // Run even when renderScene.materials is non-empty: an Omniverse
                // scene may have a couple of materials surface through tinyusdz
                // (e.g. a UsdPreviewSurface fallback on a non-instanced prim) yet
                // leave most of its 100s of meshes pointing at MDL materials inside
                // referenced .usd / instanceable prims, where GetBoundMaterial fails.
                {
                    // Cache unique material paths to avoid re-loading the same textures
                    std::unordered_map<std::string,
                                       std::shared_ptr<MeshStandardMaterial>> matPathCache;

                    std::function<void(const tinyusdz::tydra::Node&)> buildMap =
                        [&](const tinyusdz::tydra::Node& n) {
                            if (n.nodeType == tinyusdz::tydra::NodeType::Mesh &&
                                !n.abs_path.empty()) {
                                const std::string matPath =
                                    findMaterialBindingInLayer(*layer, n.abs_path);
                                if (!matPath.empty()) {
                                    auto cit = matPathCache.find(matPath);
                                    if (cit == matPathCache.end()) {
                                        auto mdlMat = materialFromMDLLayer(
                                            *layer, matPath, baseDir,
                                            env.asset_resolver, texLoader);
                                        cit = matPathCache.emplace(matPath,
                                                                    std::move(mdlMat)).first;
                                    }
                                    if (cit->second)
                                        meshPathMdlMap[n.abs_path] = cit->second;
                                }
                            }
                            for (const auto& c : n.children) buildMap(c);
                        };

                    for (const auto& n : renderScene.nodes) buildMap(n);
                }
            }

            // Brief one-line status: how many meshes resolved to a real material
            // vs. fell through to the default grey fallback.
            int statTextured = 0, statColored = 0, statFallback = 0, statTinyusdzMat = 0;
            std::function<void(const tinyusdz::tydra::Node&)> classify =
                [&](const tinyusdz::tydra::Node& n) {
                    if (n.nodeType == tinyusdz::tydra::NodeType::Mesh &&
                        n.id >= 0 && n.id < (int)renderScene.meshes.size()) {
                        const auto& rm = renderScene.meshes[n.id];
                        int matId = rm.material_id;
                        if (!rm.material_subsetMap.empty()) {
                            const int subId = rm.material_subsetMap.begin()->second.material_id;
                            if (subId >= 0) matId = subId;
                        }
                        std::shared_ptr<MeshStandardMaterial> chosen;
                        if (auto it = mdlOverrides.find(matId); it != mdlOverrides.end()) {
                            chosen = it->second;
                        } else if (auto it = meshPathMdlMap.find(n.abs_path); it != meshPathMdlMap.end()) {
                            chosen = it->second;
                        } else if (matId >= 0 && matId < (int)renderScene.materials.size()) {
                            ++statTinyusdzMat;
                        } else {
                            ++statFallback;
                        }
                        if (chosen && chosen->map) ++statTextured;
                        else if (chosen) ++statColored;
                    }
                    for (const auto& c : n.children) classify(c);
                };
            for (const auto& n : renderScene.nodes) classify(n);

            std::cerr << "[USDLoader] " << pathStr
                      << " — meshes=" << renderScene.meshes.size()
                      << " (textured=" << statTextured
                      << ", color=" << statColored
                      << ", tinyusdz=" << statTinyusdzMat
                      << ", fallback=" << statFallback << ")"
                      << std::endl;

            // --- Root group + Z-up correction ---
            // renderScene.meta.upAxis is never populated by ConvertToRenderScene;
            // read from stage metadata directly.
            // Skipped when this loader is being used by an outer system (URDF/SDF/...)
            // that owns the coordinate frame.
            auto root = Group::create();
            if (!ignoreUpDirection && stage.metas().upAxis.get_value() == tinyusdz::Axis::Z) {
                root->rotation.x = -math::PI / 2.0f;
            }

            // --- Walk node tree ---
            for (const auto& node : renderScene.nodes) {
                visitNode(renderScene, node, *root, texCache, mdlOverrides, meshPathMdlMap);
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

    USDLoader& USDLoader::setIgnoreUpDirection(bool ignore) {
        pimpl_->ignoreUpDirection = ignore;
        return *this;
    }

    std::shared_ptr<Group> USDLoader::load(const std::filesystem::path& path) {
        return pimpl_->load(path);
    }

    USDResult USDLoader::loadFull(const std::filesystem::path& path) {
        return {pimpl_->load(path), {}};
    }

}// namespace threepp
