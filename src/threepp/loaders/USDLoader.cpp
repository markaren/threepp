#include "threepp/loaders/USDLoader.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/loaders/EXRLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/textures/Texture.hpp"

#include "next/layer/asset-anchor.hh"
#include "next/layer/prim-spec.hh"
#include "next/lightusd-next.hh"
#include "next/resolver/asset-resolver.hh"
#include "next/schema/usd-shade.hh"
#include "tydra/next/render-converter.hh"
#include "tydra/next/scene-access.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp {

    namespace {

        namespace lnext = lightusd::next;
        namespace tnext = lightusd::tydra::next;

        // -----------------------------------------------------------------------
        // HTTP/HTTPS asset resolution.
        //
        // Real-world Omniverse / NVIDIA Showcases scenes embed payload paths
        // pointing into the public Nucleus mirror
        // (`https://omniverse-content-production.s3.us-west-2.amazonaws.com/...`).
        // The http and https schemes resolve by downloading the asset once via
        // `curl` (built into Win 10+, macOS, all current Linux distros) into a
        // temp on-disk cache and answering with that local path; LightUSD then
        // reads it like any other file.
        // -----------------------------------------------------------------------

        bool isHttpUrl(const std::string& s) {
            return s.compare(0, 7, "http://") == 0 ||
                   s.compare(0, 8, "https://") == 0;
        }

        // The URL comes from the USD file and goes into a shell command line.
        bool isShellSafeUrl(const std::string& url) {
            return url.find_first_of("\"'`$;&|<>\\ \t\r\n") == std::string::npos;
        }

        std::filesystem::path httpCacheDir() {
            std::filesystem::path p =
                std::filesystem::temp_directory_path() / "threepp_usd_cache";
            std::error_code ec;
            std::filesystem::create_directories(p, ec);
            return p;
        }

        // Hash a URL into a stable cache filename. Preserves the URL's
        // extension so LightUSD / our loader can format-detect from it.
        std::string urlToCacheName(const std::string& url) {
            std::hash<std::string> h;
            std::ostringstream os;
            os << std::hex << h(url);
            // strip query string before extracting extension
            std::string clean = url;
            if (auto q = clean.find('?'); q != std::string::npos) clean = clean.substr(0, q);
            auto dot = clean.rfind('.');
            auto slash = clean.rfind('/');
            if (dot != std::string::npos &&
                (slash == std::string::npos || dot > slash) &&
                clean.size() - dot <= 6 /* sanity */) {
                os << clean.substr(dot);
            }
            return os.str();
        }

        bool fetchHttp(const std::string& url,
                       const std::filesystem::path& dest,
                       std::string& err) {
            if (!isShellSafeUrl(url)) {
                err = "refusing URL with shell metacharacters: " + url;
                return false;
            }
            // -L follow redirects, -s silent, -f fail on HTTP errors,
            // --max-time bounds the download. curl on Win/macOS/Linux all
            // accept the same flags.
            std::string cmd = "curl -L -s -f --max-time 60 -o \"" +
                              dest.string() + "\" \"" + url + "\"";
#ifdef _WIN32
            // Suppress stderr from spawning a console window.
            cmd = cmd + " 2>nul";
#else
            cmd = cmd + " 2>/dev/null";
#endif
            const int rc = std::system(cmd.c_str());
            if (rc != 0) {
                err = "curl exit " + std::to_string(rc) + " for " + url;
                std::error_code ec;
                std::filesystem::remove(dest, ec);// don't leave a partial file in cache
                return false;
            }
            std::error_code ec;
            if (!std::filesystem::exists(dest, ec) ||
                std::filesystem::file_size(dest, ec) == 0) {
                err = "downloaded file empty for " + url;
                std::filesystem::remove(dest, ec);
                return false;
            }
            return true;
        }

        void registerHttpSchemes(lnext::AssetResolver& resolver) {
            auto resolveUrl = [](const std::string& url, const std::string& /*anchor*/) -> std::string {
                auto cache = httpCacheDir() / urlToCacheName(url);
                if (!std::filesystem::exists(cache)) {
                    std::string err;
                    if (!fetchHttp(url, cache, err)) {
                        std::cerr << "[USDLoader] " << err << "\n";
                        return {};
                    }
                }
                return cache.string();
            };
            resolver.RegisterScheme("http", resolveUrl);
            resolver.RegisterScheme("https", resolveUrl);
        }

        // -----------------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------------

        // tydra::next matrices are USD's row-major, row-vector layout, which is
        // threepp's column-major, column-vector layout element for element.
        Matrix4 toMatrix4(const tnext::Matrix4& m) {
            return Matrix4().set(
                    m.m[0], m.m[4], m.m[8], m.m[12],
                    m.m[1], m.m[5], m.m[9], m.m[13],
                    m.m[2], m.m[6], m.m[10], m.m[14],
                    m.m[3], m.m[7], m.m[11], m.m[15]);
        }

        // Skeletons carry their transforms in double.
        Matrix4 toMatrix4(const tnext::Matrix4d& m) {
            tnext::Matrix4 f;
            for (int i = 0; i < 16; ++i) f.m[i] = static_cast<float>(m.m[i]);
            return toMatrix4(f);
        }

        std::string lowercaseExtension(const std::filesystem::path& p) {
            std::string e = p.extension().string();
            for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return e;
        }

        // A UDIM texture set names its tiles 1001, 1002, ... (1001 + the u tile + 10 * the v
        // tile) in place of a <UDIM> token.
        bool hasUDIM(const std::string& s) {
            return s.find("<UDIM>") != std::string::npos;
        }

        // The path of one tile of a UDIM texture set (any other path as it is).
        std::string resolveUDIM(const std::string& s, int tile) {
            std::string out = s;
            const std::string tok = "<UDIM>";
            auto pos = out.find(tok);
            if (pos != std::string::npos) out.replace(pos, tok.size(), std::to_string(tile));
            return out;
        }

        bool isFile(const std::filesystem::path& p) {
            std::error_code ec;
            return !p.empty() && std::filesystem::is_regular_file(p, ec);
        }

        // -----------------------------------------------------------------------
        // Geometry
        // -----------------------------------------------------------------------

        // Static linear-blend skinning. tydra surfaces UsdSkel data (per-vertex
        // joint indices/weights, per-joint bind/rest transforms,
        // geomBindTransform) but does NOT deform `RenderMesh::points` — those
        // stay in mesh-local bind pose. For a non-animated render we bake LBS at
        // the rest pose so skinned geometry (e.g. the brake cables on the
        // CarbonFrameBike) shows in its authored shape rather than as the raw
        // bind layout.
        //
        // Maths per vertex:
        //   p_bind  = geomBindTransform * v_local
        //   p_rest  = sum_k(weight_k * (restWorld_k * inverse(bindWorld_k)) * p_bind)
        // where restWorld_k is the world-space rest transform of joint k
        // (parent_world * local_rest down the hierarchy) and bindWorld_k is the
        // joint's world-space bind transform (USD `bindTransforms` is already
        // world-space).
        std::vector<float> restPosePositions(const tnext::RenderMesh& rm,
                                             const tnext::RenderScene& scene) {
            const size_t numV = rm.point_count();
            std::vector<float> pos(numV * 3);
            rm.points.copy_to(pos.data());

            const auto* skin = rm.skin.get();
            if (!skin || skin->skeleton_id < 0 ||
                skin->skeleton_id >= static_cast<int32_t>(scene.skeletons.size()) ||
                skin->influences_per_vertex == 0) {
                return pos;
            }
            const auto& joints = scene.skeletons[skin->skeleton_id].joints;
            const size_t es = skin->influences_per_vertex;
            // Constant-interpolated skinning authors one influence set for
            // every point.
            const bool constant = skin->joint_indices.size() == es;
            if (!constant && (skin->joint_indices.size() < numV * es ||
                              skin->joint_weights.size() < numV * es)) {
                return pos;
            }

            // USD lists parents before their children, so one forward pass
            // finds every joint's world rest transform.
            std::vector<Matrix4> skinMatrices(joints.size());
            std::vector<Matrix4> restWorld(joints.size());
            for (size_t j = 0; j < joints.size(); ++j) {
                const auto& joint = joints[j];
                const Matrix4 localRest = toMatrix4(joint.rest_transform);
                if (joint.parent_id >= 0 && static_cast<size_t>(joint.parent_id) < j) {
                    restWorld[j].multiplyMatrices(restWorld[joint.parent_id], localRest);
                } else {
                    restWorld[j] = localRest;
                }
                Matrix4 bindWorldInv = toMatrix4(joint.bind_transform);
                bindWorldInv.invert();
                skinMatrices[j].multiplyMatrices(restWorld[j], bindWorldInv);
            }

            const Matrix4 geomBind = toMatrix4(skin->geom_bind_transform);
            for (size_t v = 0; v < numV; ++v) {
                Vector3 vBind(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
                vBind.applyMatrix4(geomBind);

                Vector3 vOut(0.f, 0.f, 0.f);
                float totalW = 0.f;
                const size_t base = constant ? 0 : v * es;
                for (size_t k = 0; k < es; ++k) {
                    const size_t jId = skin->joint_indices[base + k];
                    const float w = skin->joint_weights[base + k];
                    if (w <= 0.f || jId >= skinMatrices.size()) continue;
                    Vector3 d = vBind;
                    d.applyMatrix4(skinMatrices[jId]);
                    vOut.x += w * d.x;
                    vOut.y += w * d.y;
                    vOut.z += w * d.z;
                    totalW += w;
                }
                if (totalW > 0.f) {
                    vOut.x /= totalW;
                    vOut.y /= totalW;
                    vOut.z /= totalW;
                } else {
                    vOut = vBind;
                }
                pos[v * 3] = vOut.x;
                pos[v * 3 + 1] = vOut.y;
                pos[v * 3 + 2] = vOut.z;
            }
            return pos;
        }

        // One threepp mesh per material: GeomSubsets split a USD mesh into
        // per-material triangle ranges, and a threepp mesh draws one material.
        struct SubMesh {
            int materialId = -1;
            int tile = 0;// the UDIM tile its UVs were moved from (0: none, see buildSubMeshes)
            std::shared_ptr<BufferGeometry> geometry;
        };

        // The element of a primvar that corner k of a triangulated mesh reads,
        // whatever the primvar's interpolation (-1: none). Uniform primvars
        // need the authored face of each triangle.
        long long primvarElement(const tnext::RenderMesh& rm, tnext::Interpolation interp, size_t k) {
            switch (interp) {
                case tnext::Interpolation::Constant:
                    return 0;
                case tnext::Interpolation::Vertex:
                case tnext::Interpolation::Varying:
                    return rm.triangulated_indices[k];
                case tnext::Interpolation::FaceVarying:
                    return k < rm.triangulated_face_vertex_indices.size()
                                   ? static_cast<long long>(rm.triangulated_face_vertex_indices[k])
                                   : -1;
                case tnext::Interpolation::Uniform: {
                    const auto& faceOffsets = rm.face_triangle_offsets;
                    if (faceOffsets.empty()) return -1;
                    const auto t = static_cast<uint32_t>(k / 3);
                    auto it = std::upper_bound(faceOffsets.begin(), faceOffsets.end(), t);
                    return static_cast<long long>(it - faceOffsets.begin()) - 1;
                }
            }
            return -1;
        }

        // Whether a primvar of `count` elements has an element for every corner.
        bool coversCorners(const tnext::RenderMesh& rm, tnext::Interpolation interp, size_t count) {
            if (count == 0) return false;
            for (size_t k = 0; k < rm.triangulated_indices.size(); ++k) {
                const long long i = primvarElement(rm, interp, k);
                if (i < 0 || static_cast<size_t>(i) >= count) return false;
            }
            return true;
        }

        // `triMaterial` holds the material of every triangle, `triTile` (empty
        // for none) the UDIM tile the UVs of a triangle lie in when its
        // material is a UDIM texture set: a part per material and tile, the
        // tile's UVs moved into the 0..1 square its own textures cover.
        std::vector<SubMesh> buildSubMeshes(const tnext::RenderMesh& rm,
                                            const tnext::RenderScene& scene,
                                            const std::vector<int>& triMaterial,
                                            const std::vector<int>& triTile = {}) {
            std::vector<SubMesh> out;
            if (!rm.is_triangulated || rm.points.empty()) return out;

            const size_t numV = rm.point_count();
            const size_t numCorners = rm.triangulated_indices.size();
            const size_t numTris = numCorners / 3;
            if (numTris == 0 || triMaterial.size() != numTris) return out;
            if (!triTile.empty() && triTile.size() != numTris) return out;

            const std::vector<float> positions = restPosePositions(rm, scene);

            const auto& uv = rm.texcoords_0;
            const bool haveUV = coversCorners(rm, rm.texcoords_0_interp, uv.size() / 2);
            const auto& normals = rm.normals;
            const bool haveNormals = coversCorners(rm, rm.normals_interp, normals.size() / 3);

            // Corners share a vertex when they share a point, a UV and an
            // authored normal: the recomputed normals then stay smooth across
            // faces and split only where the asset authored a seam or a hard
            // edge.
            struct Key {
                uint32_t point;
                uint32_t attr[5];
                bool operator==(const Key& o) const {
                    return point == o.point && std::memcmp(attr, o.attr, sizeof(attr)) == 0;
                }
            };
            struct KeyHash {
                size_t operator()(const Key& k) const {
                    size_t h = size_t(k.point) * 0x9E3779B97F4A7C15ull;
                    for (uint32_t a : k.attr) h = (h ^ a) * 0x100000001B3ull;
                    return h;
                }
            };

            std::map<std::pair<int, int>, size_t> partOf;
            struct Part {
                int materialId;
                int tile;
                float du, dv;// the tile's corner in UV space
                std::unordered_map<Key, unsigned int, KeyHash> vertexOf;
                std::vector<float> pos, uvs;
                std::vector<unsigned int> index;
            };
            std::vector<Part> parts;

            for (size_t t = 0; t < numTris; ++t) {
                const int tile = triTile.empty() ? 0 : triTile[t];
                auto [it, inserted] = partOf.emplace(std::make_pair(triMaterial[t], tile), parts.size());
                if (inserted) {
                    const int n = tile > 0 ? tile - 1001 : 0;
                    parts.push_back({triMaterial[t], tile, static_cast<float>(n % 10), static_cast<float>(n / 10), {}, {}, {}, {}});
                }
                Part& g = parts[it->second];
                for (size_t c = 0; c < 3; ++c) {
                    const size_t k = t * 3 + c;
                    const uint32_t p = rm.triangulated_indices[k];
                    if (p >= numV) return {};
                    Key key{p, {}};
                    float s = 0, tt = 0;
                    if (haveUV) {
                        const auto i = static_cast<size_t>(primvarElement(rm, rm.texcoords_0_interp, k));
                        s = uv[i * 2];
                        tt = uv[i * 2 + 1];
                        std::memcpy(&key.attr[0], &s, 4);
                        std::memcpy(&key.attr[1], &tt, 4);
                    }
                    if (haveNormals) {
                        const auto i = static_cast<size_t>(primvarElement(rm, rm.normals_interp, k));
                        for (int a = 0; a < 3; ++a) {
                            const float n = normals[i * 3 + a];
                            std::memcpy(&key.attr[2 + a], &n, 4);
                        }
                    }
                    auto [vit, fresh] = g.vertexOf.emplace(key, static_cast<unsigned int>(g.pos.size() / 3));
                    if (fresh) {
                        g.pos.insert(g.pos.end(), {positions[p * 3], positions[p * 3 + 1], positions[p * 3 + 2]});
                        if (haveUV) g.uvs.insert(g.uvs.end(), {s - g.du, tt - g.dv});
                    }
                    g.index.push_back(vit->second);
                }
            }

            // leftHanded meshes come back with their winding already reversed,
            // so every index buffer here is counter-clockwise.
            for (auto& g : parts) {
                auto geometry = BufferGeometry::create();
                geometry->setAttribute("position", FloatBufferAttribute::create(std::move(g.pos), 3));
                if (haveUV) geometry->setAttribute("uv", FloatBufferAttribute::create(std::move(g.uvs), 2));
                geometry->setIndex(g.index);
                // Authored normals from real-world USD assets are too
                // unreliable to trust globally: exporters disagree on the
                // "outward" convention, and even within a single asset the
                // direction can flip across submeshes (chess set: King/Queen
                // leftHanded with one convention, Pawn rightHanded with
                // another). Recompute from the winding: guaranteed outward,
                // costs only smoothing-group fidelity.
                geometry->computeVertexNormals();
                out.push_back({g.materialId, g.tile, std::move(geometry)});
            }
            return out;
        }

        // Identical meshes. A scene that instances one asset through USD's
        // native instancing or many references (Intel's Jungle Ruins: 195
        // tree prototypes, 65 copies each of 3 trees) reaches us as one
        // RenderMesh per copy; their geometry is built once and shared.
        // Everything buildSubMeshes reads takes part, so equal inputs mean
        // equal geometry.

        struct ContentHash {
            uint64_t h = 0x9E3779B97F4A7C15ull;

            void word(uint64_t w) {
                h ^= w;
                h *= 0xff51afd7ed558ccdull;
                h ^= h >> 32;
            }

            void bytes(const void* data, size_t n) {
                const auto* c = static_cast<const unsigned char*>(data);
                size_t i = 0;
                for (; i + 8 <= n; i += 8) {
                    uint64_t w;
                    std::memcpy(&w, c + i, 8);
                    word(w);
                }
                uint64_t tail = 0;
                std::memcpy(&tail, c + i, n - i);
                word(tail ^ (uint64_t(n - i) << 56));
            }

            template<class Chunked>
            void chunked(const Chunked& a) {
                word(a.size());
                for (size_t k = 0; k < a.chunk_count(); ++k) {
                    if (const size_t n = a.chunk_size(k)) bytes(a.chunk_data(k), n * sizeof(*a.chunk_data(k)));
                }
            }
        };

        // Chunk by chunk; arrays sharing their chunks (LightUSD's
        // copy-on-write share_from) compare by pointer.
        template<class Chunked>
        bool sameChunked(const Chunked& a, const Chunked& b) {
            if (a.size() != b.size()) return false;
            for (size_t k = 0; k < a.chunk_count() && k < b.chunk_count(); ++k) {
                const size_t n = a.chunk_size(k);
                if (n == 0) break;
                const auto* pa = a.chunk_data(k);
                const auto* pb = b.chunk_data(k);
                if (pa != pb && std::memcmp(pa, pb, n * sizeof(*pa)) != 0) return false;
            }
            return true;
        }

        // `parts`: each triangle's part (its material, and the UDIM tile of
        // it) as an index into the mesh's parts in first-use order, which is
        // how buildSubMeshes orders them. Copies bound to copies of a
        // material match.
        uint64_t meshContentHash(const tnext::RenderMesh& rm, const std::vector<int>& parts) {
            ContentHash c;
            c.word(rm.is_triangulated);
            c.chunked(rm.points);
            c.chunked(rm.triangulated_indices);
            c.chunked(rm.triangulated_face_vertex_indices);
            c.word(static_cast<uint64_t>(rm.texcoords_0_interp));
            c.chunked(rm.texcoords_0);
            c.word(static_cast<uint64_t>(rm.normals_interp));
            c.chunked(rm.normals);
            c.bytes(rm.face_triangle_offsets.data(), rm.face_triangle_offsets.size() * sizeof(uint32_t));
            c.bytes(parts.data(), parts.size() * sizeof(int));
            return c.h;
        }

        bool sameMeshContent(const tnext::RenderMesh& a, const tnext::RenderMesh& b) {
            return a.is_triangulated == b.is_triangulated &&
                   a.texcoords_0_interp == b.texcoords_0_interp &&
                   a.normals_interp == b.normals_interp &&
                   a.face_triangle_offsets == b.face_triangle_offsets &&
                   sameChunked(a.points, b.points) &&
                   sameChunked(a.triangulated_indices, b.triangulated_indices) &&
                   sameChunked(a.triangulated_face_vertex_indices, b.triangulated_face_vertex_indices) &&
                   sameChunked(a.texcoords_0, b.texcoords_0) &&
                   sameChunked(a.normals, b.normals);
        }

    }// namespace

    // -----------------------------------------------------------------------
    // Impl
    // -----------------------------------------------------------------------

    struct USDLoader::Impl {
        TextureLoader texLoader;
        // An MDL material's maps, each a clone of the texture loader's shared
        // texture, set to repeat and placed by its material's UV transform (see
        // applyMdlInputs), by file and transform. Held as weakly as the texture
        // loader's own cache, so loads of the same asset share them.
        std::unordered_map<std::string, std::weak_ptr<Texture>> mdlTextures;
        bool ignoreUpDirection = false;
        // The latest load's DomeLight environment texture (if any) — exposed
        // to the caller via USDResult so they can assign it to scene.environment
        // / scene.background. Reset at the start of each load().
        std::shared_ptr<Texture> _envTexture;
        // The latest load's articulated robot, when it has UsdPhysics joints.
        std::shared_ptr<Robot> _robot;
        std::optional<USDArticulationDesc> _articulation;
        // variantSet -> selection, over the file's own selections.
        std::map<std::string, std::string> variants;

        // State of one load: the converted scene plus caches keyed by its ids.
        // The scene grows when a material LightUSD skipped is converted on
        // demand.
        struct Build {
            const lnext::Stage& stage;
            tnext::RenderScene& scene;
            tnext::RenderSceneConverter& converter;
            std::filesystem::path baseDir;

            std::vector<std::shared_ptr<Texture>> textures;
            std::vector<bool> textureDone;
            // by material id and UDIM tile (see material)
            std::unordered_map<uint64_t, std::shared_ptr<Material>> materials;
            std::unordered_map<int, std::vector<SubMesh>> subMeshes;
            // Content hash -> meshes built under it (see subMeshesFor).
            struct BuiltMesh {
                int32_t meshId;
                std::vector<int> parts;
                std::vector<int> tiles;// each part's UDIM tile
            };
            // UDIM texture sets: the tile the material being converted takes
            // its textures from, whether its conversion met a texture set, and
            // which materials have one.
            int udimTile = 1001;
            bool sawUDIM = false;
            std::unordered_map<int, bool> udimMaterials;
            std::unordered_map<uint64_t, std::vector<BuiltMesh>> meshesByContent;
            std::unordered_map<std::string, std::shared_ptr<Texture>> sampledTextures;
            std::unordered_set<int> prototypeNodes;
            std::unordered_map<std::string, std::shared_ptr<Object3D>> objectByPath;
            // Transmissive materials that bound a volume (thinWalledShader):
            // the first mesh that wears one gives it its thickness.
            std::unordered_set<const Material*> volumeMaterials;
            int meshCount = 0;
        };

        std::shared_ptr<Group> load(const std::filesystem::path& path) {
            _envTexture.reset();
            _robot.reset();
            _articulation.reset();
            const std::string pathStr = path.string();
            const std::filesystem::path baseDir = path.parent_path();

            // --- Load and compose (sublayers, references, payloads, inherits,
            // specializes, variants) through next::pcp ---
            lnext::Stage stage;
            std::string warn, err;
            lnext::AssetResolver resolver;
            resolver.SetWorkingDirectory(baseDir.string());
            registerHttpSchemes(resolver);
            lnext::pcp::CompositionOptions composition;
            composition.variant_overrides = variants;
            const auto* compositionOptions = variants.empty() ? nullptr : &composition;
            bool ok;
            if (lowercaseExtension(path) == ".usdz") {
                // The package's own entries resolve inside the archive; the
                // composed load anchors on its root layer.
                ok = lnext::LoadUSDComposed(pathStr, &stage, &warn, &err, compositionOptions);
            } else {
                lnext::LoadUSDOptions options;
                ok = lnext::LoadUSD(pathStr, &stage, options, &warn, &err) &&
                     lnext::ComposeLoadedStage(&stage, resolver, pathStr, options, &warn, &err, compositionOptions);
            }
            if (!ok) {
                std::cerr << "[USDLoader] Failed to load '" << pathStr << "': " << err << "\n";
                return nullptr;
            }

            // --- Convert to a RenderScene (triangulated meshes, materials,
            // texture references, lights) ---
            tnext::ConverterConfig config;
            config.mesh.triangulate = true;
            // Normals are recomputed per submesh (see buildSubMeshes).
            config.mesh.compute_normals = false;
            config.mesh.compute_tangents = false;
            // Textures are decoded by threepp's TextureLoader, which also
            // handles the formats LightUSD does not (DDS, ...).
            config.material.load_textures = false;
            config.animation.enabled = false;
            config.asset_base_dir = baseDir.string();
            config.asset_resolver = &resolver;

            tnext::RenderSceneConverter converter(config);
            tnext::ConvertResult converted = converter.Convert(stage);
            if (!converted.success) {
                std::cerr << "[USDLoader] Failed to convert '" << pathStr << "': " << converted.error << "\n";
                return nullptr;
            }
            tnext::RenderScene& scene = converted.scene;

            Build build{stage, scene, converter, baseDir};
            build.textures.resize(scene.textures.size());
            build.textureDone.assign(scene.textures.size(), false);
            // PointInstancer prototypes render only through their instancer.
            for (const auto& instancer : scene.point_instancers) {
                for (int32_t id : instancer.prototype_node_ids) markSubtree(scene, id, build.prototypeNodes);
            }

            // --- Root group + Z-up correction ---
            // Skipped when this loader is being used by an outer system
            // (URDF/SDF/...) that owns the coordinate frame.
            auto root = Group::create();
            root->name = scene.default_prim;
            if (!ignoreUpDirection && scene.up_axis == tnext::RenderScene::UpAxis::Z) {
                root->rotation.x = -math::PI / 2.0f;
            }

            for (int32_t id : scene.root_nodes) {
                if (auto child = buildNode(build, id)) root->add(child);
            }

            // --- UsdLux lights ---
            addLights(build, *root);

            if (build.meshCount == 0) {
                std::cerr << "[USDLoader] No usable geometry in '" << pathStr << "'\n";
                return nullptr;
            }

            // --- UsdPhysics joints ---
            _robot = buildRobot(build, root, _articulation);

            return root;
        }

        static void markSubtree(const tnext::RenderScene& scene, int32_t id, std::unordered_set<int>& out) {
            const auto* node = scene.get_node(id);
            if (!node || !out.insert(id).second) return;
            for (int32_t child : node->children) markSubtree(scene, child, out);
        }

        // -------------------------------------------------------------------
        // Scene graph: one threepp object per USD prim, carrying the prim's
        // local transform, so links stay addressable by name.
        // -------------------------------------------------------------------

        std::shared_ptr<Object3D> buildNode(Build& b, int32_t id) {
            const auto* node = b.scene.get_node(id);
            if (!node || b.prototypeNodes.count(id)) return nullptr;
            if (const auto prim = b.stage.GetPrimAtPath(node->prim_path); prim.IsValid()) {
                // Colliders and other guide/proxy geometry (purpose is
                // inherited, so skipping the subtree covers it) are not for
                // rendering. LightUSD works out a prim's purpose but keeps
                // every prim in the render scene.
                const auto* purpose = prim.GetPropertyValue("purpose");
                const auto token = purpose ? purpose->as_token() : lnext::StringView();
                if (token && (*token == "guide" || *token == "proxy")) return nullptr;
            }

            std::shared_ptr<Object3D> obj;
            if (node->type == tnext::NodeType::Mesh) {
                obj = meshObject(b, node->data_id);
            } else if (node->type == tnext::NodeType::PointInstancer) {
                obj = instancerObject(b, node->data_id);
            }
            if (!obj) obj = Group::create();

            obj->name = node->name;
            obj->visible = node->visible;
            b.objectByPath[node->prim_path] = obj;
            Matrix4 local = toMatrix4(node->local_transform);
            if (node->has_reset_xform) {
                // The prim's ops are world-space (rigid bodies in physics
                // layers author `!resetXformStack!` + their simulation pose);
                // express that pose under the parent this hierarchy keeps.
                Matrix4 parentWorldInv;
                if (const auto* parent = b.scene.get_node(node->parent_id)) {
                    parentWorldInv = toMatrix4(parent->world_transform);
                    parentWorldInv.invert();
                }
                local.multiplyMatrices(parentWorldInv, toMatrix4(node->world_transform));
            }
            local.decompose(obj->position, obj->quaternion, obj->scale);

            for (int32_t child : node->children) {
                if (auto c = buildNode(b, child)) obj->add(c);
            }
            return obj;
        }

        // The scene's material for a bound path. LightUSD (1.0.0 as rc4)
        // treats an `instanceable` Material prim as a native instance and
        // converts only some of them (the SimReady layout gives every mesh
        // instance its own instanceable VisualMaterials: the AgiBot G2 comes
        // back with 42 of its 64 meshes unbound), so bindings to the rest
        // would be lost. Convert those here, once per path.
        int materialId(Build& b, const std::string& path) {
            if (path.empty()) return -1;
            if (auto it = b.scene.material_by_path.find(path); it != b.scene.material_by_path.end()) return it->second;
            int id = -1;
            const auto prim = b.stage.GetPrimAtPath(path);
            tnext::RenderMaterial converted;
            if (prim.IsValid() && b.converter.ConvertMaterial(b.stage, prim, &converted, &b.scene)) {
                id = static_cast<int>(b.scene.materials.size());
                b.scene.materials.push_back(std::move(converted));
            }
            b.scene.material_by_path[path] = id;
            return id;
        }

        // The material of every triangle: the mesh's (inherited) binding,
        // overridden by its materialBind GeomSubsets. Resolved here rather
        // than taken from the RenderMesh, which drops the bindings LightUSD
        // did not convert (see materialId).
        std::vector<int> triangleMaterials(Build& b, const tnext::RenderMesh& rm) {
            const size_t numTris = rm.triangulated_indices.size() / 3;
            std::vector<int> out(numTris, rm.material_id);
            const auto prim = b.stage.GetPrimAtPath(rm.prim_path);
            const auto& offsets = rm.face_triangle_offsets;
            if (!prim.IsValid() || offsets.empty()) {
                // Untriangulated topology: LightUSD's subset runs are face
                // ranges we cannot map, so keep its whole-mesh binding.
                return out;
            }
            const int meshMaterial = materialId(b, lnext::GetInheritedBoundMaterialPathForPurpose(b.stage, rm.prim_path, "preview"));
            if (meshMaterial >= 0) std::fill(out.begin(), out.end(), meshMaterial);

            for (const auto& subset : tnext::GetGeomSubsets(prim)) {
                if (!subset.family_name.empty() && subset.family_name != "materialBind") continue;
                std::string path = subset.material_path;
                if (path.empty()) {
                    if (const auto subsetPrim = b.stage.GetPrimAtPath(subset.path); subsetPrim.IsValid()) {
                        path = lnext::GetBoundMaterialPath(subsetPrim);
                    }
                }
                const int id = materialId(b, path);
                if (id < 0) continue;
                // Subset indices number the authored faces; map them through
                // any faces sanitization dropped, then to their triangles.
                for (int32_t authored : subset.indices()) {
                    if (authored < 0) continue;
                    auto face = static_cast<uint32_t>(authored);
                    if (rm.sanitize_dropped_faces > 0) {
                        if (face >= rm.sanitize_face_remap.size() || rm.sanitize_face_remap[face] < 0) continue;
                        face = static_cast<uint32_t>(rm.sanitize_face_remap[face]);
                    }
                    if (size_t(face) + 1 >= offsets.size()) continue;
                    for (uint32_t t = offsets[face]; t < offsets[face + 1] && t < numTris; ++t) out[t] = id;
                }
            }
            return out;
        }

        // Whether a material's textures are UDIM texture sets (converting it,
        // once, finds out).
        bool usesUDIM(Build& b, int materialId, const tnext::RenderMesh& rm) {
            if (!b.scene.get_material(materialId)) return false;
            material(b, materialId, rm);
            const auto it = b.udimMaterials.find(materialId);
            return it != b.udimMaterials.end() && it->second;
        }

        // The UDIM tile (1001 + the u tile + 10 * the v tile) the UVs of each
        // triangle lie in, for the triangles whose material is a UDIM texture
        // set; 0 for the rest, and empty when none is. A UDIM asset keeps
        // every face inside one tile, so its UV centroid tells which.
        std::vector<int> triangleTiles(Build& b, const tnext::RenderMesh& rm, const std::vector<int>& triMaterial) {
            std::unordered_map<int, bool> udim;
            bool any = false;
            for (int id : triMaterial) {
                if (udim.contains(id)) continue;
                const bool u = usesUDIM(b, id, rm);
                udim[id] = u;
                any = any || u;
            }
            if (!any) return {};
            const auto& uv = rm.texcoords_0;
            const bool haveUV = coversCorners(rm, rm.texcoords_0_interp, uv.size() / 2);
            std::vector<int> out(triMaterial.size(), 0);
            for (size_t t = 0; t < triMaterial.size(); ++t) {
                if (!udim[triMaterial[t]]) continue;
                float u = 0, v = 0;
                if (haveUV) {
                    for (size_t c = 0; c < 3; ++c) {
                        const auto i = static_cast<size_t>(primvarElement(rm, rm.texcoords_0_interp, t * 3 + c));
                        u += uv[i * 2] / 3.f;
                        v += uv[i * 2 + 1] / 3.f;
                    }
                }
                const int column = std::clamp(static_cast<int>(std::floor(u)), 0, 9);
                const int row = std::max(0, static_cast<int>(std::floor(v)));
                out[t] = 1001 + column + 10 * row;
            }
            return out;
        }

        // A mesh's parts, built once per mesh, and their geometry once per
        // distinct content: a mesh identical to one already built (same
        // points, topology, primvars and material layout) takes that mesh's
        // geometries with its own materials. Skinned meshes are posed per
        // skeleton and always built.
        const std::vector<SubMesh>& subMeshesFor(Build& b, int32_t meshId, const tnext::RenderMesh& rm) {
            if (auto it = b.subMeshes.find(meshId); it != b.subMeshes.end()) return it->second;

            const std::vector<int> triMaterial = triangleMaterials(b, rm);
            const std::vector<int> triTile = triangleTiles(b, rm, triMaterial);
            if (rm.skin) return b.subMeshes.emplace(meshId, buildSubMeshes(rm, b.scene, triMaterial, triTile)).first->second;

            // The parts (a material, and a UDIM tile of it) in first-use
            // order (buildSubMeshes' part order) and each triangle's index
            // into them.
            std::vector<std::pair<int, int>> keys;
            std::vector<int> parts(triMaterial.size());
            std::map<std::pair<int, int>, int> partOf;
            for (size_t t = 0; t < triMaterial.size(); ++t) {
                const std::pair<int, int> key{triMaterial[t], triTile.empty() ? 0 : triTile[t]};
                auto [it, inserted] = partOf.emplace(key, static_cast<int>(keys.size()));
                if (inserted) keys.push_back(key);
                parts[t] = it->second;
            }
            std::vector<int> tiles(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) tiles[i] = keys[i].second;

            auto& built = b.meshesByContent[meshContentHash(rm, parts)];
            for (const auto& other : built) {
                const auto* otherMesh = b.scene.get_mesh(other.meshId);
                if (!otherMesh || other.parts != parts || other.tiles != tiles || !sameMeshContent(*otherMesh, rm)) continue;
                const auto& shared = b.subMeshes.at(other.meshId);
                std::vector<SubMesh> out;
                out.reserve(shared.size());
                for (size_t i = 0; i < shared.size() && i < keys.size(); ++i) {
                    out.push_back({keys[i].first, keys[i].second, shared[i].geometry});
                }
                return b.subMeshes.emplace(meshId, std::move(out)).first->second;
            }
            auto& out = b.subMeshes.emplace(meshId, buildSubMeshes(rm, b.scene, triMaterial, triTile)).first->second;
            built.push_back({meshId, std::move(parts), std::move(tiles)});
            return out;
        }

        // A mesh prim: one Mesh, or a Group of one Mesh per GeomSubset
        // material (and UDIM tile).
        std::shared_ptr<Object3D> meshObject(Build& b, int32_t meshId) {
            const auto* rm = b.scene.get_mesh(meshId);
            if (!rm) return nullptr;
            const auto& parts = subMeshesFor(b, meshId, *rm);
            if (parts.empty()) return nullptr;

            auto makeMesh = [&](const SubMesh& part) {
                auto mesh = Mesh::create(part.geometry, material(b, part.materialId, *rm, part.tile));
                sizeVolume(b, mesh->material(), *part.geometry);
                ++b.meshCount;
                return mesh;
            };
            if (parts.size() == 1) return makeMesh(parts.front());
            auto group = Group::create();
            for (const auto& part : parts) group->add(makeMesh(part));
            return group;
        }

        // A PointInstancer prim: one InstancedMesh per prototype mesh (per
        // material part of it), its instances placed in the instancer's space
        // (prototype mesh transform * instance transform). Scattered
        // vegetation runs to millions of instances; an object per instance
        // would not fit in memory.
        std::shared_ptr<Object3D> instancerObject(Build& b, int32_t instancerId) {
            auto group = Group::create();
            const auto draws = b.scene.get_point_instancer_draws(instancerId);
            std::vector<int32_t> meshOrder;
            std::unordered_map<int32_t, std::vector<const tnext::RenderPointInstanceDraw*>> byMesh;
            for (const auto& draw : draws) {
                auto& list = byMesh[draw.mesh_id];
                if (list.empty()) meshOrder.push_back(draw.mesh_id);
                list.push_back(&draw);
            }
            for (int32_t meshId : meshOrder) {
                const auto* rm = b.scene.get_mesh(meshId);
                if (!rm) continue;
                const auto& instances = byMesh[meshId];
                for (const auto& part : subMeshesFor(b, meshId, *rm)) {
                    auto mesh = InstancedMesh::create(part.geometry, material(b, part.materialId, *rm, part.tile), instances.size());
                    sizeVolume(b, mesh->material(), *part.geometry);
                    for (size_t i = 0; i < instances.size(); ++i) {
                        mesh->setMatrixAt(i, toMatrix4(instances[i]->transform));
                    }
                    mesh->instanceMatrix()->needsUpdate();
                    mesh->computeBoundingSphere();
                    mesh->name = rm->name;
                    ++b.meshCount;
                    group->add(mesh);
                }
            }
            return group;
        }

        // -------------------------------------------------------------------
        // Textures
        // -------------------------------------------------------------------

        // Decoded pixels from LightUSD (package entries, remote assets) as an
        // RGBA8 texture.
        static std::shared_ptr<Texture> textureFromDecoded(const tnext::TextureImage& img, const std::string& name) {
            const size_t w = img.width, h = img.height, ch = img.channels;
            if (w == 0 || h == 0 || ch == 0 || ch > 4) return nullptr;
            const size_t n = w * h * ch;
            std::vector<float> src(n);
            if (img.component_type == tnext::ComponentType::UInt8 && img.data.size() >= n) {
                for (size_t i = 0; i < n; ++i) src[i] = img.data[i] / 255.f;
            } else if (img.component_type == tnext::ComponentType::Float32 && img.data.size() >= n * 4) {
                for (size_t i = 0; i < n; ++i) {
                    uint8_t bytes[4] = {img.data[i * 4], img.data[i * 4 + 1], img.data[i * 4 + 2], img.data[i * 4 + 3]};
                    std::memcpy(&src[i], bytes, 4);
                }
            } else {
                return nullptr;
            }
            // Top-down rows in, bottom-up rows out (threepp's flipY).
            std::vector<unsigned char> pixels(w * h * 4);
            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    const float* s = &src[((h - 1 - y) * w + x) * ch];
                    unsigned char* d = &pixels[(y * w + x) * 4];
                    auto byte = [](float v) { return static_cast<unsigned char>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); };
                    const float r = s[0];
                    const float g = ch >= 3 ? s[1] : r;
                    const float bl = ch >= 3 ? s[2] : r;
                    const float a = ch == 4 ? s[3] : (ch == 2 ? s[1] : 1.f);
                    d[0] = byte(r);
                    d[1] = byte(g);
                    d[2] = byte(bl);
                    d[3] = byte(a);
                }
            }
            auto tex = Texture::create(Image(std::move(pixels), static_cast<unsigned int>(w), static_cast<unsigned int>(h)));
            tex->name = name;
            tex->format = Format::RGBA;
            tex->needsUpdate();
            return tex;
        }

        std::shared_ptr<Texture> loadImage(Build& b, const std::string& resolved, const std::string& authored) {
            const std::string path = resolveUDIM(resolved.empty() ? authored : resolved, b.udimTile);
            if (isFile(path)) return texLoader.load(path, true);
            const std::string rel = resolveUDIM(authored, b.udimTile);
            if (isFile(b.baseDir / rel)) return texLoader.load(b.baseDir / rel, true);
            // Package entries (.usdz) and anything else only LightUSD can read.
            tnext::TextureImage img;
            if (!b.converter.LoadTexture(path, &img) || !img.is_loaded()) return nullptr;
            return textureFromDecoded(img, authored);
        }

        std::shared_ptr<Texture> texture(Build& b, int32_t texId) {
            if (texId < 0 || static_cast<size_t>(texId) >= b.scene.textures.size()) return nullptr;
            if (b.textureDone.size() < b.scene.textures.size()) {
                // Materials converted on demand add textures.
                b.textures.resize(b.scene.textures.size());
                b.textureDone.resize(b.scene.textures.size(), false);
            }
            const auto& rt = b.scene.textures[texId];
            std::string resolved;
            // A UDIM set LightUSD found the tiles of: its image is the first tile
            // (1001), a concrete path with nothing left to substitute, so the tile
            // the material is for comes from the set's own list. A tile the set
            // lacks gives no map.
            const bool udimSet = rt.udim_texture_id >= 0 && static_cast<size_t>(rt.udim_texture_id) < b.scene.udim_textures.size();
            bool missingTile = false;
            if (udimSet) {
                missingTile = true;
                for (const auto& tile : b.scene.udim_textures[rt.udim_texture_id].tiles) {
                    if (static_cast<int>(tile.udim) != b.udimTile) continue;
                    if (tile.image_id >= 0 && static_cast<size_t>(tile.image_id) < b.scene.images.size()) {
                        resolved = b.scene.images[tile.image_id].resolved_path;
                        missingTile = false;
                    }
                    break;
                }
            } else if (rt.image_id >= 0 && static_cast<size_t>(rt.image_id) < b.scene.images.size()) {
                resolved = b.scene.images[rt.image_id].resolved_path;
            }
            // A UDIM texture set is one texture per tile: the one the material
            // being converted is for (see material), never remembered by id.
            const bool udim = udimSet || hasUDIM(resolved) || hasUDIM(rt.asset_path);
            if (udim) {
                b.sawUDIM = true;
            } else {
                if (b.textureDone[texId]) return b.textures[texId];
                b.textureDone[texId] = true;
            }
            if (missingTile) return nullptr;

            // Every material instance carries its own RenderTexture records;
            // uses that sample one image the same way share one texture
            // (a clone copies the image).
            std::ostringstream key;
            key << resolved << '|' << rt.asset_path << '|' << (udim ? b.udimTile : 0) << '|' << int(rt.wrap_s) << int(rt.wrap_t) << '|'
                << rt.offset.x << ',' << rt.offset.y << ',' << rt.scale.x << ',' << rt.scale.y << ',' << rt.rotation;
            if (auto it = b.sampledTextures.find(key.str()); it != b.sampledTextures.end()) {
                if (!udim) b.textures[texId] = it->second;
                return it->second;
            }
            auto tex = loadImage(b, resolved, rt.asset_path);
            b.sampledTextures[key.str()] = tex;// also remembers failures
            if (!tex) return nullptr;

            auto mapWrap = [](tnext::WrapMode w) {
                switch (w) {
                    case tnext::WrapMode::Repeat: return TextureWrapping::Repeat;
                    case tnext::WrapMode::Mirror: return TextureWrapping::MirroredRepeat;
                    default: return TextureWrapping::ClampToEdge;
                }
            };
            // UsdTransform2d: st' = rotate(scale * st) + translation.
            const bool hasXform = rt.offset.x != 0.f || rt.offset.y != 0.f ||
                                  rt.scale.x != 1.f || rt.scale.y != 1.f || rt.rotation != 0.f;
            const bool tiles = rt.scale.x != 1.f || rt.scale.y != 1.f;
            const auto wrapS = mapWrap(rt.wrap_s), wrapT = mapWrap(rt.wrap_t);
            if (hasXform || wrapS != tex->wrapS || wrapT != tex->wrapT) {
                // The texture loader caches by file path; textures sharing an
                // image would otherwise share the per-texture state below.
                tex = tex->clone();
                tex->wrapS = wrapS;
                tex->wrapT = wrapT;
                // A transform that scales the UVs means tiling; clamping
                // would defeat it.
                if (tiles && tex->wrapS == TextureWrapping::ClampToEdge) tex->wrapS = TextureWrapping::Repeat;
                if (tiles && tex->wrapT == TextureWrapping::ClampToEdge) tex->wrapT = TextureWrapping::Repeat;
                if (hasXform) {
                    const float c = std::cos(rt.rotation), s = std::sin(rt.rotation);
                    tex->matrix.set(
                            rt.scale.x * c, -rt.scale.y * s, rt.offset.x,
                            rt.scale.x * s, rt.scale.y * c, rt.offset.y,
                            0, 0, 1);
                    tex->matrixAutoUpdate = false;
                }
            }
            b.sampledTextures[key.str()] = tex;
            if (!udim) b.textures[texId] = tex;
            return tex;
        }

        std::shared_ptr<Texture> colorTexture(Build& b, int32_t texId) {
            auto t = texture(b, texId);
            if (t) t->colorSpace = ColorSpace::sRGB;
            return t;
        }

        std::shared_ptr<Texture> dataTexture(Build& b, int32_t texId) {
            auto t = texture(b, texId);
            if (t) t->colorSpace = ColorSpace::Linear;
            return t;
        }

        // -------------------------------------------------------------------
        // Materials
        // -------------------------------------------------------------------

        // A material; for the part of a mesh in UDIM `tile` (0: none), with
        // that tile's textures.
        std::shared_ptr<Material> material(Build& b, int materialId, const tnext::RenderMesh& rm, int tile = 0) {
            const auto* rmat = b.scene.get_material(materialId);
            if (!rmat) {
                // Unbound geometry shows its displayColor.
                auto mat = MeshStandardMaterial::create();
                if (rm.colors.size() >= 3 && rm.colors_interp == tnext::Interpolation::Constant) {
                    mat->color.setRGB(rm.colors[0], rm.colors[1], rm.colors[2]);
                }
                if (rm.double_sided) mat->side = Side::Double;
                return mat;
            }
            const uint64_t key = static_cast<uint64_t>(static_cast<uint32_t>(materialId)) << 32 | static_cast<uint32_t>(tile);
            auto it = b.materials.find(key);
            if (it != b.materials.end()) return it->second;

            b.udimTile = tile > 0 ? tile : 1001;
            b.sawUDIM = false;
            std::shared_ptr<MeshStandardMaterial> mat;
            if (rmat->shader_type == tnext::RenderMaterial::ShaderType::OpenPBR && rmat->openpbr) {
                mat = openPBRMaterial(b, *rmat->openpbr);
            } else if (rmat->preview_surface) {
                mat = previewSurfaceMaterial(b, *rmat->preview_surface);
            } else {
                mat = MeshStandardMaterial::create();
            }
            // No convertible surface shader (NVIDIA MDL: OmniPBR, OmniSurface,
            // OmniUe4Base): read the MDL inputs off the shader prim.
            if (rmat->default_fallback) applyMdlInputs(b, *rmat, *mat);
            if (auto* phys = dynamic_cast<MeshPhysicalMaterial*>(mat.get());
                phys && phys->transmission > 0.f && !thinWalledShader(b, *rmat)) {
                b.volumeMaterials.insert(phys);
            }

            mat->name = tile > 0 ? rmat->name + "." + std::to_string(tile) : rmat->name;
            switch (rmat->alpha_mode) {
                case tnext::RenderMaterial::AlphaMode::Mask:
                    mat->alphaTest = rmat->alpha_cutoff;
                    break;
                case tnext::RenderMaterial::AlphaMode::Blend:
                    mat->transparent = true;
                    break;
                default:
                    break;
            }
            if (rmat->double_sided || rm.double_sided) mat->side = Side::Double;

            if (tile == 0) b.udimMaterials[materialId] = b.sawUDIM;
            b.materials.emplace(key, mat);
            return mat;
        }

        std::shared_ptr<MeshStandardMaterial> previewSurfaceMaterial(Build& b, const tnext::PreviewSurfaceShader& ps) {
            const bool coated = ps.clearcoat.is_texture() || ps.clearcoat.as_float() > 0.f;
            std::shared_ptr<MeshStandardMaterial> mat;
            if (coated) {
                auto phys = MeshPhysicalMaterial::create();
                phys->clearcoat = ps.clearcoat.is_texture() ? 1.f : ps.clearcoat.as_float();
                if (ps.clearcoat.is_texture()) phys->clearcoatMap = dataTexture(b, ps.clearcoat.texture_id);
                phys->clearcoatRoughness = ps.clearcoat_roughness.as_float();
                mat = phys;
            } else {
                mat = MeshStandardMaterial::create();
            }

            if (ps.diffuse_color.is_texture()) {
                mat->map = colorTexture(b, ps.diffuse_color.texture_id);
            } else {
                const auto c = ps.diffuse_color.as_float3();
                mat->color.setRGB(c.x, c.y, c.z);
            }
            if (ps.roughness.is_texture()) {
                mat->roughnessMap = dataTexture(b, ps.roughness.texture_id);
                mat->roughness = 1.f;
            } else {
                mat->roughness = ps.roughness.as_float();
            }
            if (ps.metallic.is_texture()) {
                mat->metalnessMap = dataTexture(b, ps.metallic.texture_id);
                mat->metalness = 1.f;
            } else {
                mat->metalness = ps.metallic.as_float();
            }
            if (ps.emissive_color.is_texture()) {
                // three.js multiplies the emissive map by the emissive color.
                mat->emissiveMap = colorTexture(b, ps.emissive_color.texture_id);
                mat->emissive.setRGB(1, 1, 1);
            } else {
                const auto e = ps.emissive_color.as_float3();
                mat->emissive.setRGB(e.x, e.y, e.z);
            }
            if (ps.normal.is_texture()) mat->normalMap = dataTexture(b, ps.normal.texture_id);
            if (ps.occlusion.is_texture()) mat->aoMap = dataTexture(b, ps.occlusion.texture_id);

            if (ps.opacity.is_texture()) {
                // Usually the base color texture's alpha.
                if (ps.opacity_threshold.as_float() > 0.f) {
                    mat->alphaTest = ps.opacity_threshold.as_float();
                } else {
                    mat->transparent = true;
                }
            } else if (ps.opacity.as_float() < 1.f) {
                if (ps.opacity_threshold.as_float() > 0.f) {
                    mat->alphaTest = ps.opacity_threshold.as_float();
                } else {
                    mat->opacity = ps.opacity.as_float();
                    mat->transparent = true;
                }
            }
            return mat;
        }

        std::shared_ptr<MeshStandardMaterial> openPBRMaterial(Build& b, const tnext::OpenPBRSurfaceShader& o) {
            auto mat = MeshPhysicalMaterial::create();
            const float baseWeight = o.base_weight.is_texture() ? 1.f : o.base_weight.as_float();
            if (o.base_color.is_texture()) {
                mat->map = colorTexture(b, o.base_color.texture_id);
                mat->color.setRGB(baseWeight, baseWeight, baseWeight);
            } else {
                const auto c = o.base_color.as_float3();
                mat->color.setRGB(c.x * baseWeight, c.y * baseWeight, c.z * baseWeight);
            }
            if (o.base_metalness.is_texture()) {
                mat->metalnessMap = dataTexture(b, o.base_metalness.texture_id);
                mat->metalness = 1.f;
            } else {
                mat->metalness = o.base_metalness.as_float();
            }
            if (o.specular_roughness.is_texture()) {
                mat->roughnessMap = dataTexture(b, o.specular_roughness.texture_id);
                mat->roughness = 1.f;
            } else {
                mat->roughness = o.specular_roughness.as_float();
            }
            if (!o.specular_ior.is_texture()) mat->ior = o.specular_ior.as_float();
            if (!o.transmission_weight.is_texture()) mat->transmission = o.transmission_weight.as_float();
            if (!o.coat_weight.is_texture()) mat->clearcoat = o.coat_weight.as_float();
            if (!o.coat_roughness.is_texture()) mat->clearcoatRoughness = o.coat_roughness.as_float();

            // standard_surface graphs carry their emission weight here.
            const float emission = o.emission_luminance.is_texture() ? 1.f : o.emission_luminance.as_float();
            if (emission > 0.f) {
                if (o.emission_color.is_texture()) {
                    mat->emissiveMap = colorTexture(b, o.emission_color.texture_id);
                    mat->emissive.setRGB(1, 1, 1);
                } else {
                    const auto e = o.emission_color.as_float3();
                    mat->emissive.setRGB(e.x, e.y, e.z);
                }
                mat->emissiveIntensity = emission;
            }
            if (o.normal.is_texture()) mat->normalMap = dataTexture(b, o.normal.texture_id);
            if (!o.opacity.is_texture() && o.opacity.as_float() < 1.f) {
                mat->opacity = o.opacity.as_float();
                mat->transparent = true;
            }
            return mat;
        }

        // OpenPBR (geometry_thin_walled) and Autodesk Standard Surface
        // (thin_walled) both default to a VOLUME: a transmissive surface bounds
        // something refracted in and out. lightusd's render data does not carry
        // the flag, so it is read off the shader prim.
        static bool thinWalledShader(Build& b, const tnext::RenderMaterial& rmat) {
            const auto matPrim = b.stage.GetPrimAtPath(rmat.prim_path);
            if (!matPrim.IsValid()) return false;
            const auto shader = surfaceShader(b.stage, matPrim);
            if (!shader.IsValid()) return false;
            for (const char* name : {"inputs:geometry_thin_walled", "inputs:thin_walled"}) {
                const auto* value = shader.GetPropertyValue(name);
                const bool* thin = value ? value->as_bool() : nullptr;
                if (thin) return *thin;
            }
            return false;
        }

        // threepp says "volume" with a thickness (three.js / glTF: 0 = thin-
        // walled). A volume material takes the smallest side of the first
        // mesh that wears it: GL offsets and absorbs over that distance, the
        // ray tracers trace the mesh and only need it above zero.
        static void sizeVolume(Build& b, const std::shared_ptr<Material>& m, BufferGeometry& g) {
            auto* phys = dynamic_cast<MeshPhysicalMaterial*>(m.get());
            if (!phys || phys->thickness > 0.f || !b.volumeMaterials.count(phys)) return;
            if (!g.boundingBox) g.computeBoundingBox();
            const Vector3 size = g.boundingBox->getSize();
            phys->thickness = std::max(std::min({size.x, size.y, size.z}), 1e-3f);
        }

        // The Material's surface shader: the target of its MDL (or universal)
        // surface output, else its first Shader child.
        static lnext::UsdPrim surfaceShader(const lnext::Stage& stage, const lnext::UsdPrim& matPrim) {
            if (const auto* spec = matPrim.GetPrimSpec()) {
                for (const char* output : {"outputs:mdl:surface", "outputs:surface"}) {
                    const auto* targets = spec->connection(output);
                    if (!targets || targets->empty()) continue;
                    std::string target = targets->front().str();
                    const auto slash = target.rfind('/');
                    const auto dot = target.find('.', slash == std::string::npos ? 0 : slash);
                    if (dot != std::string::npos) target.resize(dot);
                    auto shader = stage.GetPrimAtPath(target);
                    if (shader.IsValid()) return shader;
                }
            }
            for (const auto& child : matPrim.GetChildren()) {
                if (child.GetTypeName() == "Shader") return child;
            }
            return {};
        }

        // The values a material's own .mdl file gives its parameters, by name: textures as file
        // paths, colours, and numbers (a flag is 1 or 0). SimReady assets ship materials of their
        // own as thin OmniPBR wrappers (`export material M(*) = OmniPBR(diffuse_texture:
        // texture_2d("./textures/..."), ...)`) and author none of it on the shader prim.
        struct MdlDefaults {
            std::unordered_map<std::string, std::filesystem::path> textures;
            std::unordered_map<std::string, Color> colors;
            std::unordered_map<std::string, float> numbers;
            std::unordered_map<std::string, std::array<float, 2>> float2s;
        };

        static std::string mdlName(const char* input) {
            const std::string s(input);
            return s.rfind("inputs:", 0) == 0 ? s.substr(7) : s;
        }

        static MdlDefaults mdlDefaults(const lnext::UsdPrim& shader, const std::filesystem::path& anchor, const std::filesystem::path& baseDir) {
            MdlDefaults out;
            const auto* value = shader.GetPropertyValue("info:mdl:sourceAsset");
            const auto raw = value ? value->as_asset_path() : lnext::StringView();
            if (!raw || raw->empty()) return out;
            std::string rel(raw.data(), raw.size());
            std::replace(rel.begin(), rel.end(), '\\', '/');
            std::filesystem::path file;
            for (const auto& base : {anchor, baseDir}) {
                const auto candidate = (base / rel).lexically_normal();
                if (isFile(candidate)) {
                    file = candidate;
                    break;
                }
            }
            // a library material (OmniPBR.mdl itself) is not beside the asset: nothing to read
            if (file.empty()) return out;
            std::ifstream in(file);
            std::stringstream text;
            text << in.rdbuf();
            const std::string s = text.str();

            static const std::regex texture(R"re((\w+)\s*:\s*texture_2d\(\s*"([^"]+)")re");
            static const std::regex color(R"re((\w+)\s*:\s*color\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\))re");
            static const std::regex flag(R"re((\w+)\s*:\s*(true|false)\b)re");
            // MDL writes a float as 0.5, 1., .5 or 90.f
            static const std::string real = R"re(([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)[fF]?)re";
            static const std::regex number(R"re((\w+)\s*:\s*)re" + real + R"re(\s*[,)])re");
            static const std::regex float2(R"re((\w+)\s*:\s*float2\(\s*)re" + real + R"re(\s*(?:,\s*)re" + real + R"re(\s*)?\))re");
            const std::sregex_iterator end;
            for (std::sregex_iterator it(s.begin(), s.end(), texture); it != end; ++it) {
                out.textures[(*it)[1].str()] = (file.parent_path() / (*it)[2].str()).lexically_normal();
            }
            for (std::sregex_iterator it(s.begin(), s.end(), color); it != end; ++it) {
                out.colors[(*it)[1].str()] = Color(std::stof((*it)[2].str()), std::stof((*it)[3].str()), std::stof((*it)[4].str()));
            }
            for (std::sregex_iterator it(s.begin(), s.end(), flag); it != end; ++it) {
                out.numbers[(*it)[1].str()] = (*it)[2].str() == "true" ? 1.f : 0.f;
            }
            for (std::sregex_iterator it(s.begin(), s.end(), number); it != end; ++it) {
                out.numbers[(*it)[1].str()] = std::stof((*it)[2].str());
            }
            for (std::sregex_iterator it(s.begin(), s.end(), float2); it != end; ++it) {
                const float x = std::stof((*it)[2].str());
                out.float2s[(*it)[1].str()] = {x, (*it)[3].matched ? std::stof((*it)[3].str()) : x};// float2(1.0) is (1, 1)
            }
            return out;
        }

        // NVIDIA MDL materials. The input names cover OmniPBR (snake_case),
        // OmniUe4Base / the Omniverse Unreal bridge (PascalCase, Lightwheel
        // exports) and OmniSurface (*_image, Omniverse Automotive Showcase).
        void applyMdlInputs(Build& b, const tnext::RenderMaterial& rmat, MeshStandardMaterial& mat) {
            const auto matPrim = b.stage.GetPrimAtPath(rmat.prim_path);
            if (!matPrim.IsValid()) return;
            const auto shader = surfaceShader(b.stage, matPrim);
            if (!shader.IsValid()) return;

            // Relative asset paths anchor to the layer that authored them.
            std::filesystem::path anchor = b.baseDir;
            if (const auto* spec = shader.GetPrimSpec()) {
                const std::string& dir = lnext::AssetAnchorPath(spec->asset_anchor_id());
                if (!dir.empty()) anchor = dir;
            }

            // A material of its own (a .mdl file next to the asset, wrapping OmniPBR) carries its
            // parameters' values as the defaults in that file, not on the shader prim: what the
            // prim does not author is looked up there by the same name.
            const MdlDefaults defaults = mdlDefaults(shader, anchor, b.baseDir);

            auto tryColor = [&](std::initializer_list<const char*> names) -> std::optional<Color> {
                for (const char* name : names) {
                    float c[3];
                    const auto* value = shader.GetPropertyValue(name);
                    if (value && value->to_float3(c)) return Color(c[0], c[1], c[2]);
                }
                for (const char* name : names) {
                    if (const auto it = defaults.colors.find(mdlName(name)); it != defaults.colors.end()) return it->second;
                }
                return std::nullopt;
            };
            // A flag (enable_opacity, enable_emission, ...) authored on the prim is a bool or an
            // int, which to_float does not read: it reads as 1 or 0.
            auto tryFloat = [&](std::initializer_list<const char*> names) -> std::optional<float> {
                for (const char* name : names) {
                    float f;
                    const auto* value = shader.GetPropertyValue(name);
                    if (!value) continue;
                    if (value->to_float(&f)) return f;
                    if (const bool* flag = value->as_bool()) return *flag ? 1.f : 0.f;
                    if (const int32_t* i = value->as_int()) return static_cast<float>(*i);
                }
                for (const char* name : names) {
                    if (const auto it = defaults.numbers.find(mdlName(name)); it != defaults.numbers.end()) return it->second;
                }
                return std::nullopt;
            };
            auto tryFloat2 = [&](const char* name) -> std::optional<std::array<float, 2>> {
                float f[2];
                if (const auto* value = shader.GetPropertyValue(name); value && value->to_float2(f)) return std::array<float, 2>{f[0], f[1]};
                if (const auto it = defaults.float2s.find(mdlName(name)); it != defaults.float2s.end()) return it->second;
                return std::nullopt;
            };

            // OmniPBR samples its maps with wrap_repeat at uv' = S * R * uv + T: its
            // rotation_translation_scale, with texture_rotate in degrees. About the origin that
            // is the matrix three.js's setUvTransform builds from offset, repeat and rotation.
            // project_uvw projects the maps from 3D, which no transform of the UVs gives, so it
            // keeps them as they are. The texture loader shares a texture by path, so each map
            // is a clone, shared by the materials that place it the same way.
            const auto scale = tryFloat2("inputs:texture_scale").value_or(std::array<float, 2>{1.f, 1.f});
            const auto translate = tryFloat2("inputs:texture_translate").value_or(std::array<float, 2>{0.f, 0.f});
            const float rotate = tryFloat({"inputs:texture_rotate"}).value_or(0.f);
            const bool placed = tryFloat({"inputs:project_uvw"}).value_or(0.f) <= 0.f;
            auto place = [&](const std::filesystem::path& path) -> std::shared_ptr<Texture> {
                std::ostringstream key;
                key << path.string() << '|';
                if (placed) key << scale[0] << ',' << scale[1] << ',' << translate[0] << ',' << translate[1] << ',' << rotate;
                if (auto cached = mdlTextures[key.str()].lock()) return cached;
                auto source = texLoader.load(path, true);
                if (!source) return nullptr;
                auto tex = source->clone();
                // Texture::copy leaves these out
                tex->name = source->name;
                tex->sourceFile = source->sourceFile;
                tex->encodedSource = source->encodedSource;
                tex->wrapS = TextureWrapping::Repeat;
                tex->wrapT = TextureWrapping::Repeat;
                if (placed) {
                    tex->offset.set(translate[0], translate[1]);
                    tex->repeat.set(scale[0], scale[1]);
                    tex->rotation = math::degToRad(rotate);
                    tex->center.set(0, 0);
                    tex->matrixAutoUpdate = true;
                    tex->updateMatrix();
                }
                mdlTextures[key.str()] = tex;
                return tex;
            };
            auto tryTexture = [&](std::initializer_list<const char*> names) -> std::shared_ptr<Texture> {
                for (const char* name : names) {
                    const auto* value = shader.GetPropertyValue(name);
                    const auto raw = value ? value->as_asset_path() : lnext::StringView();
                    if (!raw || raw->empty()) continue;
                    if (hasUDIM(*raw)) b.sawUDIM = true;
                    std::string rel = resolveUDIM(*raw, b.udimTile);
                    std::replace(rel.begin(), rel.end(), '\\', '/');
                    for (const auto& base : {anchor, b.baseDir}) {
                        const auto candidate = (base / rel).lexically_normal();
                        if (isFile(candidate)) return place(candidate);
                    }
                    if (isFile(rel)) return place(rel);
                }
                for (const char* name : names) {
                    const auto it = defaults.textures.find(mdlName(name));
                    if (it == defaults.textures.end()) continue;
                    const std::string file = it->second.string();
                    if (hasUDIM(file)) b.sawUDIM = true;
                    const std::filesystem::path tilePath = resolveUDIM(file, b.udimTile);
                    if (isFile(tilePath)) return place(tilePath);
                }
                return nullptr;
            };

            if (auto t = tryTexture({"inputs:diffuse_texture", "inputs:albedo_texture",
                                     "inputs:Diffuse_Texture", "inputs:Albedo_Texture",
                                     "inputs:BaseColor_Texture", "inputs:base_color_image",
                                     "inputs:diffuse_color_image"})) {
                t->colorSpace = ColorSpace::sRGB;
                mat.map = t;
                mat.color.setRGB(1, 1, 1);
            } else if (auto c = tryColor({"inputs:Diffuse_Color", "inputs:diffuse_color_constant",
                                          "inputs:diffuse_reflection_color"})) {
                mat.color = *c;
            }
            if (auto t = tryTexture({"inputs:normalmap_texture", "inputs:normal_texture",
                                     "inputs:Normal_Texture", "inputs:NormalMap_Texture",
                                     "inputs:Bump_Texture", "inputs:normal_image"})) {
                t->colorSpace = ColorSpace::Linear;
                mat.normalMap = t;
            }
            // Roughness: ORM-packed first, then dedicated maps.
            if (auto t = tryTexture({"inputs:ORM_texture"})) {
                t->colorSpace = ColorSpace::Linear;
                mat.aoMap = t;
                mat.roughnessMap = t;
                mat.metalnessMap = t;
                mat.roughness = 1.f;
                mat.metalness = 1.f;
            } else {
                if (auto t = tryTexture({"inputs:roughness_texture", "inputs:Roughness_Texture",
                                         "inputs:roughness_image"})) {
                    t->colorSpace = ColorSpace::Linear;
                    mat.roughnessMap = t;
                    mat.roughness = 1.f;
                } else if (auto v = tryFloat({"inputs:Roughness_Color", "inputs:reflection_roughness_constant",
                                              "inputs:roughness_range_position"})) {
                    mat.roughness = *v;
                }
                if (auto t = tryTexture({"inputs:metallic_texture", "inputs:Metallic_Texture",
                                         "inputs:metalness_image", "inputs:metallic_image"})) {
                    t->colorSpace = ColorSpace::Linear;
                    mat.metalnessMap = t;
                    mat.metalness = 1.f;
                } else if (auto v = tryFloat({"inputs:Metallic_Color", "inputs:metallic_constant",
                                              "inputs:metalness"})) {
                    mat.metalness = *v;
                }
                // OmniSurface "cavity" map ~= ambient occlusion.
                if (auto t = tryTexture({"inputs:ao_image", "inputs:cavity_image"})) {
                    t->colorSpace = ColorSpace::Linear;
                    mat.aoMap = t;
                }
            }
            if (auto t = tryTexture({"inputs:emissive_texture", "inputs:Emissive_Texture",
                                     "inputs:Emission_Texture", "inputs:emissive_image",
                                     "inputs:emission_image", "inputs:emission_color_image"})) {
                t->colorSpace = ColorSpace::sRGB;
                mat.emissiveMap = t;
                mat.emissive.setRGB(1, 1, 1);
            } else if (tryFloat({"inputs:enable_emission"}).value_or(0.f) > 0.f) {
                if (auto c = tryColor({"inputs:emissive_color"})) mat.emissive = *c;
                mat.emissiveIntensity = tryFloat({"inputs:emissive_intensity"}).value_or(1.f) / 1000.f;
            }
            // OmniPBR's opacity: a mask (cut out below its threshold, as a decal's edges are), or
            // one value
            const bool opacity = tryFloat({"inputs:enable_opacity"}).value_or(0.f) > 0.f;
            if (opacity && tryFloat({"inputs:enable_opacity_texture"}).value_or(0.f) > 0.f) {
                if (auto t = tryTexture({"inputs:opacity_texture"})) {
                    t->colorSpace = ColorSpace::Linear;
                    mat.alphaMap = t;
                    mat.alphaTest = std::max(0.01f, tryFloat({"inputs:opacity_threshold"}).value_or(0.5f));
                }
            }
            if (auto v = tryFloat({"inputs:opacity_constant"}); v && *v < 1.f && opacity) {
                mat.opacity = *v;
                mat.transparent = true;
            }
        }

        // -------------------------------------------------------------------
        // UsdPhysics bodies, joints and drives → USDArticulationDesc.
        //
        // The joint walk in buildRobot has already reduced every joint to a
        // frame on body0 and body1's pose at the joint's zero, so the tree is
        // rebuilt from those: links come out parents first and at zero, which
        // is the pose a reduced-coordinate articulation is built in. USD's
        // angular drive gains and targets are per degree.
        // -------------------------------------------------------------------

        struct BuiltJoint {
            const tnext::PhysicsJointAnnotation* usd;
            std::shared_ptr<Object3D> node;
            Robot::JointInfo info;
            Matrix4 frame;// joint frame in body0
            Matrix4 local;// body1 in the joint frame, at the joint's zero
            float authored;
        };

        static std::optional<float> floatProperty(const lnext::UsdPrim& prim, const std::string& name) {
            const auto* value = prim.GetPropertyValue(name);
            const float* f = value ? value->as_float() : nullptr;
            return f ? std::optional<float>(*f) : std::nullopt;
        }

        static void readJoint(Build& b, const BuiltJoint& j, USDArticulationDesc::Link& link) {
            const bool revolute = j.info.type == Robot::JointType::Revolute;
            const float perDegree = revolute ? 180.f / math::PI : 1.f;
            link.jointName = j.info.name;
            link.jointType = j.info.type;
            link.jointNode = j.node;
            link.range = j.info.range;
            link.jointValue = j.authored;
            link.jointPath = j.usd->prim_path;

            const auto prim = b.stage.GetPrimAtPath(j.usd->prim_path);
            if (!prim.IsValid() || j.info.type == Robot::JointType::Fixed) return;

            const std::string drive = std::string("drive:") + (revolute ? "angular" : "linear") + ":physics:";
            const auto stiffness = floatProperty(prim, drive + "stiffness");
            const auto damping = floatProperty(prim, drive + "damping");
            if (stiffness || damping) {
                USDArticulationDesc::Drive d;
                d.stiffness = stiffness.value_or(0.f) * perDegree;
                d.damping = damping.value_or(0.f) * perDegree;
                d.maxForce = floatProperty(prim, drive + "maxForce").value_or(std::numeric_limits<float>::infinity());
                d.target = floatProperty(prim, drive + "targetPosition").value_or(0.f) / perDegree;
                const auto* type = prim.GetPropertyValue(drive + "type");
                const auto token = type ? type->as_token() : lnext::StringView();
                d.acceleration = token && *token == "acceleration";
                link.drive = d;
            }
            if (const auto v = floatProperty(prim, "physxJoint:maxJointVelocity")) link.maxJointVelocity = *v / perDegree;
            link.armature = floatProperty(prim, "physxJoint:armature");
            link.jointFriction = floatProperty(prim, "physxJoint:jointFriction").value_or(0.f);
        }

        static void readBody(Build& b, USDArticulationDesc::Link& link) {
            const auto& bodies = b.scene.physics.rigid_bodies;
            const auto body = std::ranges::find(bodies, link.path, &tnext::PhysicsRigidBodyAnnotation::prim_path);
            if (body == bodies.end() || !body->has_mass) return;
            link.mass = body->mass;
            link.density = body->density;
            const auto& com = body->center_of_mass;
            if (std::isfinite(com.x) && std::isfinite(com.y) && std::isfinite(com.z)) link.centerOfMass = Vector3(com.x, com.y, com.z);
            const auto& inertia = body->diagonal_inertia;
            if (inertia.x > 0 && inertia.y > 0 && inertia.z > 0) link.diagonalInertia = Vector3(inertia.x, inertia.y, inertia.z);
            // Unauthored principalAxes are the invalid (0, 0, 0, 0).
            const auto& axes = body->principal_axes;
            if (axes.x != 0 || axes.y != 0 || axes.z != 0 || axes.w != 0) link.principalAxes.set(axes.y, axes.z, axes.w, axes.x);
        }

        // Each collider goes to its nearest body up the prim tree. Its mesh is
        // expressed in the link frame: rigid, so the body's scale goes into the
        // points. `bodyWorld` and `stageToWorld` are in the authored pose,
        // which is where the collider prims' own transforms hold.
        static void readColliders(Build& b, USDArticulationDesc& desc,
                                  const std::unordered_map<std::string, Matrix4>& bodyWorld,
                                  const Matrix4& stageToWorld) {
            std::unordered_map<std::string, size_t> linkOf;
            for (size_t i = 0; i < desc.links.size(); ++i) linkOf.emplace(desc.links[i].path, i);

            for (const auto& c : b.scene.physics.colliders) {
                if (!c.collision_enabled) continue;
                auto owner = linkOf.end();
                for (std::string path = c.prim_path; !path.empty() && owner == linkOf.end(); path.resize(path.rfind('/'))) {
                    owner = linkOf.find(path);
                }
                if (owner == linkOf.end()) continue;
                auto& link = desc.links[owner->second];

                const auto node = b.scene.node_by_path.find(c.prim_path);
                const auto* mesh = node == b.scene.node_by_path.end() ? nullptr : b.scene.get_mesh(b.scene.nodes[node->second].data_id);
                if (!mesh || mesh->points.size() < 4 * 3) {
                    desc.diagnostics.push_back(c.prim_path + ": a collider without a mesh is left out");
                    continue;
                }
                if (c.approximation == "convexDecomposition" || c.approximation == "meshSimplification" || c.approximation == "sdf") {
                    desc.diagnostics.push_back(c.prim_path + ": approximation " + c.approximation + " is cooked as one convex hull");
                }

                Matrix4 inverseBody = bodyWorld.at(link.path);
                inverseBody.invert();
                Matrix4 toLink = Matrix4().makeScale(link.scale.x, link.scale.y, link.scale.z);
                toLink.multiply(inverseBody).multiply(stageToWorld).multiply(toMatrix4(b.scene.nodes[node->second].world_transform));

                USDArticulationDesc::Collider collider;
                collider.path = c.prim_path;
                collider.approximation = c.approximation;
                collider.points.resize(mesh->points.size());
                mesh->points.copy_to(collider.points.data());
                for (size_t i = 0; i + 2 < collider.points.size(); i += 3) {
                    Vector3 p(collider.points[i], collider.points[i + 1], collider.points[i + 2]);
                    p.applyMatrix4(toLink);
                    collider.points[i] = p.x;
                    collider.points[i + 1] = p.y;
                    collider.points[i + 2] = p.z;
                }

                const auto bound = lnext::GetInheritedBoundMaterialPathForPurpose(b.stage, c.prim_path, "physics");
                const auto& materials = b.scene.physics.materials;
                if (const auto m = std::ranges::find(materials, bound, &tnext::PhysicsMaterialAnnotation::prim_path); m != materials.end()) {
                    collider.hasMaterial = true;
                    collider.staticFriction = m->static_friction;
                    collider.dynamicFriction = m->dynamic_friction;
                    collider.restitution = m->restitution;
                }
                link.colliders.push_back(std::move(collider));
            }
        }

        // PhysxMimicJointAPI couples two joints: `physxMimicJoint:<axis>:*` on the
        // coupled joint, its reference a relationship. The rule PhysX enforces is
        // q + gearing·qRef + offset = 0, authored with USD's angles in degrees; the
        // description is in radians and metres, so a revolute joint's offset is
        // converted, and so is the gearing between a revolute and a prismatic
        // joint. The API's instance axis names the coupled joint's DOF, of which a
        // revolute or prismatic joint has exactly one: its kind (rot or trans) is
        // checked, and the DOF is the joint's own.
        static void readMimics(Build& b, USDArticulationDesc& desc) {
            static const std::string api = "physxMimicJoint:";
            const auto linkMovedBy = [&](const std::string& jointPath) {
                for (size_t i = 0; i < desc.links.size(); ++i) {
                    if (desc.links[i].parent >= 0 && desc.links[i].jointPath == jointPath) return static_cast<int>(i);
                }
                return -1;
            };
            const auto perDegree = [](Robot::JointType type) {
                return type == Robot::JointType::Revolute ? 180.f / math::PI : 1.f;
            };
            const auto kindOf = [](Robot::JointType type) {
                return std::string(type == Robot::JointType::Revolute ? "rot" : "trans");
            };

            for (size_t i = 0; i < desc.links.size(); ++i) {
                const auto& link = desc.links[i];
                if (link.parent < 0 || link.jointPath.empty()) continue;
                const auto prim = b.stage.GetPrimAtPath(link.jointPath);
                if (!prim.IsValid()) continue;

                std::vector<std::string> instances;
                const auto collect = [&](const std::string& name) {
                    if (name.rfind(api, 0) != 0) return;
                    const auto end = name.find(':', api.size());
                    if (end == std::string::npos) return;
                    const auto instance = name.substr(api.size(), end - api.size());
                    if (std::ranges::find(instances, instance) == instances.end()) instances.push_back(instance);
                };
                for (const auto& name : prim.GetPropertyNames()) collect(name);
                for (const auto& name : prim.GetRelationshipNames()) collect(name);

                for (const auto& instance : instances) {
                    const std::string prefix = api + instance + ":";
                    const auto skip = [&](const std::string& why) {
                        desc.diagnostics.push_back(link.jointPath + ": mimic coupling " + why);
                    };
                    if (link.jointType == Robot::JointType::Fixed) {
                        skip("on a fixed joint is ignored");
                        continue;
                    }
                    if (instance.rfind(kindOf(link.jointType), 0) != 0) {
                        skip("names the " + instance + " axis, which a " +
                             (link.jointType == Robot::JointType::Revolute ? "revolute" : "prismatic") + " joint has not");
                        continue;
                    }
                    const auto* targets = prim.GetRelationship(prefix + "referenceJoint");
                    if (!targets || targets->empty()) {
                        skip("names no reference joint");
                        continue;
                    }
                    const std::string reference = targets->front().str();
                    const int ref = linkMovedBy(reference);
                    if (ref < 0 || desc.links[ref].jointType == Robot::JointType::Fixed) {
                        skip("follows " + reference + ", which is not a moving joint of this articulation");
                        continue;
                    }
                    if (ref == static_cast<int>(i)) {
                        skip("follows itself");
                        continue;
                    }
                    const auto* axisValue = prim.GetPropertyValue(prefix + "referenceJointAxis");
                    const auto axisToken = axisValue ? axisValue->as_token() : lnext::StringView();
                    if (axisToken && axisToken->rfind(kindOf(desc.links[ref].jointType), 0) != 0) {
                        skip("follows the " + *axisToken + " axis of " + reference + ", which that joint has not");
                        continue;
                    }
                    const float gearing = floatProperty(prim, prefix + "gearing").value_or(1.f);
                    if (gearing == 0.f) {
                        skip("has a gearing of zero");
                        continue;
                    }
                    USDArticulationDesc::Mimic m;
                    m.joint = link.jointPath;
                    m.referenceJoint = reference;
                    m.link = static_cast<int>(i);
                    m.referenceLink = ref;
                    const float k = perDegree(link.jointType);
                    m.gearing = gearing * perDegree(desc.links[ref].jointType) / k;
                    m.offset = floatProperty(prim, prefix + "offset").value_or(0.f) / k;
                    m.naturalFrequency = floatProperty(prim, prefix + "naturalFrequency").value_or(0.f);
                    m.dampingRatio = floatProperty(prim, prefix + "dampingRatio").value_or(0.f);
                    desc.mimics.push_back(std::move(m));
                }

                // NewtonMimicAPI, the Newton engine's own coupling (all that the SimReady
                // AgiBot G2 authors for its grippers' linkages): `newton:mimicJoint` names the
                // leader, and the follower holds q = coef0 + coef1·qLeader in the joints' USD
                // units. That is the rule above with gearing -coef1 and offset -coef0. A joint
                // that also authors a PhysX coupling keeps that one.
                if (!instances.empty()) continue;
                const auto* leader = prim.GetRelationship("newton:mimicJoint");
                if (!leader || leader->empty()) continue;
                const auto skip = [&](const std::string& why) {
                    desc.diagnostics.push_back(link.jointPath + ": Newton mimic coupling " + why);
                };
                if (link.jointType == Robot::JointType::Fixed) {
                    skip("on a fixed joint is ignored");
                    continue;
                }
                const std::string reference = leader->front().str();
                const int ref = linkMovedBy(reference);
                if (ref < 0 || desc.links[ref].jointType == Robot::JointType::Fixed) {
                    skip("follows " + reference + ", which is not a moving joint of this articulation");
                    continue;
                }
                if (ref == static_cast<int>(i)) {
                    skip("follows itself");
                    continue;
                }
                const float coef1 = floatProperty(prim, "newton:mimicCoef1").value_or(1.f);
                if (coef1 == 0.f) {
                    skip("has a coefficient of zero");
                    continue;
                }
                USDArticulationDesc::Mimic m;
                m.joint = link.jointPath;
                m.referenceJoint = reference;
                m.link = static_cast<int>(i);
                m.referenceLink = ref;
                const float k = perDegree(link.jointType);
                m.gearing = -coef1 * perDegree(desc.links[ref].jointType) / k;
                m.offset = -floatProperty(prim, "newton:mimicCoef0").value_or(0.f) / k;
                desc.mimics.push_back(std::move(m));
            }
        }

        static USDArticulationDesc describeArticulation(Build& b, const std::vector<BuiltJoint>& joints,
                                                        const std::unordered_map<std::string, Matrix4>& world,
                                                        const Matrix4& stageToWorld, bool anchored,
                                                        std::vector<std::string> diagnostics) {
            USDArticulationDesc desc;
            desc.fixedBase = anchored;
            desc.diagnostics = std::move(diagnostics);
            if (b.scene.meters_per_unit != 1.f) {
                desc.diagnostics.push_back("metersPerUnit is " + std::to_string(b.scene.meters_per_unit) + ": lengths are taken as authored");
            }

            // A second joint into a body closes a loop, which a tree cannot hold.
            std::unordered_set<std::string> driven;
            std::unordered_map<std::string, std::vector<const BuiltJoint*>> outbound;
            for (const auto& j : joints) {
                if (driven.insert(j.usd->body1).second) outbound[j.usd->body0].push_back(&j);
                else desc.diagnostics.push_back(j.usd->prim_path + ": closes a kinematic loop and is left out");
            }
            std::vector<std::string> roots;
            for (const auto& j : joints) {
                if (!driven.count(j.usd->body0) && std::ranges::find(roots, j.usd->body0) == roots.end()) roots.push_back(j.usd->body0);
            }
            if (roots.empty()) {
                desc.diagnostics.push_back("no root body: every body is moved by a joint");
                return desc;
            }
            if (roots.size() > 1) desc.diagnostics.push_back("several root bodies: only " + roots.front() + " is described");

            struct Pending {
                std::string path;
                int parent;
                const BuiltJoint* joint;
                Matrix4 jointWorld;
                Matrix4 linkWorld;
            };
            std::vector<Pending> pending{{roots.front(), -1, nullptr, Matrix4(), world.at(roots.front())}};
            for (size_t i = 0; i < pending.size(); ++i) {
                const Pending item = pending[i];

                USDArticulationDesc::Link link;
                link.path = item.path;
                link.body = b.objectByPath.at(item.path);
                link.parent = item.parent;
                item.linkWorld.decompose(link.position, link.rotation, link.scale);
                readBody(b, link);
                if (item.joint) {
                    readJoint(b, *item.joint, link);
                    Vector3 jointScale;
                    Quaternion jointRotation;
                    item.jointWorld.decompose(link.jointAnchor, jointRotation, jointScale);
                    link.jointAxis.copy(item.joint->info.axis).applyQuaternion(jointRotation).normalize();
                }
                desc.links.push_back(std::move(link));

                for (const auto* j : outbound[item.path]) {
                    Matrix4 jointWorld;
                    jointWorld.multiplyMatrices(item.linkWorld, j->frame);
                    Matrix4 childWorld;
                    childWorld.multiplyMatrices(jointWorld, j->local);
                    pending.push_back({j->usd->body1, static_cast<int>(i), j, jointWorld, childWorld});
                }
            }

            readMimics(b, desc);
            readColliders(b, desc, world, stageToWorld);

            for (const auto& path : b.scene.physics.articulation_roots) {
                const auto prim = b.stage.GetPrimAtPath(path);
                if (!prim.IsValid()) continue;
                if (const auto* v = prim.GetPropertyValue("physxArticulation:enabledSelfCollisions"); v && v->as_bool()) desc.selfCollision = *v->as_bool();
                if (const auto* v = prim.GetPropertyValue("physxArticulation:solverPositionIterationCount"); v && v->as_int()) desc.solverPositionIterations = *v->as_int();
            }
            return desc;
        }

        // -------------------------------------------------------------------
        // UsdPhysics joints → Robot.
        //
        // A joint relates two bodies: its frame sits at localPos0/localRot0
        // in body0 and at localPos1/localRot1 in body1, and the joint's zero
        // is where those two frames coincide; it moves body1 about
        // (revolute) or along (prismatic) its axis in that frame. Each joint
        // becomes a node under body0 at the body0 frame, with body1 moved
        // beneath it. Assets are authored in some pose (SimReady's FR3 in
        // Franka's "ready" pose), so the authored joint value is measured
        // from the bodies' poses and set on the robot: it loads as authored
        // and reports true joint angles. Revolute limits are degrees in USD.
        // -------------------------------------------------------------------

        static std::shared_ptr<Robot> buildRobot(Build& b, const std::shared_ptr<Group>& root,
                                                 std::optional<USDArticulationDesc>& articulation) {
            struct Joint {
                const tnext::PhysicsJointAnnotation* usd;
                Robot::JointType type;
            };
            std::vector<Joint> joints;
            std::vector<std::string> diagnostics;
            bool anchored = false;
            for (const auto& j : b.scene.physics.joints) {
                // A variant switches a joint off with `active = false` (SimReady's
                // parallel-grip grippers deactivate their five-bar joints); the
                // annotation still lists it.
                if (const auto prim = b.stage.GetPrimAtPath(j.prim_path); prim.IsValid() && !prim.IsActive()) continue;
                Robot::JointType type;
                if (j.type_name == "PhysicsRevoluteJoint") type = Robot::JointType::Revolute;
                else if (j.type_name == "PhysicsPrismaticJoint") type = Robot::JointType::Prismatic;
                else if (j.type_name == "PhysicsFixedJoint") type = Robot::JointType::Fixed;
                else {// spherical, distance, D6: no Robot equivalent
                    diagnostics.push_back(j.prim_path + ": " + j.type_name + " is not represented");
                    continue;
                }
                // A joint to the world (no body0) anchors the root; nothing to articulate.
                if (!j.has_body0 || !j.has_body1) {
                    if (type == Robot::JointType::Fixed) anchored = true;
                    else diagnostics.push_back(j.prim_path + ": a moving joint to the world is not represented");
                    continue;
                }
                if (!b.objectByPath.count(j.body0) || !b.objectByPath.count(j.body1)) continue;
                // body1 must not contain body0, or moving it under the joint
                // would make a cycle.
                if (j.body0 == j.body1 || j.body0.rfind(j.body1 + "/", 0) == 0) continue;
                joints.push_back({&j, type});
            }
            if (joints.empty()) return nullptr;

            // Parents before children, so DOF order runs root to tip.
            std::ranges::stable_sort(joints, {}, [](const Joint& j) {
                return std::ranges::count(j.usd->body1, '/');
            });

            // Rest poses, read before any body moves.
            root->updateMatrixWorld(true);
            std::unordered_map<std::string, Matrix4> world;
            for (const auto& j : joints) {
                for (const auto* path : {&j.usd->body0, &j.usd->body1}) {
                    world.emplace(*path, *b.objectByPath.at(*path)->matrixWorld);
                }
            }

            auto robot = Robot::create();
            std::vector<float> authoredValues;// parallel to the robot's joints
            std::vector<BuiltJoint> built;
            std::unordered_set<std::string> links;
            auto addLink = [&](const std::string& path) {
                if (links.insert(path).second) robot->addLink(b.objectByPath.at(path));
            };

            for (const auto& j : joints) {
                const auto& usd = *j.usd;
                const auto body0 = b.objectByPath.at(usd.body0);
                const auto body1 = b.objectByPath.at(usd.body1);

                const Vector3 pos0(usd.local_pos0.x, usd.local_pos0.y, usd.local_pos0.z);
                const Quaternion rot0(usd.local_rot0.y, usd.local_rot0.z, usd.local_rot0.w, usd.local_rot0.x);
                Matrix4 frame;
                frame.compose(pos0, rot0, Vector3(1, 1, 1));

                auto jointNode = std::make_shared<Object3D>();
                jointNode->name = usd.prim_path.substr(usd.prim_path.rfind('/') + 1);
                jointNode->position.copy(pos0);
                jointNode->quaternion.copy(rot0);
                body0->add(jointNode);

                Robot::JointInfo info;
                info.type = j.type;
                info.name = jointNode->name;
                info.parent = usd.body0;
                info.child = usd.body1;
                info.axis.set(usd.axis.x, usd.axis.y, usd.axis.z);
                if (info.axis.lengthSq() > 0) info.axis.normalize();
                const bool limited = std::isfinite(usd.lower_limit) && std::isfinite(usd.upper_limit) &&
                                     usd.lower_limit <= usd.upper_limit;
                if (limited && j.type == Robot::JointType::Revolute) {
                    info.range = Robot::JointRange{math::degToRad(usd.lower_limit), math::degToRad(usd.upper_limit)};
                } else if (limited && j.type == Robot::JointType::Prismatic) {
                    info.range = Robot::JointRange{usd.lower_limit, usd.upper_limit};
                }

                // body1 in the body0 joint frame, from the authored poses.
                Matrix4 body0Inv = world.at(usd.body0);
                body0Inv.invert();
                Matrix4 frameInv = frame;
                frameInv.invert();
                Matrix4 relative;
                relative.multiplyMatrices(frameInv, body0Inv.multiply(world.at(usd.body1)));

                // The joint motion that takes the body1 frame onto the body0
                // frame: relative * frame1 is a rotation about (or a
                // translation along) the axis by the authored value.
                const Vector3 pos1(usd.local_pos1.x, usd.local_pos1.y, usd.local_pos1.z);
                const Quaternion rot1(usd.local_rot1.y, usd.local_rot1.z, usd.local_rot1.w, usd.local_rot1.x);
                Matrix4 frame1;
                frame1.compose(pos1, rot1, Vector3(1, 1, 1));
                Matrix4 motion;
                motion.multiplyMatrices(relative, frame1);
                Vector3 motionPos, motionScale;
                Quaternion motionRot;
                motion.decompose(motionPos, motionRot, motionScale);

                float authored = 0;
                Matrix4 undo;// the inverse of the joint motion at the authored value
                if (j.type == Robot::JointType::Revolute) {
                    const float s = motionRot.x * info.axis.x + motionRot.y * info.axis.y + motionRot.z * info.axis.z;
                    authored = 2 * std::atan2(s, motionRot.w);
                    if (authored > math::PI) authored -= 2 * math::PI;
                    if (authored < -math::PI) authored += 2 * math::PI;
                    // Ranges like FR3's joint 6 (25..265 deg) run past pi.
                    if (info.range) {
                        for (float wrapped : {authored + 2 * math::PI, authored - 2 * math::PI}) {
                            if (!(authored >= info.range->min && authored <= info.range->max) &&
                                wrapped >= info.range->min && wrapped <= info.range->max) authored = wrapped;
                        }
                    }
                    undo.makeRotationAxis(info.axis, -authored);
                } else if (j.type == Robot::JointType::Prismatic) {
                    authored = motionPos.dot(info.axis);
                    undo.makeTranslation(info.axis.x * -authored, info.axis.y * -authored, info.axis.z * -authored);
                }
                // body1 at the joint's zero, keeping whatever the authored pose
                // has beyond the joint motion so the loaded pose is exact.
                Matrix4 local;
                local.multiplyMatrices(undo, relative);
                jointNode->add(body1);
                local.decompose(body1->position, body1->quaternion, body1->scale);

                addLink(usd.body0);
                addLink(usd.body1);
                robot->addJoint(jointNode, info);
                authoredValues.push_back(authored);
                built.push_back({&usd, jointNode, info, frame, local, authored});
            }

            articulation = describeArticulation(b, built, world, *root->matrixWorld, anchored, std::move(diagnostics));

            robot->add(root);
            robot->finalizeInPlace();
            const auto& nodes = robot->jointNodes();
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (const int dof = robot->jointDof(i); dof >= 0) {
                    robot->setJointValue(static_cast<size_t>(dof), authoredValues[i]);
                }
            }
            return robot;
        }

        // -------------------------------------------------------------------
        // UsdLux → threepp lights, placed in stage space under the root.
        //
        //   DistantLight  → DirectionalLight (sun-like, parallel rays)
        //   DomeLight     → AmbientLight; its texture, if any, comes back
        //                   through USDResult.environment
        //   SphereLight   → PointLight (radius ignored: three.js doesn't model
        //                   point-light area)
        //   RectLight     → RectAreaLight (width/height in metres)
        //   DiskLight     → SpotLight (with a wide cone — closest approximation)
        //
        // Intensity is USD's `intensity * 2^exposure` times `color`. UsdLux
        // lights shine along their local -Z.
        // -------------------------------------------------------------------

        void addLights(Build& b, Group& root) {
            for (const auto& l : b.scene.lights) {
                const Matrix4 wm = toMatrix4(l.transform);
                const Color color(l.color.x, l.color.y, l.color.z);
                const float intensity = l.intensity * std::pow(2.0f, l.exposure);
                Vector3 pos, scale;
                Quaternion rot;
                wm.decompose(pos, rot, scale);

                switch (l.type) {
                    case tnext::LightType::Directional: {
                        auto light = DirectionalLight::create(color, intensity);
                        // Place the light at +Z (away from its default origin
                        // target) so its rays travel along the rotated -Z.
                        Vector3 dir(0, 0, -1);
                        dir.transformDirection(wm).normalize();
                        light->position.set(-dir.x, -dir.y, -dir.z);
                        root.add(light);
                        break;
                    }
                    case tnext::LightType::Dome: {
                        // A dome covers the full sphere; half intensity so it
                        // doesn't double up with a scene.environment HDR set
                        // externally.
                        root.add(AmbientLight::create(color, intensity * 0.5f));
                        const int32_t imageId = l.params.dome.texture_id;
                        if (!_envTexture && imageId >= 0 && static_cast<size_t>(imageId) < b.scene.images.size()) {
                            const std::filesystem::path texPath = b.scene.images[imageId].resolved_path;
                            if (isFile(texPath)) {
                                const auto ext = lowercaseExtension(texPath);
                                if (ext == ".hdr") {
                                    _envTexture = RGBELoader().load(texPath, true);
                                } else if (ext == ".exr") {
                                    _envTexture = EXRLoader().load(texPath, true);
                                } else {
                                    _envTexture = texLoader.load(texPath, true);
                                }
                            }
                        }
                        break;
                    }
                    case tnext::LightType::Sphere:
                    case tnext::LightType::Point: {
                        auto light = PointLight::create(color, intensity);
                        light->position.copy(pos);
                        root.add(light);
                        break;
                    }
                    case tnext::LightType::Rect: {
                        auto light = RectAreaLight::create(color, intensity, l.params.rect.width, l.params.rect.height);
                        light->position.copy(pos);
                        light->quaternion.copy(rot);
                        root.add(light);
                        break;
                    }
                    case tnext::LightType::Disk:
                    case tnext::LightType::Spot: {
                        auto light = SpotLight::create(color, intensity, /*distance*/ 0.f,
                                                       /*angle*/ math::PI / 4.f, /*penumbra*/ 0.5f, /*decay*/ 1.f);
                        light->position.copy(pos);
                        // Aim along the rotated local -Z via a target object. The
                        // target is the light's own child (the light has no
                        // rotation, so its local position is the aim), as a glTF
                        // spot light's is: a clone of the stage then aims at its
                        // own copy, not at the original's, which dies with it.
                        Vector3 dir(0, 0, -1);
                        dir.applyQuaternion(rot).normalize();
                        auto tgt = std::make_shared<Object3D>();
                        tgt->position.copy(dir);
                        light->add(tgt);
                        light->setTarget(*tgt);
                        root.add(light);
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    };

    USDLoader::USDLoader(): pimpl_(std::make_unique<Impl>()) {}
    USDLoader::~USDLoader() = default;

    USDLoader& USDLoader::setIgnoreUpDirection(bool ignore) {
        pimpl_->ignoreUpDirection = ignore;
        return *this;
    }

    USDLoader& USDLoader::setVariant(const std::string& set, const std::string& value) {
        pimpl_->variants[set] = value;
        return *this;
    }

    std::shared_ptr<Group> USDLoader::load(const std::filesystem::path& path) {
        return pimpl_->load(path);
    }

    USDResult USDLoader::loadFull(const std::filesystem::path& path) {
        auto group = pimpl_->load(path);
        return {std::move(group), {}, pimpl_->_envTexture, pimpl_->_robot, pimpl_->_articulation};
    }

}// namespace threepp
