#include "threepp/loaders/USDLoader.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include "tinyusdz.hh"
#include "composition.hh"
#include "asset-resolution.hh"
#include "io-util.hh"
#include "stage.hh"
#include "tydra/render-data.hh"
#include "tydra/scene-access.hh"
#include "usdLux.hh"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp {

    namespace {

        // -----------------------------------------------------------------------
        // HTTP/HTTPS asset resolution.
        //
        // Real-world Omniverse / NVIDIA Showcases scenes embed payload paths
        // pointing into the public Nucleus mirror
        // (`https://omniverse-content-production.s3.us-west-2.amazonaws.com/...`).
        // tinyusdz can't fetch those by default, so we register a wildcard
        // AssetResolutionHandler that:
        //   - For http(s) URLs: download via `curl` once (built into Win 10+,
        //     macOS, all current Linux distros) into a temp on-disk cache,
        //     return that local path. Subsequent resolves hit the cache.
        //   - For local paths: replicate tinyusdz's own io::FindFile lookup
        //     against the resolver's search paths so we don't disable normal
        //     file resolution by claiming the wildcard slot.
        //
        // size_fun / read_fun work on the resolved local path (cache file or
        // the file the search-paths lookup found), so they're shared between
        // both cases.
        // -----------------------------------------------------------------------

        bool isHttpUrl(const std::string& s) {
            return s.compare(0, 7, "http://") == 0 ||
                   s.compare(0, 8, "https://") == 0;
        }

        std::filesystem::path httpCacheDir() {
            std::filesystem::path p =
                std::filesystem::temp_directory_path() / "threepp_usd_cache";
            std::error_code ec;
            std::filesystem::create_directories(p, ec);
            return p;
        }

        // Hash a URL into a stable cache filename. Preserves the URL's
        // extension so tinyusdz / our loader can format-detect from it.
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
                std::filesystem::remove(dest, ec);  // don't leave a partial file in cache
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

        // FSResolveAsset: route URLs to the cache, locals to FindFile.
        int httpAwareResolve(const char* asset_name,
                             const std::vector<std::string>& search_paths,
                             std::string* resolved_asset_name,
                             std::string* err,
                             void* /*userdata*/) {
            if (!asset_name || !resolved_asset_name) return -1;
            const std::string path = asset_name;
            if (isHttpUrl(path)) {
                auto cache = httpCacheDir() / urlToCacheName(path);
                if (!std::filesystem::exists(cache)) {
                    std::string fetchErr;
                    if (!fetchHttp(path, cache, fetchErr)) {
                        if (err) *err = fetchErr;
                        return -1;
                    }
                }
                *resolved_asset_name = cache.string();
                return 0;
            }
            // Local path: defer to tinyusdz's standard search-path lookup
            // (same logic the resolver would have used without our wildcard).
            std::string found = tinyusdz::io::FindFile(path, search_paths);
            if (found.empty()) return -1;
            *resolved_asset_name = found;
            return 0;
        }

        // FSSizeAsset: works for both URL-cache files and local files since
        // resolve() always returns a local filesystem path.
        int httpAwareSize(const char* resolved_asset_name,
                          uint64_t* nbytes,
                          std::string* err,
                          void* /*userdata*/) {
            if (!resolved_asset_name || !nbytes) return -1;
            std::error_code ec;
            const auto sz = std::filesystem::file_size(resolved_asset_name, ec);
            if (ec) { if (err) *err = ec.message(); return -1; }
            *nbytes = sz;
            return 0;
        }

        // FSReadAsset: read from the resolved local path.
        int httpAwareRead(const char* resolved_asset_name,
                          uint64_t req_nbytes,
                          uint8_t* out_buf,
                          uint64_t* nbytes,
                          std::string* err,
                          void* /*userdata*/) {
            if (!resolved_asset_name || !out_buf || !nbytes) return -1;
            std::ifstream f(resolved_asset_name, std::ios::binary);
            if (!f) { if (err) *err = "open failed"; return -1; }
            f.read(reinterpret_cast<char*>(out_buf), static_cast<std::streamsize>(req_nbytes));
            *nbytes = static_cast<uint64_t>(f.gcount());
            return 0;
        }

        void registerHttpAwareResolver(tinyusdz::AssetResolutionResolver& resolver) {
            tinyusdz::AssetResolutionHandler h;
            h.resolve_fun = httpAwareResolve;
            h.size_fun = httpAwareSize;
            h.read_fun = httpAwareRead;
            resolver.register_wildcard_asset_resolution_handler(h);
        }

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

        // tinyusdz's CompositeReferences/CompositePayload only processes arcs
        // qualified as `ResetToExplicit` (no qualifier) or `Prepend`. Real-world
        // USD uses `add references` (chess set, MaterialX example assets) and
        // `append references` (some Omniverse exports) freely. For a flat
        // single-layer composition, all four "additive" qualifiers are
        // equivalent — they just contribute the referenced opinions. Rewrite
        // Add/Append to Prepend so tinyusdz processes them; strip Delete/Order
        // since nothing weaker exists for them to act on.
        void normalizeArcsQual(tinyusdz::PrimSpec& ps) {
            using tinyusdz::ListEditQual;
            auto rewrite = [](auto& slot) {
                if (!slot) return;
                auto& q = slot->first;
                if (q == ListEditQual::Add || q == ListEditQual::Append) {
                    q = ListEditQual::Prepend;
                } else if (q == ListEditQual::Delete || q == ListEditQual::Order) {
                    slot.reset();
                }
            };
            rewrite(ps.metas().references);
            rewrite(ps.metas().payload);
            rewrite(ps.metas().inherits);
            for (auto& child : ps.children())
                normalizeArcsQual(child);
        }

        void normalizeArcsInLayer(tinyusdz::Layer& layer) {
            for (auto& [name, ps] : layer.primspecs())
                normalizeArcsQual(ps);
        }

        // ----------------------------------------------------------------------
        // PointInstancer expansion. tinyusdz's tydra::RenderSceneConverter does
        // not visit PointInstancer prims (they're skipped along with their
        // children), so the prototypes never reach the rendered scene. We work
        // around this at the layer level: rewrite each PointInstancer into a
        // plain Xform that contains N child clones of the prototype, each
        // wrapped in an Xform with the per-instance translate/orient applied.
        // The original prototype child is removed (USD semantics: prototypes
        // are templates, not rendered themselves).
        // ----------------------------------------------------------------------

        struct InstanceXform {
            float tx = 0, ty = 0, tz = 0;
            // quat order in USDA: (w, x, y, z). Here quat[0]=w (real).
            float qw = 1, qx = 0, qy = 0, qz = 0;
            float sx = 1, sy = 1, sz = 1;
            bool hasOrient = false;
            bool hasScale = false;
        };

        // Look up a prim by absolute path in `searchSpec`'s subtree.
        const tinyusdz::PrimSpec* findPrimByPathRel(const tinyusdz::PrimSpec& root,
                                                    const std::string& rootAbsPath,
                                                    const std::string& targetAbsPath) {
            if (rootAbsPath == targetAbsPath) return &root;
            // Otherwise the target must be under root's path.
            if (targetAbsPath.size() <= rootAbsPath.size() ||
                targetAbsPath.compare(0, rootAbsPath.size(), rootAbsPath) != 0 ||
                targetAbsPath[rootAbsPath.size()] != '/') {
                return nullptr;
            }
            std::string rel = targetAbsPath.substr(rootAbsPath.size() + 1);
            const tinyusdz::PrimSpec* cur = &root;
            std::string curPath = rootAbsPath;
            while (!rel.empty()) {
                auto slash = rel.find('/');
                std::string name = (slash == std::string::npos) ? rel : rel.substr(0, slash);
                rel = (slash == std::string::npos) ? std::string() : rel.substr(slash + 1);
                const tinyusdz::PrimSpec* next = nullptr;
                for (const auto& c : cur->children()) {
                    if (c.name() == name) { next = &c; break; }
                }
                if (!next) return nullptr;
                cur = next;
                curPath += "/" + name;
            }
            return cur;
        }

        // Apply translate / orient / scale xformOps to the front of a copied prim.
        // Existing xformOps on the copy are preserved and run AFTER ours so the
        // prototype's local transforms still take effect.
        void applyInstanceXform(tinyusdz::PrimSpec& copy, const InstanceXform& x) {
            using tinyusdz::Property;
            using tinyusdz::Attribute;
            // Translate
            tinyusdz::value::float3 t{x.tx, x.ty, x.tz};
            copy.props()["xformOp:translate"] = Property(Attribute::Uniform(t));
            // Orient
            if (x.hasOrient) {
                tinyusdz::value::quatf q;
                q.imag[0] = x.qx; q.imag[1] = x.qy; q.imag[2] = x.qz;
                q.real    = x.qw;
                copy.props()["xformOp:orient"] = Property(Attribute::Uniform(q));
            }
            // Scale
            if (x.hasScale) {
                tinyusdz::value::float3 s{x.sx, x.sy, x.sz};
                copy.props()["xformOp:scale"] = Property(Attribute::Uniform(s));
            }
            // xformOpOrder — ours first, then any existing ops.
            std::vector<tinyusdz::value::token> order;
            order.emplace_back(tinyusdz::value::token("xformOp:translate"));
            if (x.hasOrient) order.emplace_back(tinyusdz::value::token("xformOp:orient"));
            if (x.hasScale)  order.emplace_back(tinyusdz::value::token("xformOp:scale"));
            // Append the existing order if present
            if (auto it = copy.props().find("xformOpOrder"); it != copy.props().end()) {
                if (it->second.is_attribute()) {
                    if (auto pv = it->second.get_attribute().get_value<std::vector<tinyusdz::value::token>>()) {
                        for (auto& tok : *pv) order.push_back(tok);
                    }
                }
            }
            // Attribute::Uniform<T> requires a scalar value type; for the
            // token[] array we build the Attribute by hand.
            tinyusdz::Attribute orderAttr;
            orderAttr.set_value(order);
            orderAttr.variability() = tinyusdz::Variability::Uniform;
            copy.props()["xformOpOrder"] = Property(orderAttr);
        }

        // Rewrite all relationship/connection targetPaths inside `ps` (and its
        // descendants + variantSets) whose prefix equals `srcPrefix`. Used
        // after deep-copying a PointInstancer prototype so that absolute
        // bindings pointing at the prototype's old path are redirected to the
        // instance's new path.
        void rewriteAbsPathsRec(tinyusdz::PrimSpec& ps,
                                const tinyusdz::Path& srcPrefix,
                                const tinyusdz::Path& dstPrefix) {
            for (auto& [name, prop] : ps.props()) {
                if (prop.is_relationship()) {
                    tinyusdz::Relationship& rel = prop.relationship();
                    if (rel.is_path()) {
                        if (rel.targetPath.has_prefix(srcPrefix)) {
                            rel.targetPath.replace_prefix(srcPrefix, dstPrefix);
                        }
                    } else if (rel.is_pathvector()) {
                        for (auto& path : rel.targetPathVector) {
                            if (path.has_prefix(srcPrefix)) {
                                path.replace_prefix(srcPrefix, dstPrefix);
                            }
                        }
                    }
                } else if (prop.is_attribute_connection()) {
                    tinyusdz::Attribute& attr = prop.attribute();
                    for (auto& cp : attr.connections()) {
                        if (cp.has_prefix(srcPrefix)) {
                            cp.replace_prefix(srcPrefix, dstPrefix);
                        }
                    }
                }
            }
            for (auto& child : ps.children()) {
                rewriteAbsPathsRec(child, srcPrefix, dstPrefix);
            }
            for (auto& vs : ps.variantSets()) {
                for (auto& kv : vs.second.variantSet) {
                    rewriteAbsPathsRec(kv.second, srcPrefix, dstPrefix);
                }
            }
        }

        // Helper: walk the layer to find the absolute path of a given PrimSpec
        // by pointer match. Used so we know what prefix to rewrite for
        // PointInstancer instance clones. Returns empty if not found.
        bool findAbsPathOfSpecRec(const tinyusdz::PrimSpec& haystack,
                                  std::string& curPath,
                                  const tinyusdz::PrimSpec* needle,
                                  std::string& out) {
            if (&haystack == needle) {
                out = curPath;
                return true;
            }
            for (const auto& c : haystack.children()) {
                std::string childPath = curPath + "/" + c.name();
                if (findAbsPathOfSpecRec(c, childPath, needle, out)) return true;
            }
            return false;
        }

        std::string findAbsPathOfSpec(const tinyusdz::Layer& layer,
                                      const tinyusdz::PrimSpec* needle) {
            if (!needle) return {};
            std::string out;
            for (const auto& [name, ps] : layer.primspecs()) {
                std::string root = "/" + name;
                if (findAbsPathOfSpecRec(ps, root, needle, out)) return out;
            }
            return out;
        }

        void expandPointInstancers(tinyusdz::PrimSpec& ps,
                                   const tinyusdz::Layer& layer,
                                   const std::string& parentPath) {
            if (ps.typeName() == "PointInstancer") {
                // Read positions
                std::vector<tinyusdz::value::point3f> positions;
                if (auto it = ps.props().find("positions"); it != ps.props().end() && it->second.is_attribute()) {
                    if (auto pv = it->second.get_attribute().get_value<std::vector<tinyusdz::value::point3f>>()) {
                        positions = *pv;
                    }
                }
                // Read protoIndices
                std::vector<int> protoIndices;
                if (auto it = ps.props().find("protoIndices"); it != ps.props().end() && it->second.is_attribute()) {
                    if (auto pv = it->second.get_attribute().get_value<std::vector<int>>()) {
                        protoIndices = *pv;
                    }
                }
                // Read orientations (quath[] or quatf[])
                std::vector<tinyusdz::value::quatf> orientations;
                if (auto it = ps.props().find("orientations"); it != ps.props().end() && it->second.is_attribute()) {
                    const auto& a = it->second.get_attribute();
                    if (auto pv = a.get_value<std::vector<tinyusdz::value::quatf>>()) {
                        orientations = *pv;
                    } else if (auto pvh = a.get_value<std::vector<tinyusdz::value::quath>>()) {
                        orientations.reserve(pvh->size());
                        for (const auto& q : *pvh) {
                            tinyusdz::value::quatf qf;
                            qf.imag[0] = tinyusdz::value::half_to_float(q.imag[0]);
                            qf.imag[1] = tinyusdz::value::half_to_float(q.imag[1]);
                            qf.imag[2] = tinyusdz::value::half_to_float(q.imag[2]);
                            qf.real    = tinyusdz::value::half_to_float(q.real);
                            orientations.push_back(qf);
                        }
                    }
                }
                // Read scales (float3[])
                std::vector<tinyusdz::value::float3> scales;
                if (auto it = ps.props().find("scales"); it != ps.props().end() && it->second.is_attribute()) {
                    if (auto pv = it->second.get_attribute().get_value<std::vector<tinyusdz::value::float3>>()) {
                        scales = *pv;
                    }
                }
                // Read prototypes rel
                std::vector<std::string> protoPaths;
                if (auto it = ps.props().find("prototypes"); it != ps.props().end() && it->second.is_relationship()) {
                    const auto& rel = it->second.get_relationship();
                    if (rel.is_path()) {
                        protoPaths.push_back(rel.targetPath.prim_part());
                    } else if (rel.is_pathvector()) {
                        for (const auto& p : rel.targetPathVector) {
                            protoPaths.push_back(p.prim_part());
                        }
                    }
                }

                if (!positions.empty() && !protoIndices.empty() && !protoPaths.empty()) {
                    const std::string instancerAbsPath = parentPath + "/" + ps.name();

                    // Resolve prototype prims (children of this PointInstancer
                    // in the typical authoring pattern). Match by leaf name.
                    std::vector<const tinyusdz::PrimSpec*> protoPrims(protoPaths.size(), nullptr);
                    std::vector<std::string> protoChildNames(protoPaths.size());
                    for (size_t k = 0; k < protoPaths.size(); ++k) {
                        const std::string& pp = protoPaths[k];
                        auto slash = pp.rfind('/');
                        std::string leaf = (slash == std::string::npos) ? pp : pp.substr(slash + 1);
                        protoChildNames[k] = leaf;
                        for (const auto& c : ps.children()) {
                            if (c.name() == leaf) { protoPrims[k] = &c; break; }
                        }
                    }

                    // Build instance children
                    std::vector<tinyusdz::PrimSpec> newChildren;
                    newChildren.reserve(positions.size());
                    for (size_t i = 0; i < positions.size(); ++i) {
                        int pi = (i < protoIndices.size()) ? protoIndices[i] : 0;
                        if (pi < 0 || pi >= (int)protoPrims.size() || !protoPrims[pi]) continue;
                        tinyusdz::PrimSpec inst = *protoPrims[pi];  // deep copy
                        const std::string instName = "__inst_" + std::to_string(i);
                        inst.name() = instName;

                        // Rewrite absolute paths inside the copy from the
                        // prototype's path to the instance's path so any
                        // material:binding rels (or other rels) that point at
                        // the prototype's now-removed Materials subtree
                        // resolve to the duplicated copy instead.
                        const std::string protoAbs = instancerAbsPath + "/" + protoChildNames[pi];
                        const std::string instAbs  = instancerAbsPath + "/" + instName;
                        rewriteAbsPathsRec(inst,
                            tinyusdz::Path(protoAbs, ""),
                            tinyusdz::Path(instAbs, ""));

                        InstanceXform xf;
                        xf.tx = positions[i].x;
                        xf.ty = positions[i].y;
                        xf.tz = positions[i].z;
                        if (i < orientations.size()) {
                            xf.qx = orientations[i].imag[0];
                            xf.qy = orientations[i].imag[1];
                            xf.qz = orientations[i].imag[2];
                            xf.qw = orientations[i].real;
                            const bool isIdentity = (xf.qw == 1.f && xf.qx == 0.f && xf.qy == 0.f && xf.qz == 0.f);
                            xf.hasOrient = !isIdentity;
                        }
                        if (i < scales.size()) {
                            xf.sx = scales[i][0];
                            xf.sy = scales[i][1];
                            xf.sz = scales[i][2];
                            const bool isIdentity = (xf.sx == 1.f && xf.sy == 1.f && xf.sz == 1.f);
                            xf.hasScale = !isIdentity;
                        }
                        applyInstanceXform(inst, xf);
                        newChildren.push_back(std::move(inst));
                    }

                    // Replace children: drop the prototype children, keep any
                    // unrelated children, append the new instances.
                    std::vector<tinyusdz::PrimSpec> kept;
                    for (auto& c : ps.children()) {
                        bool isProto = false;
                        for (auto* p : protoPrims) {
                            if (p == &c) { isProto = true; break; }
                        }
                        if (!isProto) kept.push_back(c);
                    }
                    for (auto& c : newChildren) kept.push_back(std::move(c));
                    ps.children() = std::move(kept);

                    // Demote PointInstancer to a plain Xform so tydra traverses
                    // its children.
                    ps.typeName() = "";
                }
            }
            const std::string myPath = parentPath + "/" + ps.name();
            for (auto& child : ps.children()) {
                expandPointInstancers(child, layer, myPath);
            }
        }

        void expandPointInstancersInLayer(tinyusdz::Layer& layer) {
            for (auto& [name, ps] : layer.primspecs()) {
                expandPointInstancers(ps, layer, "");
            }
        }

        // ----------------------------------------------------------------------
        // Variant selection. tinyusdz's composition leaves `variantSet` blocks
        // unresolved on the layer — meaning if a `material:binding` rel only
        // exists inside a `"Black" { over Render { rel ... } }` variant block,
        // the live PrimSpec for Render is missing it. We apply selections
        // ourselves: for each prim with a `variants = {…}` selection and a
        // matching `variantSet`, merge the selected variant's child opinions
        // into the prim. The variant block is the stronger opinion, so its
        // properties win on conflict; typeName/Def specifier from the existing
        // (def-typed) prim is preserved.
        // ----------------------------------------------------------------------

        void mergeVariantInto(tinyusdz::PrimSpec& dst, const tinyusdz::PrimSpec& src) {
            using tinyusdz::Specifier;

            // Properties: variant (src) is the stronger opinion.
            for (const auto& [k, v] : src.props()) {
                dst.props()[k] = v;
            }

            // typeName / specifier: don't let an `over`-style variant prim
            // demote a `def Mesh ...`. Existing typeName + Def survive.
            if (dst.typeName().empty() && !src.typeName().empty()) {
                dst.typeName() = src.typeName();
            }
            if (dst.specifier() == Specifier::Over && src.specifier() == Specifier::Def) {
                dst.specifier() = Specifier::Def;
            }

            // Children: recurse by name; new ones get appended.
            for (const auto& srcChild : src.children()) {
                auto it = std::find_if(dst.children().begin(), dst.children().end(),
                    [&](const tinyusdz::PrimSpec& d) { return d.name() == srcChild.name(); });
                if (it != dst.children().end()) {
                    mergeVariantInto(*it, srcChild);
                } else {
                    dst.children().push_back(srcChild);
                }
            }
        }

        void applyVariantsRec(tinyusdz::PrimSpec& ps) {
            if (ps.metas().variants && !ps.variantSets().empty()) {
                for (const auto& [vsname, selection] : ps.metas().variants.value()) {
                    auto vsIt = ps.variantSets().find(vsname);
                    if (vsIt == ps.variantSets().end()) continue;
                    auto vIt = vsIt->second.variantSet.find(selection);
                    if (vIt == vsIt->second.variantSet.end()) continue;
                    mergeVariantInto(ps, vIt->second);
                }
            }
            for (auto& child : ps.children()) {
                applyVariantsRec(child);
            }
        }

        void applyVariantsInLayer(tinyusdz::Layer& layer) {
            for (auto& [name, ps] : layer.primspecs()) {
                applyVariantsRec(ps);
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

        // -----------------------------------------------------------------------
        // Geometry
        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // Static linear-blend skinning. tinyusdz tydra surfaces UsdSkel data
        // (per-vertex jointIndices/jointWeights, per-joint bind/rest
        // transforms, geomBindTransform) but does NOT deform `RenderMesh::points`
        // — those stay in mesh-local bind-pose. For a non-animated render we
        // bake LBS at the rest pose so skinned geometry (e.g. the brake
        // cables on the CarbonFrameBike) shows in its authored shape rather
        // than as the raw bind layout.
        //
        // Maths per vertex:
        //   p_bind  = geomBindTransform * v_local
        //   p_rest  = sum_k(weight_k * (restWorld_k * inverse(bindWorld_k)) * p_bind)
        // where restWorld_k is the world-space rest transform of joint k
        // (computed by walking the hierarchy with parent_world * local_rest)
        // and bindWorld_k is the joint's world-space bind transform (USD
        // `bindTransforms` is already world-space).
        // -----------------------------------------------------------------------

        std::unordered_map<int, Matrix4> buildSkinMatrices(
                const tinyusdz::tydra::SkelHierarchy& skel) {

            std::unordered_map<int, Matrix4> out;
            std::function<void(const tinyusdz::tydra::SkelNode&, const Matrix4&)> walk =
                [&](const tinyusdz::tydra::SkelNode& node,
                    const Matrix4& parentRestWorld) {
                    Matrix4 localRest = toMatrix4(node.rest_transform);
                    Matrix4 nodeRestWorld;
                    nodeRestWorld.multiplyMatrices(parentRestWorld, localRest);

                    Matrix4 bindWorld = toMatrix4(node.bind_transform);
                    Matrix4 bindWorldInv;
                    bindWorldInv.copy(bindWorld).invert();

                    Matrix4 skin;
                    skin.multiplyMatrices(nodeRestWorld, bindWorldInv);
                    if (node.joint_id >= 0) {
                        out[node.joint_id] = skin;
                    }
                    for (const auto& child : node.children) {
                        walk(child, nodeRestWorld);
                    }
                };
            walk(skel.root_node, Matrix4().identity());
            return out;
        }

        // Apply LBS to `rm.points` and return the deformed flat vec3 array.
        std::vector<float> deformPointsByLBS(
                const tinyusdz::tydra::RenderMesh& rm,
                const std::unordered_map<int, Matrix4>& skinMatrices) {

            const Matrix4 geomBind = toMatrix4(rm.joint_and_weights.geomBindTransform);
            const auto& jids = rm.joint_and_weights.jointIndices;
            const auto& jws  = rm.joint_and_weights.jointWeights;
            const int es = rm.joint_and_weights.elementSize;
            const size_t numV = rm.points.size();

            std::vector<float> out;
            out.reserve(numV * 3);

            const bool haveSkin = es > 0 &&
                jids.size() >= numV * static_cast<size_t>(es) &&
                jws.size()  >= numV * static_cast<size_t>(es);

            for (size_t v = 0; v < numV; ++v) {
                Vector3 vBind(rm.points[v][0], rm.points[v][1], rm.points[v][2]);
                vBind.applyMatrix4(geomBind);

                if (!haveSkin) {
                    out.push_back(vBind.x); out.push_back(vBind.y); out.push_back(vBind.z);
                    continue;
                }

                Vector3 vOut(0.f, 0.f, 0.f);
                float totalW = 0.f;
                for (int k = 0; k < es; ++k) {
                    const size_t idx = v * static_cast<size_t>(es) + static_cast<size_t>(k);
                    const int jId = jids[idx];
                    const float w = jws[idx];
                    if (w <= 0.f) continue;
                    auto it = skinMatrices.find(jId);
                    if (it == skinMatrices.end()) continue;
                    Vector3 d = vBind;
                    d.applyMatrix4(it->second);
                    vOut.x += w * d.x;
                    vOut.y += w * d.y;
                    vOut.z += w * d.z;
                    totalW += w;
                }
                if (totalW > 0.f) {
                    vOut.x /= totalW; vOut.y /= totalW; vOut.z /= totalW;
                } else {
                    vOut = vBind;
                }
                out.push_back(vOut.x); out.push_back(vOut.y); out.push_back(vOut.z);
            }
            return out;
        }

        std::shared_ptr<BufferGeometry> geometryFromRenderMesh(
                const tinyusdz::tydra::RenderMesh& rm,
                const tinyusdz::tydra::RenderScene& scene) {

            if (rm.points.empty()) return nullptr;

            // Prefer triangulated index buffer; fall back to raw (should not happen
            // with triangulate=true but be defensive).
            const std::vector<uint32_t>& idxBuf = rm.is_triangulated()
                ? rm.triangulatedFaceVertexIndices
                : rm.usdFaceVertexIndices;
            if (idxBuf.empty()) return nullptr;

            // Positions — vec3 = std::array<float,3>. If the mesh has skinning
            // data, bake the rest pose now (otherwise the shapes look wrong;
            // see comments above buildSkinMatrices).
            std::vector<float> pos;
            const bool isSkinned =
                rm.skel_id >= 0 &&
                rm.skel_id < static_cast<int>(scene.skeletons.size()) &&
                !rm.joint_and_weights.jointIndices.empty();
            if (isSkinned) {
                auto skinMats = buildSkinMatrices(scene.skeletons[rm.skel_id]);
                pos = deformPointsByLBS(rm, skinMats);
            } else {
                pos.reserve(rm.points.size() * 3);
                for (const auto& p : rm.points) {
                    pos.push_back(p[0]);
                    pos.push_back(p[1]);
                    pos.push_back(p[2]);
                }
            }

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position",
                    FloatBufferAttribute::create(std::move(pos), 3));

            // Copy indices (uint32 → unsigned int, same size on all modern targets).
            // USD meshes default to `orientation = "rightHanded"` (CCW front
            // faces, matches three.js). When a mesh declares `leftHanded` the
            // triangle winding is reversed; flip every triangle so we don't
            // render its faces from the back (which makes lit metals look
            // black under direct lighting).
            std::vector<unsigned int> indices(idxBuf.begin(), idxBuf.end());
            if (!rm.is_rightHanded) {
                for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                    std::swap(indices[i + 1], indices[i + 2]);
                }
            }
            geometry->setIndex(indices);

            // Normals. Authored normals from real-world USD assets are too
            // unreliable to trust globally — different exporters disagree on
            // the "outward" convention, and even within a single asset the
            // direction can flip across submeshes (chess set: King/Queen
            // leftHanded with one convention, Pawn rightHanded with another).
            // Recompute per-face from the post-swap winding: guaranteed
            // outward, costs only smoothing-group fidelity.
            geometry->computeVertexNormals();

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
                const bool flipY = true;
                std::shared_ptr<Texture> tex;
                if (std::filesystem::exists(texPath)) {
                    tex = texLoader.load(texPath, flipY);
                } else if (img.buffer_id >= 0 &&
                           img.buffer_id < static_cast<int64_t>(scene.buffers.size())) {
                    // .usdz textures live inside the zip; tinyusdz stores them
                    // in `scene.buffers[image.buffer_id]`. tydra's USDZAsset
                    // resolver normally decodes them ahead of time, so we get
                    // raw RGBA pixel rows in `buf.data`. Fall back to
                    // loadFromMemory for the un-decoded case (encoded
                    // JPEG/PNG bytes).
                    const auto& buf = scene.buffers[img.buffer_id];
                    if (!buf.data.empty()) {
                        if (img.decoded && img.width > 0 && img.height > 0 && img.channels > 0) {
                            // tinyusdz mis-tags the component type for some
                            // .usdz pipelines (reports UInt8 but stores RGBA
                            // float32). Detect bytes-per-component from the
                            // actual buffer size and convert to UInt8 so
                            // threepp's Texture sees standard RGBA8 — reading
                            // the bytes as uint8 directly produces garbled
                            // noise (the chevron-pattern carbon weave).
                            const size_t numComponents =
                                static_cast<size_t>(img.width) *
                                static_cast<size_t>(img.height) *
                                static_cast<size_t>(img.channels);
                            const size_t bytesPerComponent =
                                numComponents > 0 ? buf.data.size() / numComponents : 1;
                            const size_t srcRowBytes =
                                static_cast<size_t>(img.width) *
                                static_cast<size_t>(img.channels) *
                                bytesPerComponent;
                            const size_t srcExpected =
                                srcRowBytes * static_cast<size_t>(img.height);

                            if (buf.data.size() >= srcExpected) {
                                const size_t dstRowBytes =
                                    static_cast<size_t>(img.width) *
                                    static_cast<size_t>(img.channels);
                                std::vector<unsigned char> pixels(
                                    dstRowBytes * static_cast<size_t>(img.height));

                                auto copyAndConvertRow = [&](size_t srcRow, size_t dstRow) {
                                    const unsigned char* src = buf.data.data() + srcRow * srcRowBytes;
                                    unsigned char*       dst = pixels.data() + dstRow * dstRowBytes;
                                    const size_t rowComponents =
                                        static_cast<size_t>(img.width) *
                                        static_cast<size_t>(img.channels);
                                    if (bytesPerComponent == 4) {
                                        // RGBA float32 → uint8 (clamp [0,1] → 0..255).
                                        const float* fsrc = reinterpret_cast<const float*>(src);
                                        for (size_t k = 0; k < rowComponents; ++k) {
                                            float v = fsrc[k];
                                            if (v < 0.f) v = 0.f;
                                            if (v > 1.f) v = 1.f;
                                            dst[k] = static_cast<unsigned char>(v * 255.f + 0.5f);
                                        }
                                    } else if (bytesPerComponent == 2) {
                                        // RGBA uint16 → uint8 (drop low byte).
                                        const uint16_t* hsrc = reinterpret_cast<const uint16_t*>(src);
                                        for (size_t k = 0; k < rowComponents; ++k) {
                                            dst[k] = static_cast<unsigned char>(hsrc[k] >> 8);
                                        }
                                    } else if (bytesPerComponent == 1) {
                                        std::memcpy(dst, src, rowComponents);
                                    } else {
                                        std::memset(dst, 0, rowComponents);
                                    }
                                };

                                // tinyusdz hands us top-down rows (stb-style);
                                // threepp/GL with flipY=true expects bottom-up.
                                for (int y = 0; y < img.height; ++y) {
                                    const int srcY = flipY ? (img.height - 1 - y) : y;
                                    copyAndConvertRow(static_cast<size_t>(srcY),
                                                      static_cast<size_t>(y));
                                }
                                Image image(std::move(pixels),
                                            static_cast<unsigned int>(img.width),
                                            static_cast<unsigned int>(img.height));
                                tex = Texture::create(image);
                                tex->name = rawId;
                                tex->format = (img.channels == 4) ? Format::RGBA : Format::RGB;
                                tex->needsUpdate();
                            }
                        } else {
                            // Encoded JPEG/PNG bytes — let TextureLoader sniff
                            // the format and decode.
                            tex = texLoader.loadFromMemory(rawId, buf.data, flipY);
                        }
                    }
                }
                if (!tex) continue;

                if (tex) {
                    // Map tinyusdz wrap modes onto threepp wrap modes.
                    auto mapWrap = [](tinyusdz::tydra::UVTexture::WrapMode w) {
                        using W = tinyusdz::tydra::UVTexture::WrapMode;
                        switch (w) {
                            case W::REPEAT:           return TextureWrapping::Repeat;
                            case W::MIRROR:           return TextureWrapping::MirroredRepeat;
                            case W::CLAMP_TO_BORDER:  return TextureWrapping::ClampToEdge;
                            case W::CLAMP_TO_EDGE:
                            default:                  return TextureWrapping::ClampToEdge;
                        }
                    };

                    // Apply UsdTransform2d when authored, plus per-UVTexture
                    // wrap mode. The texLoader caches by file path, so
                    // multiple UVTextures sharing an image otherwise share
                    // the SAME Texture object; clone before mutating
                    // per-UVTexture state so transform/wrap don't leak.
                    const bool hasXform = uvtex.has_transform2d;
                    const bool wantRepeat =
                        uvtex.wrapS == tinyusdz::tydra::UVTexture::WrapMode::REPEAT ||
                        uvtex.wrapT == tinyusdz::tydra::UVTexture::WrapMode::REPEAT ||
                        uvtex.wrapS == tinyusdz::tydra::UVTexture::WrapMode::MIRROR ||
                        uvtex.wrapT == tinyusdz::tydra::UVTexture::WrapMode::MIRROR ||
                        // UsdTransform2d with scale != 1 implies the artist
                        // wants tiling; default-clamp would defeat it.
                        (hasXform && (uvtex.tx_scale[0] != 1.f || uvtex.tx_scale[1] != 1.f));

                    if (hasXform || wantRepeat) {
                        auto cloned = tex->clone();
                        cloned->wrapS = mapWrap(uvtex.wrapS);
                        cloned->wrapT = mapWrap(uvtex.wrapT);
                        if (wantRepeat) {
                            // Force repeat so tiling actually shows when the
                            // wrap was authored as repeat OR the transform
                            // implies tiling.
                            if (cloned->wrapS == TextureWrapping::ClampToEdge)
                                cloned->wrapS = TextureWrapping::Repeat;
                            if (cloned->wrapT == TextureWrapping::ClampToEdge)
                                cloned->wrapT = TextureWrapping::Repeat;
                        }
                        if (hasXform) {
                            // Set the UV matrix directly from tinyusdz's
                            // pre-computed transform (scale*rotate*translate
                            // already composed) to avoid any mismatch
                            // between three.js's offset/repeat/rotation/center
                            // composition order and USD's. The matrix is
                            // applied in the shader as `matrix * vec3(uv, 1)`,
                            // matching USD's UsdTransform2d output.
                            const auto& T = uvtex.transform.m;
                            cloned->matrix.set(
                                T[0][0], T[0][1], T[0][2],
                                T[1][0], T[1][1], T[1][2],
                                T[2][0], T[2][1], T[2][2]);
                            cloned->matrixAutoUpdate = false;
                        }
                        tex = cloned;
                    }
                }

                cache[ti] = tex;
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
            // Covers: snake_case (OmniPBR style), PascalCase (OmniUe4Base /
            // Omniverse Unreal bridge, used by Lightwheel exports), and
            // *_image (NVIDIA OmniSurface MDL — Omniverse Automotive
            // Showcase materials).
            if (auto t = tryFirst({"inputs:diffuse_texture",
                                   "inputs:albedo_texture",
                                   "inputs:Diffuse_Texture",
                                   "inputs:Albedo_Texture",
                                   "inputs:BaseColor_Texture",
                                   "inputs:base_color_image",
                                   "inputs:diffuse_color_image"})) {
                t->colorSpace = ColorSpace::sRGB;
                mat->map = t; anyTex = true;
            }
            // Normal map. Tangent-space normals are linear data; flipY=true
            // matches diffuse so the normal-map detail aligns with the base
            // color (data textures need the same Y orientation as color
            // textures otherwise the surface detail looks flipped).
            if (auto t = tryFirst({"inputs:normalmap_texture",
                                   "inputs:normal_texture",
                                   "inputs:Normal_Texture",
                                   "inputs:NormalMap_Texture",
                                   "inputs:Bump_Texture",
                                   "inputs:normal_image"}, true)) {
                t->colorSpace = ColorSpace::Linear;
                mat->normalMap = t; anyTex = true;
            }
            // Roughness — try ORM-packed first, then dedicated roughness map.
            if (auto t = tryLoad("inputs:ORM_texture", true)) {
                t->colorSpace = ColorSpace::Linear;
                mat->aoMap         = t;
                mat->roughnessMap  = t;
                mat->metalnessMap  = t;
                anyTex = true;
            } else {
                if (auto r = tryFirst({"inputs:roughness_texture",
                                       "inputs:Roughness_Texture",
                                       "inputs:roughness_image"}, true)) {
                    r->colorSpace = ColorSpace::Linear;
                    mat->roughnessMap = r; anyTex = true;
                }
                if (auto m = tryFirst({"inputs:metallic_texture",
                                       "inputs:Metallic_Texture",
                                       "inputs:metalness_image",
                                       "inputs:metallic_image"}, true)) {
                    m->colorSpace = ColorSpace::Linear;
                    mat->metalnessMap = m; anyTex = true;
                }
                // OmniSurface "cavity" map ~= ambient occlusion.
                if (auto a = tryFirst({"inputs:ao_image",
                                       "inputs:cavity_image"}, true)) {
                    a->colorSpace = ColorSpace::Linear;
                    mat->aoMap = a; anyTex = true;
                }
            }
            // Emissive
            if (auto t = tryFirst({"inputs:emissive_texture",
                                   "inputs:Emissive_Texture",
                                   "inputs:Emission_Texture",
                                   "inputs:emissive_image",
                                   "inputs:emission_image",
                                   "inputs:emission_color_image"})) {
                t->colorSpace = ColorSpace::sRGB;
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
                } else if (auto c3 = tryFloatTuple("inputs:diffuse_reflection_color")) {
                    // OmniSurface MDL constant base color.
                    mat->color.setRGB((*c3)[0], (*c3)[1], (*c3)[2]);
                    anyTex = true;
                }
            }
            if (!mat->roughnessMap) {
                if (auto v = tryFloat("inputs:Roughness_Color")) {
                    mat->roughness = *v;
                } else if (auto v2 = tryFloat("inputs:reflection_roughness_constant")) {
                    mat->roughness = *v2;
                } else if (auto v3 = tryFloat("inputs:roughness_range_position")) {
                    mat->roughness = *v3;
                }
            }
            if (!mat->metalnessMap) {
                if (auto v = tryFloat("inputs:Metallic_Color")) {
                    mat->metalness = *v;
                } else if (auto v2 = tryFloat("inputs:metallic_constant")) {
                    mat->metalness = *v2;
                } else if (auto v3 = tryFloat("inputs:metalness")) {
                    mat->metalness = *v3;
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
                        auto t = texLoader.load(path, ColorSpace::sRGB, true);
                        mat->map = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "NormalTex")) {
                    if (!mat->normalMap) {
                        auto t = texLoader.load(path, ColorSpace::Linear, true);
                        mat->normalMap = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "RoughnessTex")) {
                    if (!mat->roughnessMap) {
                        auto t = texLoader.load(path, ColorSpace::Linear, true);
                        mat->roughnessMap = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "MetallicTex")) {
                    if (!mat->metalnessMap) {
                        auto t = texLoader.load(path, ColorSpace::Linear, true);
                        mat->metalnessMap = t;
                        anyTex = true;
                    }
                } else if (containsCI(sname, "EmissiveColorTex") ||
                           containsCI(sname, "EmissionTex")) {
                    if (!mat->emissiveMap) {
                        auto t = texLoader.load(path, ColorSpace::sRGB, true);
                        mat->emissiveMap = t;
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
        // Minimal MaterialX (.mtlx) reader.
        //
        // tinyusdz only knows how to load a .mtlx whose root prim path is
        // exactly `</MaterialX>`; real-world references like the OpenChessSet's
        // `</MaterialX/Materials>` cause CompositeReferences to bail out and
        // leave the bound Material prim empty. We work around this by parsing
        // the .mtlx ourselves (XML) and extracting just the texture file paths
        // for `base_color` / `metalness` / `specular_roughness` / `normal` —
        // enough to populate a MeshStandardMaterial.
        //
        // Scope: the chess-set / standard_surface / nodegraph / image /
        // normalmap pattern. Not a general MaterialX implementation.
        // -----------------------------------------------------------------------

        struct MtlxElem {
            std::string name;
            std::map<std::string, std::string> attrs;
            std::vector<MtlxElem> children;

            const std::string* attr(const std::string& k) const {
                auto it = attrs.find(k);
                return it == attrs.end() ? nullptr : &it->second;
            }
            std::string attrOr(const std::string& k) const {
                auto p = attr(k);
                return p ? *p : std::string();
            }
        };

        // Skip whitespace, XML declaration, comments.
        size_t mtlxSkipWS(const std::string& s, size_t i) {
            while (i < s.size()) {
                if (std::isspace(static_cast<unsigned char>(s[i]))) ++i;
                else if (s.compare(i, 4, "<!--") == 0) {
                    auto e = s.find("-->", i);
                    i = (e == std::string::npos) ? s.size() : e + 3;
                }
                else if (s.compare(i, 5, "<?xml") == 0) {
                    auto e = s.find("?>", i);
                    i = (e == std::string::npos) ? s.size() : e + 2;
                }
                else break;
            }
            return i;
        }

        bool mtlxParseElem(const std::string& s, size_t& i, MtlxElem& out) {
            if (i >= s.size() || s[i] != '<') return false;
            ++i;
            const auto nameStart = i;
            while (i < s.size() &&
                   (std::isalnum(static_cast<unsigned char>(s[i])) ||
                    s[i] == '_' || s[i] == '-' || s[i] == ':')) ++i;
            out.name = s.substr(nameStart, i - nameStart);

            // Attributes
            while (i < s.size()) {
                i = mtlxSkipWS(s, i);
                if (i >= s.size()) return false;
                if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '>') {
                    i += 2;
                    return true;  // self-closing, no children
                }
                if (s[i] == '>') { ++i; break; }
                const auto aStart = i;
                while (i < s.size() &&
                       (std::isalnum(static_cast<unsigned char>(s[i])) ||
                        s[i] == '_' || s[i] == '-' || s[i] == ':')) ++i;
                std::string aName = s.substr(aStart, i - aStart);
                i = mtlxSkipWS(s, i);
                if (i >= s.size() || s[i] != '=') return false;
                ++i;
                i = mtlxSkipWS(s, i);
                if (i >= s.size() || s[i] != '"') return false;
                ++i;
                const auto vStart = i;
                while (i < s.size() && s[i] != '"') ++i;
                out.attrs[aName] = s.substr(vStart, i - vStart);
                if (i < s.size()) ++i;
            }

            // Children up to </name>
            while (i < s.size()) {
                i = mtlxSkipWS(s, i);
                if (i >= s.size()) return false;
                if (s.compare(i, 2, "</") == 0) {
                    i += 2;
                    while (i < s.size() && s[i] != '>') ++i;
                    if (i < s.size()) ++i;
                    return true;
                }
                if (s[i] == '<') {
                    MtlxElem child;
                    if (!mtlxParseElem(s, i, child)) return false;
                    out.children.push_back(std::move(child));
                } else {
                    ++i;  // stray text
                }
            }
            return true;
        }

        bool mtlxParseRoot(const std::string& src, MtlxElem& root) {
            size_t i = mtlxSkipWS(src, 0);
            return mtlxParseElem(src, i, root);
        }

        // Maps assembled from a parsed .mtlx file. Allows resolving from a
        // surfacematerial name down to the texture file path for each input.
        struct MtlxMaps {
            std::map<std::string, std::string> imageFile;       // image name -> filename attr
            std::map<std::string, std::string> normalmapToImg;  // normalmap name -> input image name
            std::map<std::pair<std::string, std::string>, std::string> outputs;  // (ng, output) -> source node
            struct ShaderInput {
                std::string ng;
                std::string output;
                std::string value;
            };
            std::map<std::pair<std::string, std::string>, ShaderInput> shaderInputs;  // (shader, input) -> SI
            std::map<std::string, std::string> matToShader;     // M_NAME -> standard_surface name
        };

        MtlxMaps buildMtlxMaps(const MtlxElem& root) {
            MtlxMaps m;
            if (root.name != "materialx") return m;
            for (const auto& top : root.children) {
                if (top.name == "nodegraph") {
                    const std::string ng = top.attrOr("name");
                    for (const auto& c : top.children) {
                        if (c.name == "image") {
                            const std::string n = c.attrOr("name");
                            for (const auto& cc : c.children) {
                                if (cc.name == "input" && cc.attrOr("name") == "file") {
                                    m.imageFile[n] = cc.attrOr("value");
                                    break;
                                }
                            }
                        } else if (c.name == "normalmap") {
                            const std::string n = c.attrOr("name");
                            for (const auto& cc : c.children) {
                                if (cc.name == "input" && cc.attrOr("name") == "in") {
                                    m.normalmapToImg[n] = cc.attrOr("nodename");
                                    break;
                                }
                            }
                        } else if (c.name == "output") {
                            m.outputs[{ng, c.attrOr("name")}] = c.attrOr("nodename");
                        }
                    }
                } else if (top.name == "standard_surface") {
                    const std::string ss = top.attrOr("name");
                    for (const auto& c : top.children) {
                        if (c.name == "input") {
                            MtlxMaps::ShaderInput si{c.attrOr("nodegraph"),
                                                     c.attrOr("output"),
                                                     c.attrOr("value")};
                            m.shaderInputs[{ss, c.attrOr("name")}] = si;
                        }
                    }
                } else if (top.name == "surfacematerial") {
                    const std::string mat = top.attrOr("name");
                    for (const auto& c : top.children) {
                        if (c.name == "input" && c.attrOr("name") == "surfaceshader") {
                            m.matToShader[mat] = c.attrOr("nodename");
                            break;
                        }
                    }
                }
            }
            return m;
        }

        // Resolve a single shader input chain (input -> nodegraph output -> source node -> image file).
        std::string resolveMtlxInput(const MtlxMaps& m,
                                     const std::string& shader,
                                     const std::string& input) {
            auto itI = m.shaderInputs.find({shader, input});
            if (itI == m.shaderInputs.end()) return {};
            const auto& si = itI->second;
            if (si.ng.empty() || si.output.empty()) return {};
            auto itO = m.outputs.find({si.ng, si.output});
            if (itO == m.outputs.end()) return {};
            std::string node = itO->second;
            // Follow normalmap → image
            if (auto itN = m.normalmapToImg.find(node); itN != m.normalmapToImg.end()) {
                node = itN->second;
            }
            auto itImg = m.imageFile.find(node);
            if (itImg == m.imageFile.end()) return {};
            return itImg->second;
        }

        // Direct-value getter for a shader input (e.g. `base_color = "1, 1, 1"`).
        // Returns the raw `value=` attribute text or empty.
        std::string resolveMtlxConstant(const MtlxMaps& m,
                                        const std::string& shader,
                                        const std::string& input) {
            auto it = m.shaderInputs.find({shader, input});
            if (it == m.shaderInputs.end()) return {};
            return it->second.value;
        }

        // Parse a comma-separated MaterialX color3 / vector3 / float-tuple.
        // "1, 1, 0.828" -> {1, 1, 0.828}. Returns false on parse failure.
        bool parseMtlxFloatTuple(const std::string& s, std::vector<float>& out) {
            out.clear();
            std::string cur;
            for (char c : s) {
                if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
                    if (!cur.empty()) { out.push_back(std::stof(cur)); cur.clear(); }
                } else {
                    cur.push_back(c);
                }
            }
            if (!cur.empty()) out.push_back(std::stof(cur));
            return !out.empty();
        }

        // Build a MeshStandardMaterial from a .mtlx file by surfacematerial
        // name (e.g. "M_King_B").
        std::shared_ptr<MeshStandardMaterial> materialFromMtlxFile(
                const std::filesystem::path& mtlxFile,
                const std::string& materialName,
                TextureLoader& texLoader) {

            std::ifstream f(mtlxFile, std::ios::binary);
            if (!f) return nullptr;
            std::stringstream ss;
            ss << f.rdbuf();
            const std::string src = ss.str();

            MtlxElem root;
            if (!mtlxParseRoot(src, root)) return nullptr;
            const MtlxMaps m = buildMtlxMaps(root);

            auto itMS = m.matToShader.find(materialName);
            if (itMS == m.matToShader.end()) return nullptr;
            const std::string& shader = itMS->second;

            const std::filesystem::path mtlxDir = mtlxFile.parent_path();
            auto resolveRel = [&](const std::string& rel) -> std::filesystem::path {
                if (rel.empty()) return {};
                std::filesystem::path p = mtlxDir / rel;
                return std::filesystem::exists(p) ? p : std::filesystem::path{};
            };

            auto baseColorPath = resolveRel(resolveMtlxInput(m, shader, "base_color"));
            auto metalnessPath = resolveRel(resolveMtlxInput(m, shader, "metalness"));
            auto roughnessPath = resolveRel(resolveMtlxInput(m, shader, "specular_roughness"));
            auto normalPath    = resolveRel(resolveMtlxInput(m, shader, "normal"));

            // Constant-value fallbacks. Some standard_surface shaders don't
            // bind a texture for an input but instead provide a literal value
            // (e.g. pawn `Top` uses base_color=(1,1,1) + transmission=1 +
            // transmission_color=(...) for tinted glass).
            std::vector<float> baseColorConst, transmissionColorConst;
            float transmissionConst = 0.f;
            float metalnessConst = -1.f;
            float roughnessConst = -1.f;
            float iorConst = -1.f;
            try {
                if (auto v = resolveMtlxConstant(m, shader, "base_color"); !v.empty())
                    parseMtlxFloatTuple(v, baseColorConst);
                if (auto v = resolveMtlxConstant(m, shader, "transmission_color"); !v.empty())
                    parseMtlxFloatTuple(v, transmissionColorConst);
                if (auto v = resolveMtlxConstant(m, shader, "transmission"); !v.empty()) {
                    std::vector<float> t; parseMtlxFloatTuple(v, t);
                    if (!t.empty()) transmissionConst = t[0];
                }
                if (auto v = resolveMtlxConstant(m, shader, "metalness"); !v.empty()) {
                    std::vector<float> t; parseMtlxFloatTuple(v, t);
                    if (!t.empty()) metalnessConst = t[0];
                }
                if (auto v = resolveMtlxConstant(m, shader, "specular_roughness"); !v.empty()) {
                    std::vector<float> t; parseMtlxFloatTuple(v, t);
                    if (!t.empty()) roughnessConst = t[0];
                }
                if (auto v = resolveMtlxConstant(m, shader, "specular_IOR"); !v.empty()) {
                    std::vector<float> t; parseMtlxFloatTuple(v, t);
                    if (!t.empty()) iorConst = t[0];
                }
            } catch (...) {
                // Best-effort numeric parse; ignore malformed values.
            }

            const bool hasAnyTexture =
                !baseColorPath.empty() || !metalnessPath.empty() ||
                !roughnessPath.empty() || !normalPath.empty();
            const bool hasAnyConstant =
                baseColorConst.size() >= 3 || metalnessConst >= 0.f ||
                roughnessConst >= 0.f || transmissionColorConst.size() >= 3 ||
                transmissionConst > 0.f;

            if (!hasAnyTexture && !hasAnyConstant) {
                return nullptr;
            }

            // Use MeshPhysicalMaterial when the shader requests transmission,
            // so the renderer evaluates a real refraction lobe (Beer–Lambert
            // attenuation, IOR-based refraction). Otherwise stay on the
            // cheaper MeshStandardMaterial. The return type is the standard
            // material base — physical IS-A standard, so callers don't care.
            //
            // Notes:
            //   - Don't set `transparent`: three.js's transmission compositing
            //     is its own pass (it samples a screenspace backdrop), and
            //     enabling alpha blending forces the surface into the
            //     transparent pass which breaks the refraction lookup. The
            //     transmission shader explicitly forces opaque output.
            //   - Leave `thinWalled` at its default. The flag is a Vulkan PT
            //     hint; turning it on for raster renderers has no effect, but
            //     keep it false so we don't surprise PT for closed pieces.
            std::shared_ptr<MeshStandardMaterial> mat;
            if (transmissionConst > 0.f) {
                auto pm = MeshPhysicalMaterial::create();
                pm->transmission = transmissionConst;
                // `thickness` drives how far the refracted ray travels inside
                // the medium and how much it bends in `getIBLVolumeRefraction`.
                // 0 (default) collapses refraction to no-op, so the surface
                // looks like a flat tinted layer rather than glass. Pick a
                // chess-piece-scale value; the result is the inverted-scene
                // lensing through pawn tops you'd expect from a glass dome.
                pm->thickness = 0.02f;
                if (transmissionColorConst.size() >= 3) {
                    pm->attenuationColor.setRGB(
                        transmissionColorConst[0],
                        transmissionColorConst[1],
                        transmissionColorConst[2]);
                    // attenuationDistance=0 means "no attenuation". Use a
                    // value comparable to thickness so the tint colours the
                    // refracted backdrop without dominating it.
                    pm->attenuationDistance = 0.05f;
                }
                pm->setIor(iorConst > 1.f ? iorConst : 1.5f);
                mat = pm;
            } else {
                mat = MeshStandardMaterial::create();
            }

            if (!baseColorPath.empty()) {
                mat->map = texLoader.load(baseColorPath, ColorSpace::sRGB, true);
            } else if (baseColorConst.size() >= 3) {
                mat->color.setRGB(baseColorConst[0], baseColorConst[1], baseColorConst[2]);
            }
            if (!metalnessPath.empty()) {
                mat->metalnessMap = texLoader.load(metalnessPath, ColorSpace::Linear, true);
            } else if (metalnessConst >= 0.f) {
                mat->metalness = metalnessConst;
            }
            if (!roughnessPath.empty()) {
                mat->roughnessMap = texLoader.load(roughnessPath, ColorSpace::Linear, true);
            } else if (roughnessConst >= 0.f) {
                mat->roughness = roughnessConst;
            }
            if (!normalPath.empty()) {
                mat->normalMap = texLoader.load(normalPath, ColorSpace::Linear, true);
            }
            return mat;
        }

        // Locate a .mtlx reference on `spec` and return the resolved absolute
        // file path. Looks at `references` metadata for any asset whose path
        // ends in .mtlx.
        std::filesystem::path findMtlxRefOnSpec(
                const tinyusdz::PrimSpec& spec,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver) {

            if (!spec.metas().references) return {};
            std::filesystem::path cwp = spec.get_current_working_path().empty()
                ? baseDir
                : std::filesystem::path(spec.get_current_working_path());
            for (const auto& ref : spec.metas().references->second) {
                std::string raw = ref.asset_path.GetAssetPath();
                if (raw.empty()) continue;
                auto dot = raw.rfind('.');
                if (dot == std::string::npos) continue;
                std::string ext = raw.substr(dot);
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext != ".mtlx") continue;
                // Resolve the asset path
                std::string norm = raw;
                std::replace(norm.begin(), norm.end(), '\\', '/');
                std::string rel = norm;
                if (rel.size() >= 2 && rel[0] == '.' && rel[1] == '/') rel = rel.substr(2);
                auto cand = cwp / rel;
                if (std::filesystem::exists(cand)) return cand;
                cand = baseDir / rel;
                if (std::filesystem::exists(cand)) return cand;
                std::string r = resolver.resolve(norm);
                if (!r.empty() && std::filesystem::exists(r)) return std::filesystem::path(r);
            }
            return {};
        }

        // Top-level MaterialX path: given the raw binding target for a mesh
        // (e.g. /ChessSet/Black/King/Materials/M_King_B), find the parent
        // Materials Scope's .mtlx reference and resolve the binding's leaf
        // name through it. Returns nullptr if not a MaterialX-backed binding.
        std::shared_ptr<MeshStandardMaterial> materialFromMtlxBinding(
                const tinyusdz::Layer& layer,
                const std::string& bindingTargetPath,
                const std::filesystem::path& baseDir,
                const tinyusdz::AssetResolutionResolver& resolver,
                TextureLoader& texLoader) {

            const auto slash = bindingTargetPath.rfind('/');
            if (slash == std::string::npos || slash == 0) return nullptr;
            const std::string parentPath = bindingTargetPath.substr(0, slash);
            const std::string matLeaf = bindingTargetPath.substr(slash + 1);
            if (matLeaf.empty()) return nullptr;

            const tinyusdz::PrimSpec* parent = nullptr;
            std::string err;
            if (!layer.find_primspec_at(tinyusdz::Path(parentPath, ""), &parent, &err) || !parent) {
                return nullptr;
            }
            auto mtlxFile = findMtlxRefOnSpec(*parent, baseDir, resolver);
            if (mtlxFile.empty()) return nullptr;

            return materialFromMtlxFile(mtlxFile, matLeaf, texLoader);
        }

        // Walk ancestors of meshAbsPath looking for a `material:binding` rel,
        // and return its raw target path string. (No resolution attempts.)
        std::string findRawMaterialBindingForMesh(
                const tinyusdz::Layer& layer,
                const std::string& meshAbsPath) {
            std::string path = meshAbsPath;
            while (!path.empty()) {
                const tinyusdz::PrimSpec* spec = nullptr;
                std::string err;
                if (layer.find_primspec_at(tinyusdz::Path(path, ""), &spec, &err) && spec) {
                    auto it = spec->props().find("material:binding");
                    if (it != spec->props().end() && it->second.is_relationship()) {
                        const auto& rel = it->second.get_relationship();
                        if (rel.is_path()) return rel.targetPath.prim_part();
                        if (rel.is_pathvector() && !rel.targetPathVector.empty())
                            return rel.targetPathVector[0].prim_part();
                    }
                }
                auto sp = path.rfind('/');
                if (sp == 0 || sp == std::string::npos) break;
                path = path.substr(0, sp);
            }
            return {};
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

            // diffuseColor / map (sRGB so the shader decodes the .jpg/.png).
            if (shader.diffuseColor.is_texture()) {
                if (auto t = getTex(shader.diffuseColor.texture_id)) {
                    t->colorSpace = ColorSpace::sRGB;
                    mat->map = t;
                }
            } else {
                const auto& c = shader.diffuseColor.value;
                mat->color.setRGB(c[0], c[1], c[2]);
            }

            // roughness (linear data).
            if (shader.roughness.is_texture()) {
                if (auto t = getTex(shader.roughness.texture_id)) {
                    t->colorSpace = ColorSpace::Linear;
                    mat->roughnessMap = t;
                }
            } else {
                mat->roughness = shader.roughness.value;
            }

            // metallic (linear data).
            if (shader.metallic.is_texture()) {
                if (auto t = getTex(shader.metallic.texture_id)) {
                    t->colorSpace = ColorSpace::Linear;
                    mat->metalnessMap = t;
                }
            } else {
                mat->metalness = shader.metallic.value;
            }

            // emissiveColor (sRGB).
            if (shader.emissiveColor.is_texture()) {
                if (auto t = getTex(shader.emissiveColor.texture_id)) {
                    t->colorSpace = ColorSpace::sRGB;
                    mat->emissiveMap = t;
                }
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

            // occlusion → aoMap (linear data).
            if (shader.occlusion.is_texture()) {
                if (auto t = getTex(shader.occlusion.texture_id)) {
                    t->colorSpace = ColorSpace::Linear;
                    mat->aoMap = t;
                }
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
                auto geometry = geometryFromRenderMesh(rm, scene);
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
        // Holds the parsed .usdz zip table during a load; the asset resolver
        // returned from SetupUSDZAssetResolution holds pointers into this so
        // it must outlive the ConvertToRenderScene call. Reset on each load.
        std::unique_ptr<tinyusdz::USDZAsset> _usdzAsset;
        // The latest load's DomeLight environment texture (if any) — exposed
        // to the caller via USDResult so they can assign it to scene.environment
        // / scene.background. Reset at the start of each load().
        std::shared_ptr<Texture> _envTexture;

        std::shared_ptr<Group> load(const std::filesystem::path& path) {
            _envTexture.reset();
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
            // Wildcard handler that fetches http(s) URLs via curl into a
            // local cache and falls through to tinyusdz's FindFile for
            // local paths. Lets Omniverse-style payloads to remote Nucleus
            // mirrors compose.
            registerHttpAwareResolver(resolver);

            // --- Load raw Layer (no composition yet) ---
            tinyusdz::Layer layer;
            std::string warn, err;
            bool ok;
            if (ext == ".usdz") {
                // USDZ is a zip. Load the Stage *and* register the zip's
                // asset table with our resolver so tinyusdz tydra can pull
                // texture bytes (`0/texgen_0.jpg` etc) from inside the zip
                // when ConvertToRenderScene runs. Without this the
                // RenderScene's TextureImage entries come back with
                // buffer_id = -1 / no width/height, and downstream
                // texture loading skips them.
                tinyusdz::Stage stageDirect;
                ok = tinyusdz::LoadUSDZFromFile(pathStr, &stageDirect, &warn, &err);
                if (!ok) {
                    std::cerr << "[USDLoader] Failed to load '" << pathStr << "': " << err << "\n";
                    return nullptr;
                }
                // USDZAsset must outlive the load — keep it on this Impl so
                // pointers held by the resolver remain valid through tydra.
                _usdzAsset = std::make_unique<tinyusdz::USDZAsset>();
                std::string zw, ze;
                if (!tinyusdz::ReadUSDZAssetInfoFromFile(pathStr, _usdzAsset.get(), &zw, &ze)) {
                    std::cerr << "[USDLoader] ReadUSDZAssetInfoFromFile failed for '"
                              << pathStr << "': " << ze << "\n";
                    // Fall through with an empty asset; textures inside the zip
                    // won't resolve, but geometry still loads.
                } else {
                    if (!tinyusdz::SetupUSDZAssetResolution(resolver, _usdzAsset.get())) {
                        std::cerr << "[USDLoader] SetupUSDZAssetResolution failed for '"
                                  << pathStr << "'\n";
                    }
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
            // Normalize list-edit qualifiers (Add/Append → Prepend; strip
            // Delete/Order) once per expansion: the original layer first, then
            // again after each successful composition pass so arcs introduced
            // by referenced files don't get silently dropped by tinyusdz's
            // CompositeReferences (which only honours ResetToExplicit and
            // Prepend).
            // NOTE: do NOT call normalizeArcsInLayer inside the
            // check_unresolved_references() predicate — the layer may be large
            // and calling it every iteration would stall on big scenes.
            normalizeArcsInLayer(layer);

            for (int pass = 0; pass < 8; ++pass) {
                bool progressed = false;

                if (layer.check_unresolved_references()) {
                    tinyusdz::Layer composed;
                    std::string cw, ce;
                    if (tinyusdz::CompositeReferences(resolver, layer, &composed, &cw, &ce)) {
                        layer = std::move(composed);
                        normalizeArcsInLayer(layer);
                        progressed = true;
                    }
                    // On failure (e.g. unsupported MaterialX subpath ref) keep
                    // going with whatever was composed in earlier passes —
                    // partial geometry is more useful than zero geometry.
                }

                if (layer.check_unresolved_payload()) {
                    tinyusdz::Layer composed;
                    std::string cw, ce;
                    if (tinyusdz::CompositePayload(resolver, layer, &composed, &cw, &ce)) {
                        layer = std::move(composed);
                        normalizeArcsInLayer(layer);
                        progressed = true;
                    }
                }

                if (!progressed) break;
            }

            // Apply variant selections before tydra runs. tinyusdz leaves
            // `variantSet` blocks unevaluated; meshes whose `material:binding`
            // only exists inside the selected variant (e.g. King's "Black"
            // block) show up unbound otherwise.
            applyVariantsInLayer(layer);

            // Expand PointInstancer prims into N instance Xforms so tydra
            // traverses them. Done after composition so referenced/payloaded
            // prototypes are present.
            expandPointInstancersInLayer(layer);

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

            if (!converter.ConvertToRenderScene(env, &renderScene)) {
                std::cerr << "[USDLoader] ConvertToRenderScene failed for '" << pathStr
                          << "': " << converter.GetError() << "\n";
                return nullptr;
            }

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
                // Compute the set of material IDs actually referenced by some
                // mesh (whole-mesh binding or per-face GeomSubset). Real-world
                // Omniverse scenes ship a master /Looks scope with 100+
                // materials, but each scene only binds a fraction of them —
                // the rest are unused defaults. Skipping the unused ones
                // avoids downloading several GB of textures for materials we'd
                // immediately throw away.
                std::unordered_set<int> usedMatIds;
                for (const auto& m : renderScene.meshes) {
                    if (m.material_id >= 0) usedMatIds.insert(m.material_id);
                    for (const auto& [_, sub] : m.material_subsetMap)
                        if (sub.material_id >= 0) usedMatIds.insert(sub.material_id);
                }

                // Case A — per-material-id overrides
                for (int mi = 0; mi < static_cast<int>(renderScene.materials.size()); ++mi) {
                    if (!usedMatIds.count(mi)) continue;
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
                                std::shared_ptr<MeshStandardMaterial> mat;

                                // 1. Standard MDL / UsdPreviewSurface flow:
                                //    findMaterialBindingInLayer resolves the
                                //    binding to an existing Material PrimSpec,
                                //    then materialFromMDLLayer extracts inputs.
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
                                    mat = cit->second;
                                }

                                // 2. MaterialX fallback: if the binding's
                                //    target prim doesn't exist (typical when
                                //    the parent Materials Scope references a
                                //    .mtlx file that tinyusdz couldn't fully
                                //    compose), try parsing the .mtlx ourselves.
                                if (!mat) {
                                    const std::string raw =
                                        findRawMaterialBindingForMesh(*layer, n.abs_path);
                                    if (!raw.empty()) {
                                        const std::string key = "MTLX:" + raw;
                                        auto cit = matPathCache.find(key);
                                        if (cit == matPathCache.end()) {
                                            auto mtlxMat = materialFromMtlxBinding(
                                                *layer, raw, baseDir,
                                                env.asset_resolver, texLoader);
                                            cit = matPathCache.emplace(key,
                                                                       std::move(mtlxMat)).first;
                                        }
                                        mat = cit->second;
                                    }
                                }

                                if (mat) meshPathMdlMap[n.abs_path] = mat;
                            }
                            for (const auto& c : n.children) buildMap(c);
                        };

                    for (const auto& n : renderScene.nodes) buildMap(n);
                }
            }

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

            // --- UsdLux lights ---
            // Walk the Stage with an XformNode tree so we have each light's
            // world transform, then map UsdLux types onto threepp lights and
            // attach them to the root group.
            addLightsFromStage(stage, *root, baseDir, resolver);

            if (root->children.empty()) {
                std::cerr << "[USDLoader] No usable geometry in '" << pathStr << "'\n";
                return nullptr;
            }

            return root;
        }

        // ----------------------------------------------------------------------
        // UsdLux → threepp light conversion.
        //
        //   DistantLight  → DirectionalLight (sun-like, parallel rays)
        //   DomeLight     → AmbientLight (HDR texture handling is left to the
        //                   caller — they typically set scene.environment to
        //                   their own HDR; doing it here would require
        //                   returning more than just a Group)
        //   SphereLight   → PointLight (we ignore radius — three.js doesn't
        //                   model point-light area)
        //   RectLight     → RectAreaLight (width/height in metres)
        //   DiskLight     → SpotLight (with a wide cone — closest approximation)
        //
        // Intensity uses USD's `intensity * 2^exposure` and `color`. Light
        // direction (DistantLight) and orientation (RectLight, DiskLight) come
        // from the world transform's rotation applied to the canonical local
        // -Z direction (UsdLux convention).
        // ----------------------------------------------------------------------

        void addLightsFromStage(const tinyusdz::Stage& stage, Group& root,
                                const std::filesystem::path& baseDir,
                                const tinyusdz::AssetResolutionResolver& resolver) {
            tinyusdz::tydra::XformNode xroot;
            std::string xerr;
            if (!tinyusdz::tydra::BuildXformNodeFromStage(stage, &xroot)) {
                return;  // no transform info; safest is to skip lights
            }

            std::function<void(const tinyusdz::tydra::XformNode&)> walk =
                [&](const tinyusdz::tydra::XformNode& xn) {
                    const tinyusdz::Prim* prim = xn.prim;
                    if (prim) {
                        // World matrix → threepp Matrix4 (USD row-vector to threepp column-vector).
                        Matrix4 wm = toMatrix4(xn.get_world_matrix());

                        // Read scalar values out of Animatable<T> wrappers.
                        auto readFloat = [](const auto& attr, float fallback = 0.f) {
                            float v = fallback;
                            attr.get_value().get_scalar(&v);
                            return v;
                        };
                        auto readColor = [](const auto& attr) {
                            tinyusdz::value::color3f c{1.f, 1.f, 1.f};
                            attr.get_value().get_scalar(&c);
                            return Color(c.r, c.g, c.b);
                        };
                        auto effectiveIntensity = [&](const auto& iAttr, const auto& eAttr) {
                            return readFloat(iAttr, 1.f) *
                                   std::pow(2.0f, readFloat(eAttr, 0.f));
                        };

                        // DistantLight → DirectionalLight
                        if (auto* dl = prim->as<tinyusdz::DistantLight>()) {
                            auto light = DirectionalLight::create(
                                readColor(dl->color),
                                effectiveIntensity(dl->intensity, dl->exposure));

                            // UsdLux DistantLight shines along its local -Z.
                            // Place threepp light at +Z (away from target) so
                            // its default (0,0,0) target gives rays travelling
                            // in -Z (after world rotation).
                            Vector3 dir(0, 0, -1);
                            dir.transformDirection(wm).normalize();
                            light->position.set(-dir.x, -dir.y, -dir.z);
                            root.add(light);
                        }
                        // DomeLight → AmbientLight + optional HDR env texture
                        // exposed via USDResult.environment.
                        else if (auto* dome = prim->as<tinyusdz::DomeLight>()) {
                            // DomeLight covers a full hemisphere; treat as
                            // ambient at half intensity so it doesn't double
                            // up with a scene.environment HDR set externally.
                            auto light = AmbientLight::create(
                                readColor(dome->color),
                                effectiveIntensity(dome->intensity, dome->exposure) * 0.5f);
                            root.add(light);

                            // If the DomeLight has `inputs:texture:file`,
                            // resolve and load it. Cache on _envTexture so
                            // USDResult.environment can hand it back.
                            if (!_envTexture && dome->file.authored()) {
                                auto fileOpt = dome->file.get_value();
                                tinyusdz::value::AssetPath ap;
                                const bool gotAp =
                                    fileOpt.has_value() &&
                                    fileOpt.value().get_scalar(&ap);
                                if (gotAp) {
                                    const std::string raw = ap.GetAssetPath();
                                    if (!raw.empty()) {
                                        const auto texPath = resolveAssetPath(
                                            raw, baseDir, baseDir, resolver);
                                        if (!texPath.empty()) {
                                            // .hdr → RGBELoader, otherwise
                                            // texLoader handles png/jpg/exr.
                                            std::string ext = texPath.extension().string();
                                            std::transform(ext.begin(), ext.end(), ext.begin(),
                                                [](unsigned char c) { return std::tolower(c); });
                                            if (ext == ".hdr") {
                                                RGBELoader rgbe;
                                                _envTexture = rgbe.load(texPath, true);
                                            } else {
                                                _envTexture = texLoader.load(texPath, true);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // SphereLight → PointLight (radius ignored)
                        else if (auto* sph = prim->as<tinyusdz::SphereLight>()) {
                            auto light = PointLight::create(
                                readColor(sph->color),
                                effectiveIntensity(sph->intensity, sph->exposure));
                            Vector3 pos, scale; Quaternion rot;
                            wm.decompose(pos, rot, scale);
                            light->position.copy(pos);
                            root.add(light);
                        }
                        // RectLight → RectAreaLight
                        else if (auto* rect = prim->as<tinyusdz::RectLight>()) {
                            auto light = RectAreaLight::create(
                                readColor(rect->color),
                                effectiveIntensity(rect->intensity, rect->exposure),
                                readFloat(rect->width,  1.f),
                                readFloat(rect->height, 1.f));
                            Vector3 pos, scale; Quaternion rot;
                            wm.decompose(pos, rot, scale);
                            light->position.copy(pos);
                            light->quaternion.copy(rot);
                            root.add(light);
                        }
                        // DiskLight → SpotLight (threepp has no disk-area light;
                        // use a moderate cone)
                        else if (auto* disk = prim->as<tinyusdz::DiskLight>()) {
                            auto light = SpotLight::create(
                                readColor(disk->color),
                                effectiveIntensity(disk->intensity, disk->exposure),
                                /*distance*/ 0.f,
                                /*angle*/ math::PI / 4.f,
                                /*penumbra*/ 0.5f,
                                /*decay*/ 1.f);
                            Vector3 pos, scale; Quaternion rot;
                            wm.decompose(pos, rot, scale);
                            light->position.copy(pos);
                            // Aim the SpotLight along the rotated local -Z.
                            // The light's `target` defaults to origin, so we
                            // create a child target Object3D positioned along
                            // that direction and bind it via setTarget().
                            Vector3 dir(0, 0, -1);
                            dir.applyQuaternion(rot).normalize();
                            auto tgt = std::make_shared<Object3D>();
                            tgt->position.set(pos.x + dir.x, pos.y + dir.y, pos.z + dir.z);
                            light->setTarget(*tgt);
                            root.add(light);
                            root.add(tgt);
                        }
                    }
                    for (const auto& c : xn.children) walk(c);
                };
            for (const auto& c : xroot.children) walk(c);
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
        auto group = pimpl_->load(path);
        return {std::move(group), {}, pimpl_->_envTexture};
    }

}// namespace threepp
