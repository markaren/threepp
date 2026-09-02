
#include "threepp/loaders/SogLoader.hpp"

#include "nlohmann/json.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/utils/ZipReader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    [[noreturn]] void fail(const std::string& msg) {

        throw std::runtime_error("SogLoader: " + msg);
    }

    // 64 palette entries per ROW, independent of the image width. The encoder
    // sizes the centroids image as width = 64 * shCoeffs, but the addressing is
    // defined by this constant, not by the width — see gotcha 6.
    constexpr unsigned int PALETTE_PER_ROW = 64;

    // A SOG asset is a set of files resolved by name, and it arrives either as a
    // directory tree or as a ZIP. Everything above this layer works in terms of
    // names relative to the asset root ("meta.json", "0_0/means_l.webp"), so the
    // two containers differ in exactly one method each and nothing else in the
    // file has to know which one it got.
    class Source {

    public:
        virtual ~Source() = default;
        [[nodiscard]] virtual bool has(const std::string& rel) const = 0;
        [[nodiscard]] virtual std::vector<unsigned char> read(const std::string& rel) const = 0;
        [[nodiscard]] virtual std::string describe() const = 0;
    };

    class DirSource: public Source {

    public:
        explicit DirSource(std::filesystem::path root): root_(std::move(root)) {}

        [[nodiscard]] bool has(const std::string& rel) const override {

            std::error_code ec;
            return std::filesystem::is_regular_file(root_ / rel, ec);
        }

        [[nodiscard]] std::vector<unsigned char> read(const std::string& rel) const override {

            const auto p = root_ / rel;
            // Constructed from the path object, never from path.string(): on
            // Windows the narrow overload decodes as ANSI and a scan living in
            // a folder with an accented name then fails to open.
            std::ifstream in(p, std::ios::binary | std::ios::ate);
            if (!in) fail("cannot open '" + p.string() + "'");

            const auto size = in.tellg();
            if (size < 0) fail("cannot size '" + p.string() + "'");
            in.seekg(0);

            std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
            in.read(reinterpret_cast<char*>(bytes.data()), size);
            if (in.gcount() != size) fail("truncated read of '" + p.string() + "'");
            return bytes;
        }

        [[nodiscard]] std::string describe() const override { return root_.string(); }

    private:
        std::filesystem::path root_;
    };

    class ZipSource: public Source {

    public:
        explicit ZipSource(const std::filesystem::path& archive)
            : zip_(archive), name_(archive.string()) {}

        [[nodiscard]] bool has(const std::string& rel) const override { return zip_.has(rel); }

        [[nodiscard]] std::vector<unsigned char> read(const std::string& rel) const override {

            return zip_.read(rel);
        }

        [[nodiscard]] std::string describe() const override { return name_; }

    private:
        ZipReader zip_;
        std::string name_;
    };

    // Where the asset root is, and whether it is one chunk or a whole pyramid.
    struct Resolved {

        std::unique_ptr<Source> source;
        bool multiLevel = false;// lod-meta.json present at the root
    };

    // A directory (or archive root) is probed for lod-meta.json FIRST, then
    // meta.json, so an asset root is never mistaken for a chunk.
    void probe(Resolved& r) {

        if (r.source->has("lod-meta.json")) {

            r.multiLevel = true;
            return;
        }
        if (r.source->has("meta.json")) {

            r.multiLevel = false;
            return;
        }
        fail("no lod-meta.json or meta.json in '" + r.source->describe() + "'");
    }

    Resolved resolve(const std::filesystem::path& path) {

        std::error_code ec;

        Resolved r;

        if (std::filesystem::is_directory(path, ec)) {

            r.source = std::make_unique<DirSource>(path);
            probe(r);
            return r;
        }

        if (!std::filesystem::is_regular_file(path, ec)) {

            fail("'" + path.string() + "' is neither a file nor a directory");
        }

        // An archive, recognised by its magic rather than its extension: the
        // same container ships as .zip from superspl.at and as .sog from the
        // spec's bundled form.
        if (ZipReader::looksLikeZip(path)) {

            r.source = std::make_unique<ZipSource>(path);
            probe(r);
            return r;
        }

        // A json named directly. Its parent is the root.
        const auto name = path.filename().string();
        if (name == "lod-meta.json" || name == "meta.json") {

            r.source = std::make_unique<DirSource>(path.parent_path());
            r.multiLevel = (name == "lod-meta.json");
            return r;
        }

        fail("'" + path.string() + "' is not a SOG asset (expected a directory, a "
                                   "meta.json/lod-meta.json, or a zip archive)");
    }

    nlohmann::json parseJson(const Source& src, const std::string& rel) {

        const auto bytes = src.read(rel);
        try {

            return nlohmann::json::parse(bytes.begin(), bytes.end());

        } catch (const std::exception& e) {

            fail("'" + rel + "' is not valid JSON: " + e.what());
        }
    }

    // Mirrors parsePly's require() so a missing member produces one uniform
    // message instead of an nlohmann type_error naming a JSON pointer nobody
    // can act on.
    const nlohmann::json& require(const nlohmann::json& j, const char* member, const std::string& where) {

        const auto it = j.find(member);
        if (it == j.end()) fail(where + ": required member '" + std::string(member) + "' missing");
        return *it;
    }

    std::vector<float> requireCodebook(const nlohmann::json& section, const std::string& where) {

        const auto& cb = require(section, "codebook", where);
        if (!cb.is_array() || cb.size() != 256) {

            fail(where + ": codebook has " + std::to_string(cb.is_array() ? cb.size() : 0) +
                 " entries, expected 256");
        }
        return cb.get<std::vector<float>>();
    }

    std::vector<std::string> requireFiles(const nlohmann::json& section, std::size_t arity,
                                          const std::string& where) {

        const auto& f = require(section, "files", where);
        if (!f.is_array() || f.size() != arity) {

            fail(where + ": files[] has " + std::to_string(f.is_array() ? f.size() : 0) +
                 " entries, expected " + std::to_string(arity));
        }
        return f.get<std::vector<std::string>>();
    }

    // Everything one chunk's meta.json says, validated before anything indexes
    // on it. Plane paths come from each section's files[] array and are never
    // hardcoded — which is also what lets the test writer emit PNG planes under
    // the .webp names its own meta.json declares.
    struct ChunkMeta {

        std::size_t count = 0;
        int shDegree = 0;// = shN.bands, 0 when shN is absent

        float meansMin[3]{};
        float meansMax[3]{};

        std::vector<float> scalesCodebook;
        std::vector<float> sh0Codebook;
        std::vector<float> shNCodebook;
        std::size_t palette = 0;

        std::string prefix;// "" for a lone chunk, "0_0/" inside a pyramid
        std::string meansLo, meansHi, scales, quats, sh0, shNCentroids, shNLabels;
    };

    ChunkMeta parseChunkMeta(const Source& src, const std::string& prefix) {

        const std::string rel = prefix + "meta.json";
        const auto j = parseJson(src, rel);

        ChunkMeta m;
        m.prefix = prefix;

        const auto& version = require(j, "version", rel);
        if (!version.is_number_integer() || version.get<int>() != 2) {

            fail(rel + ": version is " + version.dump() + ", expected 2");
        }

        m.count = require(j, "count", rel).get<std::size_t>();
        if (m.count == 0) fail(rel + ": count is 0");

        {
            const auto& means = require(j, "means", rel);
            const auto& mins = require(means, "mins", rel + " means");
            const auto& maxs = require(means, "maxs", rel + " means");
            if (!mins.is_array() || mins.size() != 3 || !maxs.is_array() || maxs.size() != 3) {

                fail(rel + " means: mins/maxs must each have 3 entries");
            }
            for (int i = 0; i < 3; ++i) {

                m.meansMin[i] = mins[i].get<float>();
                m.meansMax[i] = maxs[i].get<float>();
            }
            const auto files = requireFiles(means, 2, rel + " means");
            m.meansLo = prefix + files[0];
            m.meansHi = prefix + files[1];
        }

        {
            const auto& scales = require(j, "scales", rel);
            m.scalesCodebook = requireCodebook(scales, rel + " scales");
            m.scales = prefix + requireFiles(scales, 1, rel + " scales")[0];
        }

        {
            // quats carries NO codebook — a generic "every section has one"
            // loop throws on the real asset.
            const auto& quats = require(j, "quats", rel);
            m.quats = prefix + requireFiles(quats, 1, rel + " quats")[0];
        }

        {
            const auto& sh0 = require(j, "sh0", rel);
            m.sh0Codebook = requireCodebook(sh0, rel + " sh0");
            m.sh0 = prefix + requireFiles(sh0, 1, rel + " sh0")[0];
        }

        // shN is optional: absent means degree 0 and no palette.
        if (const auto it = j.find("shN"); it != j.end()) {

            const auto& shN = *it;
            const int bands = require(shN, "bands", rel + " shN").get<int>();
            if (bands < 1 || bands > splats::MAX_SH_DEGREE) {

                fail(rel + " shN: bands is " + std::to_string(bands) + ", expected 1..3");
            }
            m.shDegree = bands;
            m.palette = require(shN, "count", rel + " shN").get<std::size_t>();
            if (m.palette == 0) fail(rel + " shN: count is 0");

            m.shNCodebook = requireCodebook(shN, rel + " shN");
            const auto files = requireFiles(shN, 2, rel + " shN");
            m.shNCentroids = prefix + files[0];
            m.shNLabels = prefix + files[1];
        }

        // `antialias`, when present, is a scalar render-model flag. It is read
        // and dropped rather than carried: SplatData::extras is validated as
        // per-splat arrays of length count(), so a scalar has no home there, and
        // nothing in threepp's rasterizers consumes it.

        return m;
    }

    // THE ONLY place ImageLoader is called from this file. flipY is hardcoded
    // false and is deliberately not a parameter: seven call sites each one bool
    // away from a silently wrong decode is six too many (gotcha 1).
    Image loadPlane(const Source& src, const std::string& rel) {

        const auto bytes = src.read(rel);
        auto img = ImageLoader().load(bytes, 4, false);
        if (!img) fail("'" + rel + "' could not be decoded as an image");

        if (img->isFloat() || img->isHalfFloat()) fail("'" + rel + "' did not decode to 8-bit data");
        if (img->channels() != 4) {

            fail("'" + rel + "' decoded to " + std::to_string(img->channels()) +
                 " channels, expected 4");
        }
        return std::move(*img);
    }

    const unsigned char* pixels(const Image& img) {

        return img.data().data();
    }

    void requireArea(const Image& img, std::size_t count, const std::string& rel) {

        const std::size_t area = static_cast<std::size_t>(img.width()) * img.height();
        if (area < count) {

            fail("'" + rel + "': " + std::to_string(img.width()) + "x" + std::to_string(img.height()) +
                 " holds " + std::to_string(area) + " pixels, fewer than the declared count " +
                 std::to_string(count));
        }
    }

    // Decodes [0, meta.count) of one chunk directly into out[dstFirst...].
    // `out` is already sized and carries the level's degree; this neither
    // normalizes nor validates, because the caller does both once when the whole
    // level is assembled.
    void decodeChunkInto(const Source& src, const ChunkMeta& m, SplatData& out, std::size_t dstFirst) {

        const auto lo = loadPlane(src, m.meansLo);
        const auto hi = loadPlane(src, m.meansHi);
        const auto sc = loadPlane(src, m.scales);
        const auto qt = loadPlane(src, m.quats);
        const auto s0 = loadPlane(src, m.sh0);

        requireArea(lo, m.count, m.meansLo);

        // Every per-splat plane is addressed by ONE index i -> (i % w, i / w),
        // so they must agree on the grid. They do on every real chunk; a writer
        // that disagreed would otherwise silently pair splat i's mean with a
        // different splat's rotation.
        const unsigned int width = lo.width();
        const unsigned int height = lo.height();

        const std::pair<const Image*, const std::string*> planes[]{
                {&hi, &m.meansHi}, {&sc, &m.scales}, {&qt, &m.quats}, {&s0, &m.sh0}};

        for (const auto& [img, name] : planes) {

            if (img->width() != width || img->height() != height) {

                fail("'" + *name + "' is " + std::to_string(img->width()) + "x" +
                     std::to_string(img->height()) + " but '" + m.meansLo + "' is " +
                     std::to_string(width) + "x" + std::to_string(height) +
                     " (all per-splat planes share one grid)");
            }
        }
        requireArea(qt, m.count, m.quats);

        const unsigned char* pLo = pixels(lo);
        const unsigned char* pHi = pixels(hi);
        const unsigned char* pSc = pixels(sc);
        const unsigned char* pQt = pixels(qt);
        const unsigned char* pS0 = pixels(s0);

        const int coeffCount = out.coeffCount();

        // The higher-order palette, decoded once per chunk.
        std::vector<unsigned char> labels;
        std::vector<float> centroids;// [palette][shCoeffs][3]
        int shCoeffs = 0;

        if (m.shDegree > 0) {

            shCoeffs = splats::shCoeffCount(m.shDegree) - 1;

            const auto ce = loadPlane(src, m.shNCentroids);
            const auto lb = loadPlane(src, m.shNLabels);
            if (lb.width() != width || lb.height() != height) {

                fail("'" + m.shNLabels + "' is " + std::to_string(lb.width()) + "x" +
                     std::to_string(lb.height()) + " but '" + m.meansLo + "' is " +
                     std::to_string(width) + "x" + std::to_string(height) +
                     " (all per-splat planes share one grid)");
            }

            const std::size_t rows = (m.palette + PALETTE_PER_ROW - 1) / PALETTE_PER_ROW;
            const std::size_t needW = PALETTE_PER_ROW * static_cast<std::size_t>(shCoeffs);
            if (ce.width() < needW || ce.height() < rows) {

                fail("'" + m.shNCentroids + "': " + std::to_string(ce.width()) + "x" +
                     std::to_string(ce.height()) + " is too small for " + std::to_string(m.palette) +
                     " palette entries at " + std::to_string(shCoeffs) + " coefficients (need at least " +
                     std::to_string(needW) + "x" + std::to_string(rows) + ")");
            }

            const unsigned char* pCe = pixels(ce);
            centroids.resize(m.palette * static_cast<std::size_t>(shCoeffs) * 3);

            for (std::size_t n = 0; n < m.palette; ++n) {

                const std::size_t v = n / PALETTE_PER_ROW;
                const std::size_t uBase = (n % PALETTE_PER_ROW) * static_cast<std::size_t>(shCoeffs);

                for (int c = 0; c < shCoeffs; ++c) {

                    const std::size_t px = (v * ce.width() + uBase + static_cast<std::size_t>(c)) * 4;
                    float* dst = centroids.data() + (n * static_cast<std::size_t>(shCoeffs) + c) * 3;
                    for (int ch = 0; ch < 3; ++ch) dst[ch] = m.shNCodebook[pCe[px + ch]];
                }
            }

            const unsigned char* pLb = pixels(lb);
            labels.assign(pLb, pLb + static_cast<std::size_t>(lb.width()) * lb.height() * 4);
        }

        for (std::size_t i = 0; i < m.count; ++i) {

            const std::size_t px = i * 4;// planes share one width, so one index serves all
            const std::size_t o = dstFirst + i;

            // Gotcha 3: the quantised value is in LOG space over [mins, maxs],
            // and attains both endpoints — so 65535, not 65536.
            for (int a = 0; a < 3; ++a) {

                const unsigned int q = (static_cast<unsigned int>(pHi[px + a]) << 8) | pLo[px + a];
                const float t = static_cast<float>(q) / 65535.f;
                const float n = m.meansMin[a] + (m.meansMax[a] - m.meansMin[a]) * t;
                const float p = std::copysign(std::expm1(std::fabs(n)), n);
                if (a == 0) out.means[o].x = p;
                else if (a == 1) out.means[o].y = p;
                else out.means[o].z = p;
            }

            out.scales[o].set(std::exp(m.scalesCodebook[pSc[px + 0]]),
                              std::exp(m.scalesCodebook[pSc[px + 1]]),
                              std::exp(m.scalesCodebook[pSc[px + 2]]));

            // Gotcha 2 + gotcha 4, and the one tripwire in this file. Valid mode
            // bytes are exactly {252, 253, 254, 255} across a real chunk, while
            // the padding past `count` carries alpha 0 — so this single check
            // turns BOTH of the two worst silent bugs (flipY left true, and
            // iterating width*height) into a named exception on the first splat,
            // before the -252 index into a four-element tuple can happen. It is
            // load-bearing; do not relax it into a clamp.
            const unsigned int mode = pQt[px + 3];
            if (mode < 252u || mode > 255u) {

                fail("'" + m.quats + "': splat " + std::to_string(i) + " has mode byte " +
                     std::to_string(mode) + ", expected 252..255 (a flipped or mis-sized image "
                                            "reads padding as data)");
            }

            constexpr float SCALE = 1.4142135623730951f;// sqrt(2)
            float wxyz[4];
            const int omitted = static_cast<int>(mode - 252u);
            float sum = 0.f;
            for (int k = 0, s = 0; k < 4; ++k) {

                if (k == omitted) continue;
                const float c = (static_cast<float>(pQt[px + s]) / 255.f - 0.5f) * 2.f / SCALE;
                wxyz[k] = c;
                sum += c * c;
                ++s;
            }
            wxyz[omitted] = std::sqrt(std::max(0.f, 1.f - sum));

            // The tuple is (w, x, y, z); threepp's Quaternion is (x, y, z, w).
            out.rotations[o].set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);

            // Gotcha 5: alpha is already-activated opacity, not a logit.
            out.opacities[o] = static_cast<float>(pS0[px + 3]) / 255.f;

            float* c = out.shAt(o);
            // Gotcha 5 again: raw DC coefficients, not colours. No 0.5 + c*SH_C0
            // and no clamp — the large values are the sky smears removeOutliers
            // exists to find.
            for (int ch = 0; ch < 3; ++ch) c[ch] = m.sh0Codebook[pS0[px + ch]];

            if (m.shDegree > 0) {

                const std::size_t lpx = i * 4;
                const std::size_t label = static_cast<std::size_t>(labels[lpx + 0]) |
                                          (static_cast<std::size_t>(labels[lpx + 1]) << 8);
                if (label >= m.palette) {

                    fail("'" + m.shNLabels + "': splat " + std::to_string(i) + " references palette entry " +
                         std::to_string(label) + " of " + std::to_string(m.palette));
                }

                // Gotcha 6: already coefficient-major with rgb in the pixel,
                // which is SplatData's own layout. NO channel-major reorder —
                // this is the one line of SplatLoader's PLY path that must not
                // be copied.
                const float* srcCoeff = centroids.data() + label * static_cast<std::size_t>(shCoeffs) * 3;
                for (int k = 0; k < shCoeffs && (1 + k) < coeffCount; ++k) {

                    for (int ch = 0; ch < 3; ++ch) c[(1 + k) * 3 + ch] = srcCoeff[k * 3 + ch];
                }
            }
        }
    }

    // ── lod-meta.json ───────────────────────────────────────────────────────
    //
    // The tree is the only thing in the file that says which chunk belongs to
    // which level. Each of the 1,386 leaves carries a `lods` object keyed by
    // level, and each entry names a `file` index into `filenames` plus an
    // (offset, count) range inside that chunk. Directory NAMES are never parsed:
    // `filenames` is not even in numeric order — its index 2 is "3_0/meta.json".
    struct LodMeta {

        std::vector<std::string> filenames;// as declared, e.g. "0_0/meta.json"
        std::vector<std::size_t> counts;   // declared splats per level
        int lodLevels = 0;
        Box3 bound;

        // level -> file index -> (summed count, accumulated bound)
        std::vector<std::map<std::size_t, std::pair<std::size_t, Box3>>> byLevel;

        // The leaves themselves, in tree order, with the (file, offset, count)
        // each level's entry names. `offset` is what makes per-node LOD
        // possible; `haveOffsets` goes false the moment one entry omits it,
        // and the leaves are then dropped rather than half-trusted.
        struct LeafLod {

            int level{};
            std::size_t file{};
            std::size_t offset{};
            std::size_t count{};
        };
        struct Leaf {

            Box3 bound;
            std::vector<LeafLod> lods;
        };
        std::vector<Leaf> leaves;
        bool haveOffsets = true;
    };

    Box3 boundOf(const nlohmann::json& node, const std::string& where) {

        const auto& b = require(node, "bound", where);
        const auto& mn = require(b, "min", where + " bound");
        const auto& mx = require(b, "max", where + " bound");
        if (!mn.is_array() || mn.size() != 3 || !mx.is_array() || mx.size() != 3) {

            fail(where + " bound: min/max must each have 3 entries");
        }
        return Box3(Vector3(mn[0].get<float>(), mn[1].get<float>(), mn[2].get<float>()),
                    Vector3(mx[0].get<float>(), mx[1].get<float>(), mx[2].get<float>()));
    }

    void mergeBound(Box3& into, const Box3& b, bool& first) {

        if (first) {

            into = b;
            first = false;
            return;
        }
        into.set(Vector3(std::min(into.min().x, b.min().x),
                         std::min(into.min().y, b.min().y),
                         std::min(into.min().z, b.min().z)),
                 Vector3(std::max(into.max().x, b.max().x),
                         std::max(into.max().y, b.max().y),
                         std::max(into.max().z, b.max().z)));
    }

    void walkTree(const nlohmann::json& node, LodMeta& out, std::vector<std::vector<bool>>& seeded) {

        const auto kids = node.find("children");
        if (kids != node.end() && kids->is_array() && !kids->empty()) {

            for (const auto& child : *kids) walkTree(child, out, seeded);
            return;
        }

        const auto lods = node.find("lods");
        if (lods == node.end() || !lods->is_object()) return;

        const Box3 nodeBound = boundOf(node, "lod-meta.json tree leaf");

        LodMeta::Leaf leaf;
        leaf.bound = nodeBound;

        for (auto it = lods->begin(); it != lods->end(); ++it) {

            const int level = std::stoi(it.key());
            if (level < 0 || level >= out.lodLevels) {

                fail("lod-meta.json: tree leaf names level " + it.key() + ", but lodLevels is " +
                     std::to_string(out.lodLevels));
            }

            const auto& entry = it.value();
            const auto file = require(entry, "file", "lod-meta.json tree leaf").get<std::size_t>();
            const auto count = require(entry, "count", "lod-meta.json tree leaf").get<std::size_t>();

            if (file >= out.filenames.size()) {

                fail("lod-meta.json: tree leaf references file " + std::to_string(file) + " of " +
                     std::to_string(out.filenames.size()));
            }

            auto& slot = out.byLevel[static_cast<std::size_t>(level)][file];
            slot.first += count;

            bool first = !seeded[static_cast<std::size_t>(level)][file];
            mergeBound(slot.second, nodeBound, first);
            seeded[static_cast<std::size_t>(level)][file] = true;

            const auto offsetIt = entry.find("offset");
            if (offsetIt == entry.end() || !offsetIt->is_number_unsigned()) {

                out.haveOffsets = false;
            } else {

                LodMeta::LeafLod ll;
                ll.level = level;
                ll.file = file;
                ll.offset = offsetIt->get<std::size_t>();
                ll.count = count;
                leaf.lods.push_back(ll);
            }
        }

        // Ascending level, so a consumer can walk a leaf's levels finest-first
        // without re-sorting. (nlohmann's object iteration is by string key,
        // which agrees for one-digit levels and stops agreeing at ten.)
        std::sort(leaf.lods.begin(), leaf.lods.end(),
                  [](const LodMeta::LeafLod& a, const LodMeta::LeafLod& b) { return a.level < b.level; });
        out.leaves.push_back(std::move(leaf));
    }

    LodMeta parseLodMeta(const Source& src) {

        const auto j = parseJson(src, "lod-meta.json");

        LodMeta lm;
        lm.filenames = require(j, "filenames", "lod-meta.json").get<std::vector<std::string>>();
        lm.counts = require(j, "counts", "lod-meta.json").get<std::vector<std::size_t>>();
        lm.lodLevels = require(j, "lodLevels", "lod-meta.json").get<int>();

        if (lm.lodLevels < 1) fail("lod-meta.json: lodLevels is " + std::to_string(lm.lodLevels));
        if (lm.counts.size() != static_cast<std::size_t>(lm.lodLevels)) {

            fail("lod-meta.json: counts[] has " + std::to_string(lm.counts.size()) +
                 " entries but lodLevels is " + std::to_string(lm.lodLevels));
        }

        lm.bound = boundOf(require(j, "tree", "lod-meta.json"), "lod-meta.json tree");
        lm.byLevel.resize(static_cast<std::size_t>(lm.lodLevels));

        std::vector<std::vector<bool>> seeded(static_cast<std::size_t>(lm.lodLevels),
                                              std::vector<bool>(lm.filenames.size(), false));
        walkTree(require(j, "tree", "lod-meta.json"), lm, seeded);

        // A wrong file-to-level map would otherwise quietly render a subset.
        // The tree's own arithmetic has to agree with the declared counts.
        for (std::size_t l = 0; l < lm.byLevel.size(); ++l) {

            std::size_t sum = 0;
            for (const auto& [file, slot] : lm.byLevel[l]) sum += slot.first;

            if (sum != lm.counts[l]) {

                fail("lod-meta.json: level " + std::to_string(l) + " tree ranges sum to " +
                     std::to_string(sum) + " but counts[" + std::to_string(l) + "] declares " +
                     std::to_string(lm.counts[l]));
            }
        }

        // No chunk may belong to two levels — the levels are alternatives.
        std::map<std::size_t, std::size_t> owner;
        for (std::size_t l = 0; l < lm.byLevel.size(); ++l) {

            for (const auto& [file, slot] : lm.byLevel[l]) {

                const auto [it, inserted] = owner.emplace(file, l);
                if (!inserted && it->second != l) {

                    fail("lod-meta.json: '" + lm.filenames[file] + "' is claimed by both level " +
                         std::to_string(it->second) + " and level " + std::to_string(l));
                }
            }
        }

        return lm;
    }

    // "0_0/meta.json" -> "0_0/". A chunk's planes resolve against its own
    // directory, whatever the writer chose to call it.
    std::string prefixOf(const std::string& metaPath) {

        const auto slash = metaPath.find_last_of('/');
        return slash == std::string::npos ? std::string{} : metaPath.substr(0, slash + 1);
    }

}// namespace


SplatData SogLoader::load(const std::filesystem::path& path) {

    return load(path, Options{});
}

SplatData SogLoader::load(const std::filesystem::path& path, const Options& options) {

    const auto resolved = resolve(path);
    const Source& src = *resolved.source;

    std::vector<ChunkMeta> chunks;

    if (!resolved.multiLevel) {

        if (options.lod != 0) {

            fail("lod " + std::to_string(options.lod) + " requested, but '" + src.describe() +
                 "' is a single chunk and declares no levels");
        }
        chunks.push_back(parseChunkMeta(src, ""));

    } else {

        const auto lm = parseLodMeta(src);

        if (options.lod < 0 || options.lod >= lm.lodLevels) {

            fail("lod " + std::to_string(options.lod) + " requested, but the asset declares " +
                 std::to_string(lm.lodLevels) + " level(s)");
        }

        // Ascending file index, which on every writer seen so far is also
        // numeric chunk order — and, for level 0, the order the scan's own
        // scene.ply concatenates its chunks in.
        for (const auto& [file, slot] : lm.byLevel[static_cast<std::size_t>(options.lod)]) {

            auto meta = parseChunkMeta(src, prefixOf(lm.filenames[file]));

            if (meta.count != slot.first) {

                fail("'" + lm.filenames[file] + "': meta.json declares count " +
                     std::to_string(meta.count) + " but the lod-meta tree ranges sum to " +
                     std::to_string(slot.first));
            }
            chunks.push_back(std::move(meta));
        }
    }

    if (chunks.empty()) fail("level " + std::to_string(options.lod) + " has no chunks");

    // Merging a bands-1 chunk into a bands-3 buffer silently shifts every
    // coefficient, so disagreement is an error rather than something to
    // reconcile.
    const int degree = chunks.front().shDegree;
    std::size_t total = 0;
    for (const auto& c : chunks) {

        if (c.shDegree != degree) {

            fail("'" + c.prefix + "meta.json' is SH degree " + std::to_string(c.shDegree) +
                 " but '" + chunks.front().prefix + "meta.json' is degree " + std::to_string(degree) +
                 " (all chunks of a level must agree)");
        }
        total += c.count;
    }

    // Build, do not concatenate: appending nine SplatData peaks near twice a
    // 1.2 GB result. One resize, then each chunk decodes straight into its own
    // slice, and its seven decoded planes are released before the next opens.
    SplatData data;
    data.resize(total, degree);

    std::size_t first = 0;
    for (const auto& c : chunks) {

        decodeChunkInto(src, c, data, first);
        first += c.count;
    }

    data.normalizeRotations();

    std::string why;
    if (!data.validate(&why)) fail("internal consistency check failed: " + why);

    return data;
}

SogLoader::Info SogLoader::describe(const std::filesystem::path& path) {

    const auto resolved = resolve(path);
    const Source& src = *resolved.source;

    Info info;

    if (!resolved.multiLevel) {

        const auto m = parseChunkMeta(src, "");
        info.lodLevels = 1;
        info.shDegree = m.shDegree;

        LevelInfo lvl;
        lvl.lod = 0;
        lvl.count = m.count;
        lvl.chunks.push_back(ChunkInfo{"", m.count, Box3()});
        info.levels.push_back(std::move(lvl));
        return info;
    }

    const auto lm = parseLodMeta(src);

    info.lodLevels = lm.lodLevels;
    info.bound = lm.bound;

    for (int l = 0; l < lm.lodLevels; ++l) {

        LevelInfo lvl;
        lvl.lod = l;
        lvl.count = lm.counts[static_cast<std::size_t>(l)];

        for (const auto& [file, slot] : lm.byLevel[static_cast<std::size_t>(l)]) {

            lvl.chunks.push_back(ChunkInfo{prefixOf(lm.filenames[file]), slot.first, slot.second});
        }
        info.levels.push_back(std::move(lvl));
    }

    // The tree's leaves, with each level's file index turned into an index into
    // that level's own chunk list — the order load() decodes them in, and the
    // only form a caller can turn into an absolute offset without re-reading
    // lod-meta.json. Dropped wholesale when any entry lacked an `offset`: a
    // partial tree would address some nodes right and some nodes wrong, which
    // is worse than not having one.
    if (lm.haveOffsets) {

        std::vector<std::map<std::size_t, std::size_t>> chunkIndex(static_cast<std::size_t>(lm.lodLevels));
        for (std::size_t l = 0; l < chunkIndex.size(); ++l) {

            std::size_t k = 0;
            for (const auto& [file, slot] : lm.byLevel[l]) chunkIndex[l][file] = k++;
        }

        info.nodes.reserve(lm.leaves.size());
        for (const auto& leaf : lm.leaves) {

            NodeInfo n;
            n.bound = leaf.bound;
            for (const auto& e : leaf.lods) {

                NodeRange r;
                r.lod = e.level;
                r.chunk = chunkIndex[static_cast<std::size_t>(e.level)].at(e.file);
                r.offset = e.offset;
                r.count = e.count;
                n.lods.push_back(r);
            }
            info.nodes.push_back(std::move(n));
        }
    }

    // One chunk's meta.json for the degree; they are required to agree.
    if (!lm.byLevel.empty() && !lm.byLevel[0].empty()) {

        info.shDegree = parseChunkMeta(src, prefixOf(lm.filenames[lm.byLevel[0].begin()->first])).shDegree;
    }

    return info;
}

bool SogLoader::isSog(const std::filesystem::path& path) {

    try {

        const auto resolved = resolve(path);
        const Source& src = *resolved.source;

        if (resolved.multiLevel) {

            // lod-meta.json is 502 KB on the real asset, almost all of it
            // octree, so the sniff deliberately does not parse it — presence
            // plus the chunk it points at is enough to claim the asset.
            return true;
        }

        const auto j = parseJson(src, "meta.json");
        const auto version = j.find("version");
        if (version == j.end() || !version->is_number_integer() || version->get<int>() != 2) return false;

        const auto means = j.find("means");
        return means != j.end() && means->contains("mins") && means->contains("maxs") &&
               means->contains("files");

    } catch (...) {

        // A missing, unreadable or malformed asset is simply "not a SOG", which
        // leaves the caller's existing paths to report the failure in their own
        // words rather than pre-empting them with a SOG-flavoured complaint.
        return false;
    }
}
