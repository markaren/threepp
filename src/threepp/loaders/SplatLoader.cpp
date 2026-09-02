
#include "threepp/loaders/SplatLoader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace threepp;

namespace {

    // PLY scalar types. The splat parser *consumes* only float, but every
    // type has to be recognised so a property it skips still contributes the
    // right number of bytes to the stride; the point-cloud parser reads all
    // of them and needs the kind to normalise a colour.
    enum class ScalarKind { Signed, Unsigned, Float };

    struct ScalarType {
        const char* name;
        int size;
        ScalarKind kind;
    };

    constexpr ScalarType SCALAR_TYPES[] = {
            {"char", 1, ScalarKind::Signed}, {"int8", 1, ScalarKind::Signed},
            {"uchar", 1, ScalarKind::Unsigned}, {"uint8", 1, ScalarKind::Unsigned},
            {"short", 2, ScalarKind::Signed}, {"int16", 2, ScalarKind::Signed},
            {"ushort", 2, ScalarKind::Unsigned}, {"uint16", 2, ScalarKind::Unsigned},
            {"int", 4, ScalarKind::Signed}, {"int32", 4, ScalarKind::Signed},
            {"uint", 4, ScalarKind::Unsigned}, {"uint32", 4, ScalarKind::Unsigned},
            {"float", 4, ScalarKind::Float}, {"float32", 4, ScalarKind::Float},
            {"double", 8, ScalarKind::Float}, {"float64", 8, ScalarKind::Float}};

    const ScalarType* scalarType(const std::string& type) {

        for (const auto& t : SCALAR_TYPES) {

            if (type == t.name) return &t;
        }
        return nullptr;
    }

    int scalarSize(const std::string& type) {

        const auto* t = scalarType(type);
        return t ? t->size : 0;
    }

    bool isFloat32(const std::string& type) {

        return type == "float" || type == "float32";
    }

    struct Property {
        std::string name;
        std::string type;
        size_t offset{};
        int size{};
    };

    [[noreturn]] void fail(const std::string& msg) {

        throw std::runtime_error("SplatLoader: " + msg);
    }

    // Reads one header line, tolerating both LF and CRLF. The header is ASCII
    // even in a binary PLY, so std::getline is the right tool — but the binary
    // body starts immediately after "end_header\n", so we must not read past it.
    bool headerLine(std::istream& in, std::string& line) {

        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    }

    std::vector<std::string> tokens(const std::string& line) {

        std::istringstream iss(line);
        std::vector<std::string> out;
        std::string tok;
        while (iss >> tok) out.push_back(tok);
        return out;
    }

    // Number of higher-order SH coefficients per channel for a degree:
    // 0, 3, 8, 15 — i.e. shCoeffCount(degree) - 1.
    int restCoeffsForDegree(int degree) {

        return splats::shCoeffCount(degree) - 1;
    }

    int degreeFromRestCount(size_t restCount) {

        for (int d = 0; d <= splats::MAX_SH_DEGREE; ++d) {

            if (restCount == static_cast<size_t>(restCoeffsForDegree(d)) * 3) return d;
        }

        fail("f_rest_* count " + std::to_string(restCount) +
             " matches no supported SH degree (expected 0, 9, 24 or 45)");
    }

    float readFloat(const unsigned char* base, size_t offset) {

        float v;
        std::memcpy(&v, base + offset, sizeof(float));
        return v;
    }

}// namespace


SplatData SplatLoader::loadPly(const std::filesystem::path& path) {

    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open '" + path.string() + "'");

    try {

        return parsePly(in);

    } catch (const std::runtime_error& e) {

        // Re-throw with the file named — a bare "unsupported property" is
        // useless when a scene loads twenty clouds.
        throw std::runtime_error(std::string(e.what()) + " (in '" + path.string() + "')");
    }
}

bool SplatLoader::isSplatPly(const std::filesystem::path& path) {

    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    return isSplatPly(in);
}

bool SplatLoader::isSplatPly(std::istream& stream) {

    std::string line;
    if (!headerLine(stream, line) || line != "ply") return false;

    // A bound on the scan rather than a trust in end_header: a truncated or
    // non-PLY file that happens to start with the magic must not turn this
    // into a read of the whole thing. Real headers are ~70 lines at SH
    // degree 3; a thousand is generous and still bounded.
    constexpr int MAX_HEADER_LINES = 1000;

    for (int i = 0; i < MAX_HEADER_LINES && headerLine(stream, line); ++i) {

        const auto tok = tokens(line);
        if (tok.empty()) continue;
        if (tok[0] == "end_header") return false;
        // "property float f_dc_0" — the name is the last token, and only the
        // name is checked. The type is parsePly's business to reject; a file
        // that declares f_dc_0 as a double is a splat file this loader cannot
        // read, and it should say so rather than be quietly handed to a mesh
        // parser that will say something else.
        if (tok[0] == "property" && tok.back() == "f_dc_0") return true;
    }
    return false;
}

SplatData SplatLoader::parsePly(std::istream& stream) {

    std::string line;

    if (!headerLine(stream, line) || line != "ply") fail("not a PLY file (missing 'ply' magic)");

    if (!headerLine(stream, line)) fail("truncated header");
    if (line != "format binary_little_endian 1.0") {

        fail("unsupported format '" + line + "' (only binary_little_endian 1.0)");
    }

    size_t vertexCount = 0;
    bool sawVertexElement = false;
    std::vector<Property> properties;
    size_t stride = 0;

    while (true) {

        if (!headerLine(stream, line)) fail("truncated header (no end_header)");

        const auto tok = tokens(line);
        if (tok.empty()) continue;

        if (tok[0] == "comment" || tok[0] == "obj_info") continue;

        if (tok[0] == "end_header") break;

        if (tok[0] == "element") {

            if (tok.size() < 3) fail("malformed element line '" + line + "'");

            if (sawVertexElement) {

                // A second element means the vertex block has ended. We read
                // only the vertex block, which the format guarantees comes
                // first here, so anything after it is simply ignored.
                break;
            }

            if (tok[1] != "vertex") {

                fail("first element is '" + tok[1] + "', expected 'vertex'");
            }

            sawVertexElement = true;
            vertexCount = static_cast<size_t>(std::stoull(tok[2]));
            continue;
        }

        if (tok[0] == "property") {

            if (!sawVertexElement) fail("property '" + line + "' before any element");

            if (tok.size() >= 2 && tok[1] == "list") {

                fail("list property in the vertex element is not supported ('" + line + "')");
            }
            if (tok.size() < 3) fail("malformed property line '" + line + "'");

            const int size = scalarSize(tok[1]);
            if (size == 0) fail("unknown property type '" + tok[1] + "'");

            properties.push_back({tok[2], tok[1], stride, size});
            stride += static_cast<size_t>(size);
            continue;
        }

        // Unknown header keyword: ignore rather than fail. PLY writers add
        // their own, and none of them change the vertex layout.
    }

    if (!sawVertexElement) fail("no 'vertex' element in header");

    // If a second element existed we broke out of the loop before consuming
    // end_header; skip forward to it so the binary body starts where we think.
    if (line.rfind("element", 0) == 0) {

        while (headerLine(stream, line)) {

            if (line == "end_header") break;
        }
        if (line != "end_header") fail("truncated header (no end_header)");
    }

    std::unordered_map<std::string, const Property*> byName;
    for (const auto& p : properties) byName[p.name] = &p;

    auto require = [&](const std::string& name) -> const Property& {
        const auto it = byName.find(name);
        if (it == byName.end()) fail("required property '" + name + "' missing");
        if (!isFloat32(it->second->type)) {

            fail("property '" + name + "' has type '" + it->second->type + "', expected float");
        }
        return *it->second;
    };

    const Property& px = require("x");
    const Property& py = require("y");
    const Property& pz = require("z");
    const Property& pOpacity = require("opacity");

    const Property* pScale[3];
    const Property* pRot[4];
    const Property* pDc[3];
    for (int i = 0; i < 3; ++i) pScale[i] = &require("scale_" + std::to_string(i));
    for (int i = 0; i < 4; ++i) pRot[i] = &require("rot_" + std::to_string(i));
    for (int i = 0; i < 3; ++i) pDc[i] = &require("f_dc_" + std::to_string(i));

    // f_rest_N, in declared numeric order. Gaps (f_rest_0 and f_rest_2 but no
    // f_rest_1) would mean the degree inference is meaningless, so require the
    // set to be contiguous from 0.
    std::vector<const Property*> rest;
    for (size_t i = 0;; ++i) {

        const auto it = byName.find("f_rest_" + std::to_string(i));
        if (it == byName.end()) break;
        if (!isFloat32(it->second->type)) {

            fail("property 'f_rest_" + std::to_string(i) + "' has type '" +
                 it->second->type + "', expected float");
        }
        rest.push_back(it->second);
    }

    {
        size_t declared = 0;
        for (const auto& p : properties) {

            if (p.name.rfind("f_rest_", 0) == 0) ++declared;
        }
        if (declared != rest.size()) {

            fail("f_rest_* indices are not contiguous from 0 (" + std::to_string(declared) +
                 " declared, " + std::to_string(rest.size()) + " usable)");
        }
    }

    const int degree = degreeFromRestCount(rest.size());
    const int restPerChannel = restCoeffsForDegree(degree);

    // Everything not consumed above, in declaration order. `normal` (nx/ny/nz)
    // is tolerated and dropped: 3DGS writes it as zeros and nothing reads it.
    std::vector<const Property*> extras;
    for (const auto& p : properties) {

        const auto& n = p.name;
        const bool consumed =
                n == "x" || n == "y" || n == "z" ||
                n == "opacity" ||
                n == "nx" || n == "ny" || n == "nz" ||
                n.rfind("scale_", 0) == 0 ||
                n.rfind("rot_", 0) == 0 ||
                n.rfind("f_dc_", 0) == 0 ||
                n.rfind("f_rest_", 0) == 0;

        if (!consumed) {

            if (!isFloat32(p.type)) {

                fail("extra property '" + n + "' has type '" + p.type +
                     "', expected float (non-float extras are not supported)");
            }
            extras.push_back(&p);
        }
    }

    SplatData data;
    data.resize(vertexCount, degree);
    for (const auto* p : extras) data.extras[p->name].assign(vertexCount, 0.f);

    // Read the body in blocks rather than per-splat: a 1M-splat, degree-3 file
    // is ~250 MB and one istream::read per vertex is measurably slower.
    constexpr size_t BLOCK_SPLATS = 4096;
    std::vector<unsigned char> block(stride * BLOCK_SPLATS);

    for (size_t first = 0; first < vertexCount; first += BLOCK_SPLATS) {

        const size_t n = std::min(BLOCK_SPLATS, vertexCount - first);

        stream.read(reinterpret_cast<char*>(block.data()),
                    static_cast<std::streamsize>(stride * n));
        if (static_cast<size_t>(stream.gcount()) != stride * n) {

            fail("truncated body: wanted " + std::to_string(vertexCount) +
                 " vertices, ran out after " + std::to_string(first + static_cast<size_t>(stream.gcount()) / stride));
        }

        for (size_t k = 0; k < n; ++k) {

            const unsigned char* v = block.data() + k * stride;
            const size_t i = first + k;

            data.means[i].set(readFloat(v, px.offset), readFloat(v, py.offset), readFloat(v, pz.offset));

            // Gotcha 2a: file scale is log-scale.
            data.scales[i].set(
                    std::exp(readFloat(v, pScale[0]->offset)),
                    std::exp(readFloat(v, pScale[1]->offset)),
                    std::exp(readFloat(v, pScale[2]->offset)));

            // Gotcha 2b: rot_0 is w. threepp's Quaternion is (x, y, z, w).
            data.rotations[i].set(
                    readFloat(v, pRot[1]->offset),
                    readFloat(v, pRot[2]->offset),
                    readFloat(v, pRot[3]->offset),
                    readFloat(v, pRot[0]->offset));

            // Gotcha 2c: file opacity is a logit.
            data.opacities[i] = splats::sigmoid(readFloat(v, pOpacity.offset));

            float* c = data.shAt(i);
            for (int ch = 0; ch < 3; ++ch) c[ch] = readFloat(v, pDc[ch]->offset);

            // Gotcha 1: f_rest is channel-major. Disk index is
            // channel * restPerChannel + r; memory index is (1 + r) * 3 + channel.
            for (int ch = 0; ch < 3; ++ch) {

                for (int r = 0; r < restPerChannel; ++r) {

                    c[(1 + r) * 3 + ch] = readFloat(v, rest[ch * restPerChannel + r]->offset);
                }
            }

            for (const auto* p : extras) {

                data.extras[p->name][i] = readFloat(v, p->offset);
            }
        }
    }

    data.normalizeRotations();

    std::string why;
    if (!data.validate(&why)) fail("internal consistency check failed: " + why);

    return data;
}


// ---------------------------------------------------------------------------
// Colour-only point clouds
// ---------------------------------------------------------------------------

namespace {

    enum class PlyFormat { BinaryLittleEndian, BinaryBigEndian, Ascii };

    struct PlyProperty {
        std::string name;
        const ScalarType* type = nullptr;     // scalar, or the list's item type
        const ScalarType* listCount = nullptr;// non-null for a list property
        size_t offset = 0;                    // within a list-free element
    };

    struct PlyElement {
        std::string name;
        size_t count = 0;
        std::vector<PlyProperty> properties;
        size_t stride = 0;// meaningful only when !hasList
        bool hasList = false;
    };

    struct PlyHeader {
        PlyFormat format = PlyFormat::BinaryLittleEndian;
        std::vector<PlyElement> elements;
    };

    // The full element table, unlike parsePly's vertex-first read: a point
    // cloud's vertices may follow a `camera` or `extra` element, and a mesh's
    // faces follow them. Returns false with `why` set rather than throwing so
    // isPointCloudPly can share it.
    bool parsePlyHeader(std::istream& stream, PlyHeader& header, std::string& why) {

        std::string line;
        if (!headerLine(stream, line) || line != "ply") {
            why = "not a PLY file (missing 'ply' magic)";
            return false;
        }

        constexpr int MAX_HEADER_LINES = 1000;
        bool sawFormat = false;
        bool sawEnd = false;
        for (int i = 0; i < MAX_HEADER_LINES && headerLine(stream, line); ++i) {

            const auto tok = tokens(line);
            if (tok.empty() || tok[0] == "comment" || tok[0] == "obj_info") continue;

            if (tok[0] == "format") {
                if (tok.size() < 2) {
                    why = "malformed format line '" + line + "'";
                    return false;
                }
                if (tok[1] == "binary_little_endian") header.format = PlyFormat::BinaryLittleEndian;
                else if (tok[1] == "binary_big_endian") header.format = PlyFormat::BinaryBigEndian;
                else if (tok[1] == "ascii") header.format = PlyFormat::Ascii;
                else {
                    why = "unsupported format '" + tok[1] + "'";
                    return false;
                }
                sawFormat = true;
                continue;
            }

            if (tok[0] == "end_header") {
                sawEnd = true;
                break;
            }

            if (tok[0] == "element") {
                if (tok.size() < 3) {
                    why = "malformed element line '" + line + "'";
                    return false;
                }
                PlyElement e;
                e.name = tok[1];
                e.count = static_cast<size_t>(std::strtoull(tok[2].c_str(), nullptr, 10));
                header.elements.push_back(std::move(e));
                continue;
            }

            if (tok[0] == "property") {
                if (header.elements.empty()) {
                    why = "property '" + line + "' before any element";
                    return false;
                }
                auto& e = header.elements.back();
                PlyProperty p;
                if (tok.size() >= 5 && tok[1] == "list") {
                    p.listCount = scalarType(tok[2]);
                    p.type = scalarType(tok[3]);
                    p.name = tok[4];
                    if (!p.listCount || !p.type) {
                        why = "unknown list property type in '" + line + "'";
                        return false;
                    }
                    e.hasList = true;
                } else {
                    if (tok.size() < 3) {
                        why = "malformed property line '" + line + "'";
                        return false;
                    }
                    p.type = scalarType(tok[1]);
                    p.name = tok[2];
                    if (!p.type) {
                        why = "unknown property type '" + tok[1] + "'";
                        return false;
                    }
                    p.offset = e.stride;
                    e.stride += static_cast<size_t>(p.type->size);
                }
                e.properties.push_back(std::move(p));
                continue;
            }
            // Unknown keyword: ignored, like parsePly.
        }

        if (!sawEnd) {
            why = "truncated header (no end_header)";
            return false;
        }
        if (!sawFormat) {
            why = "no format line";
            return false;
        }
        return true;
    }

    double readScalar(const unsigned char* p, const ScalarType& t, bool bigEndian) {

        unsigned char b[8];
        std::memcpy(b, p, static_cast<size_t>(t.size));
        if (bigEndian) std::reverse(b, b + t.size);

        switch (t.kind) {
            case ScalarKind::Float: {
                if (t.size == 4) {
                    float v;
                    std::memcpy(&v, b, 4);
                    return v;
                }
                double v;
                std::memcpy(&v, b, 8);
                return v;
            }
            case ScalarKind::Unsigned: {
                uint64_t v = 0;
                for (int i = t.size - 1; i >= 0; --i) v = (v << 8) | b[i];
                return static_cast<double>(v);
            }
            case ScalarKind::Signed: {
                uint64_t v = 0;
                for (int i = t.size - 1; i >= 0; --i) v = (v << 8) | b[i];
                const int shift = 64 - 8 * t.size;
                return static_cast<double>(static_cast<int64_t>(v << shift) >> shift);
            }
        }
        return 0.0;
    }

    // An integer colour channel divided by its type's range; a float taken
    // as is (the [0, 255] float case is handled after the whole cloud is
    // read, when its maximum is known).
    double normalisedChannel(double v, const ScalarType& t) {

        switch (t.kind) {
            case ScalarKind::Unsigned:
                return v / static_cast<double>((uint64_t{1} << (8 * t.size)) - 1);
            case ScalarKind::Signed:
                return std::max(0.0, v) / static_cast<double>((uint64_t{1} << (8 * t.size - 1)) - 1);
            case ScalarKind::Float:
                return v;
        }
        return v;
    }

    std::string lowered(std::string s) {

        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    // Rotation taking +Z onto `n`, as (x, y, z, w). Identity for a zero
    // normal; a half-turn about X when n is -Z.
    SplatQuat quatFromNormal(Vector3 n) {

        const float len = n.length();
        if (!(len > 0.f) || !std::isfinite(len)) return {};
        n.multiplyScalar(1.f / len);
        const float w = 1.f + n.z;
        if (w < 1e-6f) return {1.f, 0.f, 0.f, 0.f};
        SplatQuat q{-n.y, n.x, 0.f, w};
        return q.normalize();
    }

    [[noreturn]] void failPc(const std::string& msg) {

        throw std::runtime_error("SplatLoader (point cloud): " + msg);
    }

}// namespace


bool SplatLoader::isPointCloudPly(const std::filesystem::path& path) {

    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    return isPointCloudPly(in);
}

bool SplatLoader::isPointCloudPly(std::istream& stream) {

    PlyHeader header;
    std::string why;
    if (!parsePlyHeader(stream, header, why)) return false;

    bool hasVertex = false;
    for (const auto& e : header.elements) {

        if (e.name == "face" && e.count > 0) return false;
        if (e.name != "vertex") continue;

        bool x = false, y = false, z = false;
        for (const auto& p : e.properties) {
            if (p.name == "f_dc_0") return false;
            if (p.name == "x") x = true;
            if (p.name == "y") y = true;
            if (p.name == "z") z = true;
        }
        hasVertex = x && y && z;
    }
    return hasVertex;
}

SplatData SplatLoader::loadPointCloudPly(const std::filesystem::path& path) {

    return loadPointCloudPly(path, PointCloudOptions{}, nullptr);
}

SplatData SplatLoader::loadPointCloudPly(const std::filesystem::path& path,
                                         const PointCloudOptions& options, PointCloudInfo* info) {

    std::ifstream in(path, std::ios::binary);
    if (!in) failPc("cannot open '" + path.string() + "'");

    try {

        return parsePointCloudPly(in, options, info);

    } catch (const std::runtime_error& e) {

        throw std::runtime_error(std::string(e.what()) + " (in '" + path.string() + "')");
    }
}

SplatData SplatLoader::parsePointCloudPly(std::istream& stream) {

    return parsePointCloudPly(stream, PointCloudOptions{}, nullptr);
}

SplatData SplatLoader::parsePointCloudPly(std::istream& stream, const PointCloudOptions& options,
                                          PointCloudInfo* info) {

    PlyHeader header;
    std::string why;
    if (!parsePlyHeader(stream, header, why)) failPc(why);

    const PlyElement* vertex = nullptr;
    for (const auto& e : header.elements)
        if (e.name == "vertex") {
            vertex = &e;
            break;
        }
    if (!vertex) failPc("no 'vertex' element in header");
    if (vertex->hasList) failPc("list property in the vertex element is not supported");

    // ── which property is which ─────────────────────────────────────────
    const PlyProperty* px = nullptr;
    const PlyProperty* py = nullptr;
    const PlyProperty* pz = nullptr;
    const PlyProperty* pRgb[3] = {nullptr, nullptr, nullptr};
    const PlyProperty* pNormal[3] = {nullptr, nullptr, nullptr};
    const PlyProperty* pIntensity = nullptr;
    std::vector<const PlyProperty*> extras;

    for (const auto& p : vertex->properties) {

        const std::string n = lowered(p.name);
        if (n == "x") px = &p;
        else if (n == "y") py = &p;
        else if (n == "z") pz = &p;
        else if (n == "red" || n == "r" || n == "diffuse_red") pRgb[0] = &p;
        else if (n == "green" || n == "g" || n == "diffuse_green") pRgb[1] = &p;
        else if (n == "blue" || n == "b" || n == "diffuse_blue") pRgb[2] = &p;
        else if (n == "nx" || n == "normal_x") pNormal[0] = &p;
        else if (n == "ny" || n == "normal_y") pNormal[1] = &p;
        else if (n == "nz" || n == "normal_z") pNormal[2] = &p;
        else if (n == "intensity" || n == "scalar_intensity") pIntensity = &p;
        else if (n == "alpha" || n == "a" || n == "diffuse_alpha") continue;
        else extras.push_back(&p);
    }
    if (!px || !py || !pz) failPc("vertex element has no x, y, z");

    const bool hasColor = pRgb[0] && pRgb[1] && pRgb[2];
    const bool hasNormals = pNormal[0] && pNormal[1] && pNormal[2];

    const size_t count = vertex->count;
    SplatData data;
    data.resize(count, 0);
    for (const auto* p : extras) data.extras[p->name].assign(count, 0.f);

    std::vector<float> rgb;// interleaved, [0, 1] unless a float file used [0, 255]
    if (hasColor) rgb.assign(count * 3, 0.f);
    std::vector<Vector3> normals;
    if (hasNormals) normals.assign(count, Vector3{});
    std::vector<float> intensity;
    if (pIntensity && !hasColor) intensity.assign(count, 0.f);

    // ── one vertex, from its property values in declaration order ───────
    const size_t propCount = vertex->properties.size();
    const auto indexOf = [&](const PlyProperty* p) {
        return static_cast<size_t>(p - vertex->properties.data());
    };
    const size_t ix = indexOf(px), iy = indexOf(py), iz = indexOf(pz);

    const auto storeVertex = [&](size_t i, const std::vector<double>& v) {
        data.means[i].set(static_cast<float>(v[ix]), static_cast<float>(v[iy]),
                          static_cast<float>(v[iz]));
        if (hasColor) {
            for (int ch = 0; ch < 3; ++ch)
                rgb[i * 3 + static_cast<size_t>(ch)] = static_cast<float>(
                        normalisedChannel(v[indexOf(pRgb[ch])], *pRgb[ch]->type));
        }
        if (hasNormals) {
            normals[i].set(static_cast<float>(v[indexOf(pNormal[0])]),
                           static_cast<float>(v[indexOf(pNormal[1])]),
                           static_cast<float>(v[indexOf(pNormal[2])]));
        }
        if (!intensity.empty()) intensity[i] = static_cast<float>(v[indexOf(pIntensity)]);
        for (const auto* p : extras) data.extras[p->name][i] = static_cast<float>(v[indexOf(p)]);
    };

    // ── body ────────────────────────────────────────────────────────────
    std::vector<double> values(propCount);

    if (header.format == PlyFormat::Ascii) {

        std::string line;
        const auto nextLine = [&](std::vector<std::string>& tok) {
            while (headerLine(stream, line)) {
                tok = tokens(line);
                if (!tok.empty()) return true;
            }
            return false;
        };

        for (const auto& e : header.elements) {

            std::vector<std::string> tok;
            for (size_t i = 0; i < e.count; ++i) {

                if (!nextLine(tok)) {
                    failPc("truncated body: element '" + e.name + "' wanted " +
                           std::to_string(e.count) + " lines, ran out after " + std::to_string(i));
                }
                if (&e != vertex) continue;// a skipped element is one line per instance
                if (tok.size() < propCount) {
                    failPc("vertex " + std::to_string(i) + " has " + std::to_string(tok.size()) +
                           " values, header declares " + std::to_string(propCount));
                }
                for (size_t k = 0; k < propCount; ++k) values[k] = std::strtod(tok[k].c_str(), nullptr);
                storeVertex(i, values);
            }
        }

    } else {

        const bool bigEndian = header.format == PlyFormat::BinaryBigEndian;

        for (const auto& e : header.elements) {

            if (&e != vertex) {

                // Skip. A list-free element is one seek; a list element has
                // to be walked, since each instance's size is in its data.
                if (!e.hasList) {
                    stream.seekg(static_cast<std::streamoff>(e.stride * e.count), std::ios::cur);
                    if (!stream) failPc("truncated body while skipping element '" + e.name + "'");
                    continue;
                }
                unsigned char buf[8];
                for (size_t i = 0; i < e.count; ++i) {
                    for (const auto& p : e.properties) {
                        if (!p.listCount) {
                            stream.seekg(p.type->size, std::ios::cur);
                            continue;
                        }
                        stream.read(reinterpret_cast<char*>(buf), p.listCount->size);
                        if (!stream) failPc("truncated body in element '" + e.name + "'");
                        const auto n = static_cast<std::streamoff>(readScalar(buf, *p.listCount, bigEndian));
                        stream.seekg(n * p.type->size, std::ios::cur);
                    }
                }
                if (!stream) failPc("truncated body while skipping element '" + e.name + "'");
                continue;
            }

            constexpr size_t BLOCK = 4096;
            const size_t stride = e.stride;
            std::vector<unsigned char> block(stride * BLOCK);
            for (size_t first = 0; first < count; first += BLOCK) {

                const size_t n = std::min(BLOCK, count - first);
                stream.read(reinterpret_cast<char*>(block.data()),
                            static_cast<std::streamsize>(stride * n));
                if (static_cast<size_t>(stream.gcount()) != stride * n) {
                    failPc("truncated body: wanted " + std::to_string(count) +
                           " vertices, ran out after " +
                           std::to_string(first + static_cast<size_t>(stream.gcount()) / stride));
                }
                for (size_t k = 0; k < n; ++k) {
                    const unsigned char* v = block.data() + k * stride;
                    for (size_t q = 0; q < propCount; ++q) {
                        const auto& p = e.properties[q];
                        values[q] = readScalar(v + p.offset, *p.type, bigEndian);
                    }
                    storeVertex(first + k, values);
                }
            }
        }
    }

    // ── colour ──────────────────────────────────────────────────────────
    if (hasColor) {

        // A float colour above 1 anywhere means the file used [0, 255].
        const bool floatChannels = pRgb[0]->type->kind == ScalarKind::Float ||
                                   pRgb[1]->type->kind == ScalarKind::Float ||
                                   pRgb[2]->type->kind == ScalarKind::Float;
        if (floatChannels) {
            float peak = 0.f;
            for (float c : rgb) peak = std::max(peak, c);
            if (peak > 1.0001f) {
                for (float& c : rgb) c /= 255.f;
            }
        }
        for (size_t i = 0; i < count; ++i) {
            data.setDcColor(i, Vector3{std::clamp(rgb[i * 3], 0.f, 1.f),
                                       std::clamp(rgb[i * 3 + 1], 0.f, 1.f),
                                       std::clamp(rgb[i * 3 + 2], 0.f, 1.f)});
        }
    } else if (!intensity.empty()) {

        // Grey from intensity, scaled by the cloud's own maximum: scanners
        // write raw returns in whatever range the instrument has.
        float peak = 0.f;
        for (float v : intensity) if (std::isfinite(v)) peak = std::max(peak, v);
        const float scale = peak > 0.f ? 1.f / peak : 1.f;
        for (size_t i = 0; i < count; ++i) {
            const float g = std::clamp(intensity[i] * scale, 0.f, 1.f);
            data.setDcColor(i, Vector3{g, g, g});
        }
    } else {

        for (size_t i = 0; i < count; ++i) data.setDcColor(i, Vector3{1.f, 1.f, 1.f});
    }

    // ── size and orientation ────────────────────────────────────────────
    float spacing = 0.f;
    float sigma = options.sigma;
    if (!(sigma > 0.f)) {
        spacing = splats::medianNeighbourSpacing(data.means);
        sigma = spacing > 0.f ? spacing * options.sigmaPerSpacing : 0.01f;
    }

    const bool orient = hasNormals && options.useNormals;
    const float thin = std::clamp(options.normalThickness, 0.f, 1.f);
    const float opacity = std::clamp(options.opacity, 0.f, 1.f);
    for (size_t i = 0; i < count; ++i) {

        data.opacities[i] = opacity;
        if (orient) {
            data.rotations[i] = quatFromNormal(normals[i]);
            // A zero normal got the identity above; it stays isotropic too.
            const bool flat = normals[i].lengthSq() > 0.f;
            data.scales[i].set(sigma, sigma, flat ? sigma * thin : sigma);
        } else {
            data.scales[i].set(sigma, sigma, sigma);
        }
    }

    if (!data.validate(&why)) failPc("internal consistency check failed: " + why);

    if (info) {
        info->count = count;
        info->spacing = spacing;
        info->sigma = sigma;
        info->hadColor = hasColor;
        info->hadNormals = hasNormals;
        info->hadIntensity = !intensity.empty();
    }

    return data;
}


// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

void SplatLoader::writePly(const SplatData& data, std::ostream& out) {

    const int coeffs = data.coeffCount();
    const int restPerChannel = coeffs - 1;
    const size_t n = data.count();

    out << "ply\nformat binary_little_endian 1.0\n"
        << "comment written by threepp SplatLoader\n"
        << "element vertex " << n << "\n"
        << "property float x\nproperty float y\nproperty float z\n";
    for (int i = 0; i < 3; ++i) out << "property float f_dc_" << i << "\n";
    for (int i = 0; i < 3 * restPerChannel; ++i) out << "property float f_rest_" << i << "\n";
    out << "property float opacity\n";
    for (int i = 0; i < 3; ++i) out << "property float scale_" << i << "\n";
    for (int i = 0; i < 4; ++i) out << "property float rot_" << i << "\n";
    for (const auto& [name, values] : data.extras) out << "property float " << name << "\n";
    out << "end_header\n";

    // One splat per row, assembled in a reusable buffer: a 1M-splat degree-3
    // cloud is 250 MB, and one stream write per float is measurably slower.
    const size_t floatsPerSplat = 3 + 3 + static_cast<size_t>(3 * restPerChannel) + 1 + 3 + 4 + data.extras.size();
    std::vector<float> row(floatsPerSplat);
    const auto logScale = [](float s) { return std::log(std::max(s, 1e-9f)); };

    for (size_t i = 0; i < n; ++i) {

        size_t k = 0;
        row[k++] = data.means[i].x;
        row[k++] = data.means[i].y;
        row[k++] = data.means[i].z;

        const float* c = data.shAt(i);
        for (int ch = 0; ch < 3; ++ch) row[k++] = c[ch];
        for (int ch = 0; ch < 3; ++ch)
            for (int r = 0; r < restPerChannel; ++r) row[k++] = c[(1 + r) * 3 + ch];

        row[k++] = splats::logit(std::clamp(data.opacities[i], 1e-6f, 1.f - 1e-6f));

        row[k++] = logScale(data.scales[i].x);
        row[k++] = logScale(data.scales[i].y);
        row[k++] = logScale(data.scales[i].z);

        const auto& q = data.rotations[i];
        row[k++] = q.w;
        row[k++] = q.x;
        row[k++] = q.y;
        row[k++] = q.z;

        for (const auto& [name, values] : data.extras) row[k++] = i < values.size() ? values[i] : 0.f;

        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
}

void SplatLoader::writePly(const SplatData& data, const std::filesystem::path& path) {

    std::ofstream out(path, std::ios::binary);
    if (!out) fail("cannot open '" + path.string() + "' for writing");
    writePly(data, out);
    if (!out) fail("write failed for '" + path.string() + "'");
}
