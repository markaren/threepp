
#include "threepp/loaders/SplatLoader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace threepp;

namespace {

    // PLY scalar types. Only float is *consumed*, but every type has to be
    // recognised so a property we skip still contributes the right number of
    // bytes to the stride.
    struct ScalarType {
        const char* name;
        int size;
    };

    constexpr ScalarType SCALAR_TYPES[] = {
            {"char", 1}, {"int8", 1}, {"uchar", 1}, {"uint8", 1},
            {"short", 2}, {"int16", 2}, {"ushort", 2}, {"uint16", 2},
            {"int", 4}, {"int32", 4}, {"uint", 4}, {"uint32", 4},
            {"float", 4}, {"float32", 4},
            {"double", 8}, {"float64", 8}};

    int scalarSize(const std::string& type) {

        for (const auto& t : SCALAR_TYPES) {

            if (type == t.name) return t.size;
        }
        return 0;
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
