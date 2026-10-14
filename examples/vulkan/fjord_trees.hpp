// FJORD TREES — the card trees of the Nørvasundet twin, read from threepp_data.
//
// The procedural TreeGenerator trees this demo used to grow were judged
// "pretty bad" next to the twin's, which are Poly Haven's scanned models
// (fir_tree_01, pine_tree_01, tree_small_02, all CC0) rebuilt in Blender as
// card trees: the trunk and the thick limbs kept as decimated wood, the
// foliage gathered into clusters, each cluster a card showing a rendered
// image of real needles or leaves. Two tiers per species, three variants each:
//
//   card   ~1 400 leaf triangles: the trees the camera walks among
//   lite   ~  440 leaf triangles: everything from a few hundred metres out
//
// threepp_data/models/fjord/trees/<species>.tree holds every mesh of a species
// quantised (16-bit positions/UVs, 8-bit normals and baked canopy occlusion);
// the atlases and bark are lossy WebP beside it. README.md there has the
// sources and the licence.

#ifndef THREEPP_FJORD_TREES_HPP
#define THREEPP_FJORD_TREES_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fjord {

    // Every mesh of one species, by its name in the file ("card0_leaf",
    // "lite2_trunk", "card1_wood1", ...).
    using KitMeshes = std::unordered_map<std::string, std::shared_ptr<threepp::BufferGeometry>>;

    namespace detail {

        struct Reader {
            const std::vector<char>& b;
            size_t at = 0;

            template<class T>
            T get() {
                if (at + sizeof(T) > b.size()) throw std::runtime_error("tree file truncated");
                T v;
                std::memcpy(&v, b.data() + at, sizeof(T));
                at += sizeof(T);
                return v;
            }
            template<class T>
            void array(std::vector<T>& out, size_t n) {
                if (at + n * sizeof(T) > b.size()) throw std::runtime_error("tree file truncated");
                out.resize(n);
                std::memcpy(out.data(), b.data() + at, n * sizeof(T));
                at += n * sizeof(T);
            }
        };

    }// namespace detail

    // Reads <species>.tree (format "TRE1", written by the packer named in
    // threepp_data/models/fjord/README.md). Positions in metres, y up, the
    // tree standing on the origin; UVs as the twin's kit had them.
    inline KitMeshes loadTreeFile(const std::filesystem::path& path) {
        using namespace threepp;
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path.string());
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        detail::Reader r{bytes};
        char magic[4];
        for (char& c : magic) c = r.get<char>();
        if (std::memcmp(magic, "TRE1", 4) != 0) throw std::runtime_error("not a TRE1 file: " + path.string());

        KitMeshes out;
        const auto meshCount = r.get<uint32_t>();
        for (uint32_t m = 0; m < meshCount; ++m) {
            const auto nameLen = r.get<uint8_t>();
            std::string name(nameLen, '\0');
            for (auto& c : name) c = r.get<char>();
            const auto nv = r.get<uint32_t>();
            const auto nt = r.get<uint32_t>();
            const auto flags = r.get<uint8_t>();
            float pmin[3], pmax[3], uvmin[2], uvmax[2];
            for (float& v : pmin) v = r.get<float>();
            for (float& v : pmax) v = r.get<float>();
            for (float& v : uvmin) v = r.get<float>();
            for (float& v : uvmax) v = r.get<float>();

            std::vector<uint16_t> qp, qu;
            std::vector<int8_t> qn;
            std::vector<uint8_t> qc;
            r.array(qp, size_t(nv) * 3);
            r.array(qn, size_t(nv) * 4);
            r.array(qu, size_t(nv) * 2);
            if (flags & 1u) r.array(qc, size_t(nv) * 4);

            std::vector<unsigned int> index(size_t(nt) * 3);
            if (flags & 2u) {
                std::vector<uint32_t> i32;
                r.array(i32, index.size());
                for (size_t i = 0; i < index.size(); ++i) index[i] = i32[i];
            } else {
                std::vector<uint16_t> i16;
                r.array(i16, index.size());
                for (size_t i = 0; i < index.size(); ++i) index[i] = i16[i];
            }

            std::vector<float> pos(size_t(nv) * 3), nrm(size_t(nv) * 3), uv(size_t(nv) * 2);
            for (size_t i = 0; i < nv; ++i) {
                for (int k = 0; k < 3; ++k) {
                    pos[i * 3 + k] = pmin[k] + (pmax[k] - pmin[k]) * (float(qp[i * 3 + k]) / 65535.f);
                }
                float n[3];
                for (int k = 0; k < 3; ++k) n[k] = std::max(-1.f, float(qn[i * 4 + k]) / 127.f);
                const float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                for (int k = 0; k < 3; ++k) nrm[i * 3 + k] = l > 1e-6f ? n[k] / l : (k == 1 ? 1.f : 0.f);
                for (int k = 0; k < 2; ++k) {
                    uv[i * 2 + k] = uvmin[k] + (uvmax[k] - uvmin[k]) * (float(qu[i * 2 + k]) / 65535.f);
                }
            }

            auto geo = BufferGeometry::create();
            geo->setIndex(index);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            if (flags & 1u) {
                std::vector<float> col(size_t(nv) * 3);
                for (size_t i = 0; i < nv; ++i)
                    for (int k = 0; k < 3; ++k) col[i * 3 + k] = float(qc[i * 4 + k]) / 255.f;
                geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
            }
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            out[name] = geo;
        }
        if (r.at != bytes.size()) {
            std::cerr << "[fjord] " << path.filename().string() << ": " << (bytes.size() - r.at)
                      << " trailing bytes" << std::endl;
        }
        return out;
    }

}// namespace fjord

#endif//THREEPP_FJORD_TREES_HPP
