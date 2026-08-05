// Test-only writer for the INRIA 3D Gaussian Splatting .ply layout.
//
// It exists so the loader can be pinned by round trip rather than by a
// checked-in binary blob: generator -> write -> load -> compare. That only
// proves anything because the writer applies the file's conventions in the
// opposite direction to the loader — channel-major f_rest, log scale, logit
// opacity, w-first quaternions — so a sign or ordering error in one of them
// cannot cancel out against the same error in the other.
//
// Not public API. A production exporter would need to decide what to do about
// splats the format cannot represent (see the note on scale below).

#ifndef THREEPP_SPLAT_PLY_WRITER_HPP
#define THREEPP_SPLAT_PLY_WRITER_HPP

#include "threepp/splats/SplatData.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace splattest {

    struct PlyWriteOptions {

        // Emit the nx/ny/nz the reference implementation writes (as zeros).
        // The loader must tolerate and ignore them.
        bool withNormals = false;

        // Scale every quaternion by this before writing, to reproduce the
        // unnormalised rotations real files carry.
        float quatScale = 1.f;
    };

    inline void putFloat(std::string& out, float value) {

        // Little-endian, which every platform threepp targets is.
        char bytes[sizeof(float)];
        std::memcpy(bytes, &value, sizeof(float));
        out.append(bytes, sizeof(float));
    }

    // Returns the whole file as a buffer. Feed it to std::istringstream for
    // SplatLoader::parsePly, or write it out for SplatLoader::loadPly.
    //
    // NOTE: scale is stored as log(scale), so a splat with an exactly zero axis
    // cannot be represented — it would be written as -inf. That is a property of
    // the format, not of this writer; round-trip tests use non-degenerate clouds.
    inline std::string writeSplatPly(const threepp::SplatData& data, const PlyWriteOptions& options = {}) {

        using namespace threepp;

        const int coeffs = data.coeffCount();
        const int restPerChannel = coeffs - 1;

        std::ostringstream header;
        header << "ply\n";
        header << "format binary_little_endian 1.0\n";
        header << "element vertex " << data.count() << "\n";
        header << "property float x\nproperty float y\nproperty float z\n";
        if (options.withNormals) {

            header << "property float nx\nproperty float ny\nproperty float nz\n";
        }
        for (int i = 0; i < 3; ++i) header << "property float f_dc_" << i << "\n";
        for (int i = 0; i < 3 * restPerChannel; ++i) header << "property float f_rest_" << i << "\n";
        header << "property float opacity\n";
        for (int i = 0; i < 3; ++i) header << "property float scale_" << i << "\n";
        for (int i = 0; i < 4; ++i) header << "property float rot_" << i << "\n";
        for (const auto& [name, values] : data.extras) header << "property float " << name << "\n";
        header << "end_header\n";

        std::string out = header.str();

        for (size_t i = 0; i < data.count(); ++i) {

            putFloat(out, data.means[i].x);
            putFloat(out, data.means[i].y);
            putFloat(out, data.means[i].z);

            if (options.withNormals) {

                putFloat(out, 0.f);
                putFloat(out, 0.f);
                putFloat(out, 0.f);
            }

            const float* c = data.shAt(i);
            for (int ch = 0; ch < 3; ++ch) putFloat(out, c[ch]);

            // Channel-major: every higher-order coefficient of red, then green,
            // then blue. The loader has to undo exactly this.
            for (int ch = 0; ch < 3; ++ch) {

                for (int r = 0; r < restPerChannel; ++r) putFloat(out, c[(1 + r) * 3 + ch]);
            }

            const float opacity = std::clamp(data.opacities[i], 1e-6f, 1.f - 1e-6f);
            putFloat(out, splats::logit(opacity));

            putFloat(out, std::log(data.scales[i].x));
            putFloat(out, std::log(data.scales[i].y));
            putFloat(out, std::log(data.scales[i].z));

            // W first.
            const auto& q = data.rotations[i];
            putFloat(out, q.w * options.quatScale);
            putFloat(out, q.x * options.quatScale);
            putFloat(out, q.y * options.quatScale);
            putFloat(out, q.z * options.quatScale);

            for (const auto& [name, values] : data.extras) putFloat(out, values[i]);
        }

        return out;
    }

    inline void writeSplatPlyFile(const threepp::SplatData& data,
                                  const std::filesystem::path& path,
                                  const PlyWriteOptions& options = {}) {

        const auto buffer = writeSplatPly(data, options);
        std::ofstream out(path, std::ios::binary);
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    }

}// namespace splattest

#endif//THREEPP_SPLAT_PLY_WRITER_HPP
