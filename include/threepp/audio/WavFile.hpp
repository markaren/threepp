#ifndef THREEPP_WAVFILE_HPP
#define THREEPP_WAVFILE_HPP

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace threepp::audio {

    // Write a mono 16-bit PCM WAV file from normalised float samples ∈ [-1, 1].
    inline void writeWav(const std::filesystem::path& path,
                         const std::vector<float>& samples,
                         int sampleRate = 44100) {
        std::ofstream f(path, std::ios::binary);
        auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
        auto u16 = [&](std::uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); };
        const auto dataBytes = static_cast<std::uint32_t>(samples.size()) * 2u;
        f.write("RIFF", 4);
        u32(36 + dataBytes);
        f.write("WAVE", 4);
        f.write("fmt ", 4);
        u32(16);
        u16(1);// PCM
        u16(1);// mono
        u32(static_cast<std::uint32_t>(sampleRate));
        u32(static_cast<std::uint32_t>(sampleRate) * 2u);
        u16(2);
        u16(16);
        f.write("data", 4);
        u32(dataBytes);
        for (float x : samples) {
            const auto q = static_cast<std::int16_t>(
                    std::lround(std::clamp(x, -1.f, 1.f) * 32767.f));
            f.write(reinterpret_cast<const char*>(&q), 2);
        }
    }

}// namespace threepp::audio

#endif//THREEPP_WAVFILE_HPP
