
#include "threepp/loaders/ImageLoader.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "webp/decode.h"

#include <cstring>
#include <fstream>
#include <iostream>

using namespace threepp;

namespace {

    // WebP is a RIFF container: "RIFF" <uint32 size> "WEBP". stb has no RIFF-based
    // format, so sniffing this cannot steal a file from the stb path or vice versa.
    constexpr size_t kWebpMagicSize = 12;

    bool isWebp(const unsigned char* data, size_t size) {

        return size >= kWebpMagicSize &&
               std::memcmp(data, "RIFF", 4) == 0 &&
               std::memcmp(data + 8, "WEBP", 4) == 0;
    }

    bool isWebp(const std::vector<unsigned char>& data) {

        return isWebp(data.data(), data.size());
    }

    std::optional<Image> loadWebp(const unsigned char* data, size_t size, int channels, bool flipY) {

        WebPBitstreamFeatures features{};
        if (WebPGetFeatures(data, size, &features) != VP8_STATUS_OK) {
            return std::nullopt;
        }

        if (features.has_animation) {
            // Animated WebP lives in the demux module, which is not vendored. Decoding
            // the still image out of an animation container would silently hand back one
            // arbitrary frame, so say what happened instead.
            std::cerr << "[ImageLoader] Animated WebP is not supported\n";
            return std::nullopt;
        }

        WEBP_CSP_MODE colorspace;
        if (channels == 4) {
            colorspace = MODE_RGBA;
        } else if (channels == 3) {
            colorspace = MODE_RGB;
        } else {
            // 1 and 2 channels are legal for the stb path but used by no caller, and
            // libwebp has no greyscale output mode to map them onto.
            return std::nullopt;
        }

        WebPDecoderConfig config{};
        if (!WebPInitDecoderConfig(&config)) {
            return std::nullopt;
        }

        const auto width = static_cast<size_t>(features.width);
        const auto height = static_cast<size_t>(features.height);
        const size_t stride = width * static_cast<size_t>(channels);

        std::vector<unsigned char> pixels(stride * height);

        // libwebp flips while it writes scanlines, so flipY costs nothing here — unlike
        // the stb path, which mirrors rows afterwards.
        config.options.flip = flipY ? 1 : 0;
        config.output.colorspace = colorspace;
        config.output.is_external_memory = 1;
        config.output.u.RGBA.rgba = pixels.data();
        config.output.u.RGBA.stride = static_cast<int>(stride);
        config.output.u.RGBA.size = pixels.size();

        const VP8StatusCode status = WebPDecode(data, size, &config);
        // Nothing to release while is_external_memory is set, but the buffer must still be
        // reset: WebPDecode can have attached scratch state to it on the way through.
        WebPFreeDecBuffer(&config.output);

        if (status != VP8_STATUS_OK) {
            return std::nullopt;
        }

        return Image{std::move(pixels),
                     static_cast<unsigned int>(features.width),
                     static_cast<unsigned int>(features.height)};
    }

    struct ImageStruct {

        int width{};
        int height{};
        int channels;
        unsigned char* pixels = nullptr;

        ImageStruct(const std::vector<unsigned char>& data, int channels): channels(channels) {
            pixels = stbi_load_from_memory(data.data(), static_cast<int>(data.size()),
                                           &width, &height, nullptr, channels);
        }

        ImageStruct(const std::filesystem::path& imagePath, int channels): channels(channels) {
            pixels = stbi_load(imagePath.string().c_str(), &width, &height, nullptr, channels);
        }

        [[nodiscard]] bool ok() const noexcept { return pixels != nullptr; }

        [[nodiscard]] std::vector<unsigned char> get(bool flipY) const {
            const size_t rowBytes = static_cast<size_t>(channels) * width;
            const size_t total = rowBytes * height;
            std::vector<unsigned char> result(total);
            if (flipY) {
                for (int y = 0; y < height; ++y) {
                    const int srcY = height - 1 - y;
                    std::memcpy(result.data() + static_cast<size_t>(y) * rowBytes,
                                pixels + static_cast<size_t>(srcY) * rowBytes,
                                rowBytes);
                }
            } else {
                std::memcpy(result.data(), pixels, total);
            }
            return result;
        }

        ~ImageStruct() {
            stbi_image_free(pixels);
        }
    };

}// namespace

std::optional<Image> ImageLoader::load(const std::filesystem::path& imagePath, int channels, bool flipY) {

    if (!std::filesystem::exists(imagePath)) {
        return std::nullopt;
    }

    // Sniff the magic rather than the extension, so a webp byte stream loads out of a file
    // named .png the same way SplatLoader trusts a PLY header over a file name.
    {
        std::ifstream in(imagePath, std::ios::binary);
        unsigned char magic[kWebpMagicSize]{};
        if (in.read(reinterpret_cast<char*>(magic), kWebpMagicSize) &&
            isWebp(magic, kWebpMagicSize)) {

            in.seekg(0, std::ios::end);
            const auto size = in.tellg();
            if (size < 0) return std::nullopt;
            in.seekg(0, std::ios::beg);

            std::vector<unsigned char> data(static_cast<size_t>(size));
            if (!in.read(reinterpret_cast<char*>(data.data()), size)) {
                return std::nullopt;
            }
            return loadWebp(data.data(), data.size(), channels, flipY);
        }
    }

    ImageStruct image{imagePath, channels};
    if (!image.ok()) return std::nullopt;

    return Image{
            image.get(flipY),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height)};
}

std::optional<Image> ImageLoader::load(const std::vector<unsigned char>& data, int channels, bool flipY) {

    if (isWebp(data)) {
        return loadWebp(data.data(), data.size(), channels, flipY);
    }

    ImageStruct image{data, channels};
    if (!image.ok()) return std::nullopt;

    return Image{
            image.get(flipY),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height)};
}
