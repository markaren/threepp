
#include "threepp/loaders/tiff/TiffDecode.hpp"

// Declarations only: the stb implementation is compiled in ImageLoader.cpp.
// Its zlib decoder serves TIFF's Deflate compression.
#include "stb_image.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

    constexpr uint16_t kTagWidth = 256;
    constexpr uint16_t kTagHeight = 257;
    constexpr uint16_t kTagBitsPerSample = 258;
    constexpr uint16_t kTagCompression = 259;
    constexpr uint16_t kTagPhotometric = 262;
    constexpr uint16_t kTagFillOrder = 266;
    constexpr uint16_t kTagStripOffsets = 273;
    constexpr uint16_t kTagSamplesPerPixel = 277;
    constexpr uint16_t kTagRowsPerStrip = 278;
    constexpr uint16_t kTagStripByteCounts = 279;
    constexpr uint16_t kTagPlanarConfig = 284;
    constexpr uint16_t kTagPredictor = 317;
    constexpr uint16_t kTagColorMap = 320;
    constexpr uint16_t kTagTileWidth = 322;
    constexpr uint16_t kTagTileLength = 323;
    constexpr uint16_t kTagTileOffsets = 324;
    constexpr uint16_t kTagTileByteCounts = 325;
    constexpr uint16_t kTagExtraSamples = 338;
    constexpr uint16_t kTagSampleFormat = 339;

    constexpr uint32_t kCompressionNone = 1;
    constexpr uint32_t kCompressionLzw = 5;
    constexpr uint32_t kCompressionDeflate = 8;
    constexpr uint32_t kCompressionDeflateOld = 32946;
    constexpr uint32_t kCompressionPackBits = 32773;

    std::nullopt_t fail(const std::string& why) {
        std::cerr << "[ImageLoader] TIFF: " << why << "\n";
        return std::nullopt;
    }

    struct Reader {
        const unsigned char* data;
        size_t size;
        bool little;

        [[nodiscard]] bool has(size_t offset, size_t n) const {
            return offset <= size && n <= size - offset;
        }

        [[nodiscard]] uint16_t u16(size_t offset) const {
            const unsigned char* p = data + offset;
            return little ? static_cast<uint16_t>(p[0] | (p[1] << 8))
                          : static_cast<uint16_t>((p[0] << 8) | p[1]);
        }

        [[nodiscard]] uint32_t u32(size_t offset) const {
            const unsigned char* p = data + offset;
            return little ? (uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24))
                          : ((uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]));
        }
    };

    // One IFD entry: its type, count and where its value bytes sit in the file
    // (inline in the entry when they fit in four bytes).
    struct Entry {
        uint16_t type;
        uint32_t count;
        size_t valueOffset;
    };

    size_t typeSize(uint16_t type) {
        switch (type) {
            case 1:// BYTE
            case 2:// ASCII
            case 6:// SBYTE
            case 7:// UNDEFINED
                return 1;
            case 3:// SHORT
            case 8:// SSHORT
                return 2;
            case 4: // LONG
            case 9: // SLONG
            case 11:// FLOAT
                return 4;
            case 5: // RATIONAL
            case 10:// SRATIONAL
            case 12:// DOUBLE
                return 8;
            default:
                return 0;
        }
    }

    // The unsigned integer values of a BYTE, SHORT or LONG entry.
    std::optional<std::vector<uint32_t>> values(const Reader& r, const Entry& e) {
        if (e.type != 1 && e.type != 3 && e.type != 4) return std::nullopt;
        const size_t ts = typeSize(e.type);
        if (!r.has(e.valueOffset, ts * size_t(e.count))) return std::nullopt;
        std::vector<uint32_t> out(e.count);
        for (size_t i = 0; i < out.size(); ++i) {
            const size_t at = e.valueOffset + i * ts;
            out[i] = e.type == 1 ? r.data[at] : e.type == 3 ? r.u16(at) : r.u32(at);
        }
        return out;
    }

    // PackBits: a signed count byte n, then n + 1 literal bytes (n >= 0) or one
    // byte repeated 1 - n times (n < 0; -128 is a no-op).
    void unpackBits(const unsigned char* in, size_t inLen, unsigned char* out, size_t outLen) {
        size_t i = 0, o = 0;
        while (i < inLen && o < outLen) {
            const auto n = static_cast<int8_t>(in[i++]);
            if (n >= 0) {
                const size_t count = std::min({size_t(n) + 1, inLen - i, outLen - o});
                std::memcpy(out + o, in + i, count);
                i += size_t(n) + 1;
                o += count;
            } else if (n != -128) {
                if (i >= inLen) break;
                const size_t count = std::min(size_t(1 - n), outLen - o);
                std::memset(out + o, in[i++], count);
                o += count;
            }
        }
    }

    // TIFF LZW: MSB-first codes of 9 to 12 bits, Clear = 256, EOI = 257, and
    // the code width grows one code early. A table entry is a run of the
    // output already written (the previous string plus the first byte of the
    // one after it, which follows it in the output), so no string is stored.
    bool unLzw(const unsigned char* in, size_t inLen, unsigned char* out, size_t outLen) {
        if (inLen >= 2 && in[0] == 0 && (in[1] & 1)) {
            // libtiff's test for the pre-6.0, LSB-first variant.
            fail("old-style (pre-TIFF 6.0) LZW is not supported");
            return false;
        }
        constexpr uint32_t kClear = 256, kEoi = 257, kFirst = 258, kMaxCodes = 4096;
        std::vector<uint32_t> pos(kMaxCodes), len(kMaxCodes);

        uint64_t bits = 0;
        int bitCount = 0;
        size_t i = 0, o = 0;
        int width = 9;
        uint32_t next = kFirst;
        bool havePrev = false;
        size_t prevPos = 0, prevLen = 0;

        while (o < outLen) {
            while (bitCount < width && i < inLen) {
                bits = (bits << 8) | in[i++];
                bitCount += 8;
            }
            if (bitCount < width) break;// out of data
            const auto code = static_cast<uint32_t>((bits >> (bitCount - width)) & ((1u << width) - 1));
            bitCount -= width;

            if (code == kEoi) break;
            if (code == kClear) {
                width = 9;
                next = kFirst;
                havePrev = false;
                continue;
            }

            const size_t curPos = o;
            size_t curLen;
            if (code < 256) {
                out[o++] = static_cast<unsigned char>(code);
                curLen = 1;
            } else if (code < next && code >= kFirst) {
                curLen = len[code];
                const size_t n = std::min(curLen, outLen - o);
                std::memmove(out + o, out + pos[code], n);
                o += n;
            } else if (code == next && havePrev) {
                // KwKwK: the previous string plus its own first byte.
                curLen = prevLen + 1;
                const size_t n = std::min(prevLen, outLen - o);
                std::memmove(out + o, out + prevPos, n);
                o += n;
                if (o < outLen) out[o++] = out[prevPos];
            } else {
                return o > 0;// corrupt stream: keep what decoded
            }

            if (havePrev && next < kMaxCodes) {
                pos[next] = static_cast<uint32_t>(prevPos);
                len[next] = static_cast<uint32_t>(prevLen + 1);
                ++next;
                if (next + 1 >= (1u << width) && width < 12) ++width;
            }
            havePrev = true;
            prevPos = curPos;
            prevLen = curLen;
        }
        return true;
    }

    bool inflate(const unsigned char* in, size_t inLen, unsigned char* out, size_t outLen) {
        if (inLen > size_t(INT_MAX) || outLen > size_t(INT_MAX)) return false;
        int decoded = 0;
        // The growing variant: a writer may pad a segment past the rows the
        // image needs, which the fixed-size stb_zlib_decode_buffer rejects.
        char* buf = stbi_zlib_decode_malloc_guesssize_headerflag(
                reinterpret_cast<const char*>(in), static_cast<int>(inLen),
                static_cast<int>(std::max<size_t>(outLen, 1)), &decoded, 1);
        if (!buf) return false;
        std::memcpy(out, buf, std::min(size_t(decoded), outLen));
        stbi_image_free(buf);
        return true;
    }

}// namespace

namespace threepp::detail {

    bool isTiff(const unsigned char* data, std::size_t size) {
        return size >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] == 42 && data[3] == 0) ||
                             (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == 42));
    }

    std::optional<Image> decodeTiff(const unsigned char* data, std::size_t size, int channels, bool flipY) {

        if (channels < 1 || channels > 4) return fail("requested " + std::to_string(channels) + " channels");
        if (size >= 4 && (data[0] == 'I' || data[0] == 'M') && data[0] == data[1] &&
            (data[0] == 'I' ? data[2] == 43 : data[3] == 43)) {
            return fail("BigTIFF is not supported");
        }
        if (!isTiff(data, size) || size < 8) return fail("not a TIFF stream");

        const Reader r{data, size, data[0] == 'I'};

        // --- The first IFD ---
        const size_t ifd = r.u32(4);
        if (!r.has(ifd, 2)) return fail("IFD offset out of range");
        const size_t entryCount = r.u16(ifd);
        if (!r.has(ifd + 2, entryCount * 12)) return fail("IFD out of range");

        std::map<uint16_t, Entry> tags;
        for (size_t k = 0; k < entryCount; ++k) {
            const size_t at = ifd + 2 + k * 12;
            Entry e{r.u16(at + 2), r.u32(at + 4), 0};
            const size_t bytes = typeSize(e.type) * size_t(e.count);
            e.valueOffset = bytes <= 4 ? at + 8 : r.u32(at + 8);
            tags[r.u16(at)] = e;
        }
        auto list = [&](uint16_t tag) -> std::optional<std::vector<uint32_t>> {
            const auto it = tags.find(tag);
            if (it == tags.end()) return std::nullopt;
            return values(r, it->second);
        };
        auto scalar = [&](uint16_t tag, uint32_t fallback) -> uint32_t {
            const auto v = list(tag);
            return v && !v->empty() ? v->front() : fallback;
        };

        // --- What the image is ---
        const uint32_t width = scalar(kTagWidth, 0);
        const uint32_t height = scalar(kTagHeight, 0);
        if (width == 0 || height == 0) return fail("missing or zero ImageWidth/ImageLength");
        if (uint64_t(width) * height > (uint64_t(1) << 30)) return fail("image too large");

        const uint32_t spp = scalar(kTagSamplesPerPixel, 1);
        if (spp == 0 || spp > 16) return fail("SamplesPerPixel " + std::to_string(spp));

        const auto bpsList = list(kTagBitsPerSample).value_or(std::vector<uint32_t>{1});
        const uint32_t bps = bpsList.empty() ? 1 : bpsList.front();
        for (uint32_t b : bpsList) {
            if (b != bps) return fail("mixed BitsPerSample");
        }
        if (bps != 8 && bps != 16) return fail(std::to_string(bps) + "-bit samples are not supported (8 and 16 are)");
        const size_t bytesPerSample = bps / 8;

        if (const auto formats = list(kTagSampleFormat)) {
            for (uint32_t f : *formats) {
                if (f != 1) return fail(f == 3 ? "floating-point samples are not supported" : "signed samples are not supported");
            }
        }
        if (scalar(kTagFillOrder, 1) != 1) return fail("FillOrder 2 is not supported");

        const uint32_t compression = scalar(kTagCompression, kCompressionNone);
        if (compression != kCompressionNone && compression != kCompressionLzw && compression != kCompressionDeflate &&
            compression != kCompressionDeflateOld && compression != kCompressionPackBits) {
            return fail("compression " + std::to_string(compression) +
                        (compression == 6 || compression == 7 ? " (JPEG)" : "") + " is not supported");
        }
        const uint32_t predictor = scalar(kTagPredictor, 1);
        if (predictor != 1 && predictor != 2) return fail("predictor " + std::to_string(predictor) + " is not supported");

        const uint32_t photometric = scalar(kTagPhotometric, spp >= 3 ? 2 : 1);
        if (photometric > 3) return fail("photometric interpretation " + std::to_string(photometric) + " (CMYK/YCbCr/...) is not supported");
        const uint32_t colourSamples = photometric == 2 ? 3 : 1;
        if (spp < colourSamples) return fail("RGB image with " + std::to_string(spp) + " samples per pixel");
        const int alphaSample = spp > colourSamples ? static_cast<int>(colourSamples) : -1;
        const auto extra = list(kTagExtraSamples);
        const bool associatedAlpha = alphaSample >= 0 && extra && !extra->empty() && extra->front() == 1;

        std::vector<uint32_t> colorMap;
        const size_t paletteSize = size_t(1) << bps;
        if (photometric == 3) {
            colorMap = list(kTagColorMap).value_or(std::vector<uint32_t>{});
            if (colorMap.size() < 3 * paletteSize) return fail("palette image without a full ColorMap");
        }

        const uint32_t planar = scalar(kTagPlanarConfig, 1);
        if (planar != 1 && planar != 2) return fail("PlanarConfiguration " + std::to_string(planar));
        const size_t planes = planar == 2 ? spp : 1;
        const size_t samplesPerSegment = planar == 2 ? 1 : spp;

        // --- Where the pixels are: strips or tiles ---
        const bool tiled = tags.count(kTagTileWidth) > 0;
        uint32_t segWidth, segHeight;
        std::optional<std::vector<uint32_t>> offsets, counts;
        if (tiled) {
            segWidth = scalar(kTagTileWidth, 0);
            segHeight = scalar(kTagTileLength, 0);
            offsets = list(kTagTileOffsets);
            counts = list(kTagTileByteCounts);
        } else {
            segWidth = width;
            segHeight = std::min(scalar(kTagRowsPerStrip, height), height);
            offsets = list(kTagStripOffsets);
            counts = list(kTagStripByteCounts);
        }
        if (segWidth == 0 || segHeight == 0) return fail("zero tile or strip size");
        const size_t across = (size_t(width) + segWidth - 1) / segWidth;
        const size_t down = (size_t(height) + segHeight - 1) / segHeight;
        const size_t segmentsPerPlane = across * down;
        if (!offsets || !counts || offsets->size() < segmentsPerPlane * planes || counts->size() < segmentsPerPlane * planes) {
            return fail("missing or short strip/tile offsets");
        }

        // --- Decode every segment into one chunky buffer of raw samples ---
        const size_t pixelBytes = spp * bytesPerSample;
        std::vector<unsigned char> raw(size_t(width) * height * pixelBytes);
        const size_t segRowBytes = size_t(segWidth) * samplesPerSegment * bytesPerSample;
        std::vector<unsigned char> seg;

        for (size_t plane = 0; plane < planes; ++plane) {
            for (size_t s = 0; s < segmentsPerPlane; ++s) {
                const size_t index = plane * segmentsPerPlane + s;
                const size_t x0 = (s % across) * segWidth;
                const size_t y0 = (s / across) * segHeight;
                const size_t rows = tiled ? segHeight : std::min<size_t>(segHeight, height - y0);

                const size_t offset = (*offsets)[index];
                if (offset >= size) return fail("strip/tile offset out of range");
                const size_t inLen = std::min<size_t>((*counts)[index], size - offset);
                const unsigned char* in = data + offset;

                seg.assign(segRowBytes * rows, 0);
                bool ok = true;
                switch (compression) {
                    case kCompressionNone:
                        std::memcpy(seg.data(), in, std::min(inLen, seg.size()));
                        break;
                    case kCompressionPackBits:
                        unpackBits(in, inLen, seg.data(), seg.size());
                        break;
                    case kCompressionLzw:
                        ok = unLzw(in, inLen, seg.data(), seg.size());
                        break;
                    default:
                        ok = inflate(in, inLen, seg.data(), seg.size());
                        break;
                }
                if (!ok) return fail("corrupt compressed data");

                if (predictor == 2) {
                    // Horizontal differencing: each sample is stored as its
                    // difference from the same sample one pixel to the left.
                    const size_t rowSamples = size_t(segWidth) * samplesPerSegment;
                    for (size_t row = 0; row < rows; ++row) {
                        unsigned char* p = seg.data() + row * segRowBytes;
                        if (bytesPerSample == 1) {
                            for (size_t k = samplesPerSegment; k < rowSamples; ++k) {
                                p[k] = static_cast<unsigned char>(p[k] + p[k - samplesPerSegment]);
                            }
                        } else {
                            const Reader rr{p, segRowBytes, r.little};
                            for (size_t k = samplesPerSegment; k < rowSamples; ++k) {
                                const auto v = static_cast<uint16_t>(rr.u16(k * 2) + rr.u16((k - samplesPerSegment) * 2));
                                p[k * 2 + (r.little ? 0 : 1)] = static_cast<unsigned char>(v & 0xff);
                                p[k * 2 + (r.little ? 1 : 0)] = static_cast<unsigned char>(v >> 8);
                            }
                        }
                    }
                }

                const size_t columns = std::min<size_t>(segWidth, width - x0);
                for (size_t row = 0; row < rows && y0 + row < height; ++row) {
                    const unsigned char* src = seg.data() + row * segRowBytes;
                    unsigned char* dst = raw.data() + ((y0 + row) * width + x0) * pixelBytes;
                    if (planes == 1) {
                        std::memcpy(dst, src, columns * pixelBytes);
                    } else {
                        for (size_t x = 0; x < columns; ++x) {
                            std::memcpy(dst + x * pixelBytes + plane * bytesPerSample, src + x * bytesPerSample, bytesPerSample);
                        }
                    }
                }
            }
        }

        // --- Convert to the requested 8-bit layout ---
        // 16-bit samples keep their high byte, as stb does for 16-bit PNGs.
        const size_t highByte = bytesPerSample == 2 && r.little ? 1 : 0;
        auto sample8 = [&](const unsigned char* px, size_t k) -> unsigned {
            return px[k * bytesPerSample + highByte];
        };
        auto sampleFull = [&](const unsigned char* px, size_t k) -> size_t {
            if (bytesPerSample == 1) return px[k];
            const Reader rr{px + k * 2, 2, r.little};
            return rr.u16(0);
        };

        std::vector<unsigned char> pixels(size_t(width) * height * channels);
        for (size_t y = 0; y < height; ++y) {
            const size_t outRow = flipY ? height - 1 - y : y;
            const unsigned char* src = raw.data() + y * width * pixelBytes;
            unsigned char* dst = pixels.data() + outRow * width * channels;
            for (size_t x = 0; x < width; ++x, src += pixelBytes, dst += channels) {
                unsigned red, green, blue;
                if (photometric == 2) {
                    red = sample8(src, 0);
                    green = sample8(src, 1);
                    blue = sample8(src, 2);
                } else if (photometric == 3) {
                    const size_t i = std::min(sampleFull(src, 0), paletteSize - 1);
                    red = colorMap[i] >> 8;
                    green = colorMap[paletteSize + i] >> 8;
                    blue = colorMap[2 * paletteSize + i] >> 8;
                } else {
                    const unsigned v = sample8(src, 0);
                    red = green = blue = photometric == 0 ? 255 - v : v;
                }
                const unsigned alpha = alphaSample >= 0 ? sample8(src, alphaSample) : 255;
                if (associatedAlpha && alpha > 0 && alpha < 255) {
                    red = std::min(255u, (red * 255 + alpha / 2) / alpha);
                    green = std::min(255u, (green * 255 + alpha / 2) / alpha);
                    blue = std::min(255u, (blue * 255 + alpha / 2) / alpha);
                }
                // stb's luma weights, so a grey TIFF and a grey PNG agree.
                const auto luma = static_cast<unsigned char>((red * 77 + green * 150 + blue * 29) >> 8);
                switch (channels) {
                    case 1:
                        dst[0] = luma;
                        break;
                    case 2:
                        dst[0] = luma;
                        dst[1] = static_cast<unsigned char>(alpha);
                        break;
                    case 3:
                        dst[0] = static_cast<unsigned char>(red);
                        dst[1] = static_cast<unsigned char>(green);
                        dst[2] = static_cast<unsigned char>(blue);
                        break;
                    default:
                        dst[0] = static_cast<unsigned char>(red);
                        dst[1] = static_cast<unsigned char>(green);
                        dst[2] = static_cast<unsigned char>(blue);
                        dst[3] = static_cast<unsigned char>(alpha);
                        break;
                }
            }
        }

        return Image{std::move(pixels), width, height};
    }

}// namespace threepp::detail
