#include "threepp/loaders/DDSLoader.hpp"

#include "threepp/textures/Texture.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace threepp {

    namespace {

        constexpr uint32_t DDS_MAGIC    = 0x20534444u; // "DDS "
        constexpr uint32_t DDPF_FOURCC  = 0x00000004u;

        // GL compressed-format tokens (numeric to avoid requiring extension headers).
        constexpr uint32_t GL_COMPRESSED_RGB_S3TC_DXT1_EXT          = 0x83F0u;
        constexpr uint32_t GL_COMPRESSED_RGBA_S3TC_DXT1_EXT         = 0x83F1u;
        constexpr uint32_t GL_COMPRESSED_RGBA_S3TC_DXT3_EXT         = 0x83F2u;
        constexpr uint32_t GL_COMPRESSED_RGBA_S3TC_DXT5_EXT         = 0x83F3u;
        constexpr uint32_t GL_COMPRESSED_RED_RGTC1                  = 0x8DBBu;
        constexpr uint32_t GL_COMPRESSED_RG_RGTC2                   = 0x8DBDu;
        constexpr uint32_t GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT    = 0x8E8Fu;
        constexpr uint32_t GL_COMPRESSED_RGBA_BPTC_UNORM            = 0x8E8Cu;
        // sRGB variants — used for BaseColor / Emissive textures.
        constexpr uint32_t GL_COMPRESSED_SRGB_S3TC_DXT1_EXT         = 0x8C4Cu;
        constexpr uint32_t GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT   = 0x8C4Fu;
        constexpr uint32_t GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM      = 0x8E8Du;

        constexpr uint32_t makeFourCC(char a, char b, char c, char d) {
            return static_cast<uint32_t>(a)
                 | (static_cast<uint32_t>(b) << 8)
                 | (static_cast<uint32_t>(c) << 16)
                 | (static_cast<uint32_t>(d) << 24);
        }

#pragma pack(push, 1)
        struct DDSPixelFormat {
            uint32_t dwSize;
            uint32_t dwFlags;
            uint32_t dwFourCC;
            uint32_t dwRGBBitCount;
            uint32_t dwRBitMask, dwGBitMask, dwBBitMask, dwABitMask;
        };

        struct DDSHeader {
            uint32_t     dwSize;           // must be 124
            uint32_t     dwFlags;
            uint32_t     dwHeight;
            uint32_t     dwWidth;
            uint32_t     dwPitchOrLinearSize;
            uint32_t     dwDepth;
            uint32_t     dwMipMapCount;
            uint32_t     dwReserved1[11];
            DDSPixelFormat ddspf;
            uint32_t     dwCaps;
            uint32_t     dwCaps2;
            uint32_t     dwCaps3;
            uint32_t     dwCaps4;
            uint32_t     dwReserved2;
        };  // 124 bytes

        struct DDSHeaderDXT10 {
            uint32_t dxgiFormat;
            uint32_t resourceDimension;
            uint32_t miscFlag;
            uint32_t arraySize;
            uint32_t miscFlags2;
        };  // 20 bytes
#pragma pack(pop)

        // Map DXGI_FORMAT to GL compressed format; returns 0 if unsupported.
        uint32_t dxgiToGL(uint32_t dxgi) {
            switch (dxgi) {
                case 71: return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;        // BC1_UNORM
                case 72: return GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;        // BC1_UNORM_SRGB
                case 74: return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;        // BC2_UNORM
                case 77: return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;        // BC3_UNORM
                case 78: return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;  // BC3_UNORM_SRGB
                case 80: return GL_COMPRESSED_RED_RGTC1;                  // BC4_UNORM
                case 83: return GL_COMPRESSED_RG_RGTC2;                   // BC5_UNORM
                case 95: return GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;   // BC6H_UF16
                case 98: return GL_COMPRESSED_RGBA_BPTC_UNORM;           // BC7_UNORM
                case 99: return GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM;     // BC7_UNORM_SRGB
                default: return 0;
            }
        }

        uint32_t blockBytes(uint32_t glFmt) {
            if (glFmt == GL_COMPRESSED_RGB_S3TC_DXT1_EXT  ||
                glFmt == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT ||
                glFmt == GL_COMPRESSED_RED_RGTC1)
                return 8;
            return 16;
        }

        uint32_t mipDataSize(uint32_t w, uint32_t h, uint32_t bb) {
            return ((w + 3u) / 4u) * ((h + 3u) / 4u) * bb;
        }

        // -----------------------------------------------------------------------
        // Vertical flip of a compressed mip level.
        //
        // DDS files store rows top-to-bottom; OpenGL (and stb_image-loaded
        // textures) expect bottom-to-top.  We must:
        //   1. Reverse the order of 4×4 block-rows in memory.
        //   2. Flip the pixel-row indexing *within* each block so the four
        //      pixel rows are also reversed.
        //
        // This keeps DDS textures consistent with the stb_image flipY=true
        // convention used for all other texture formats.
        // -----------------------------------------------------------------------

        // Flip the 3-bit index rows inside a BC4/DXT5-alpha 6-byte bitfield.
        // The 48 bits encode 16 pixels × 3 bits in row-major order (row 0 first).
        inline void flipBC4Bits(uint8_t* p6) {
            uint64_t bits = 0;
            for (int i = 0; i < 6; ++i) bits |= static_cast<uint64_t>(p6[i]) << (i * 8);
            const uint64_t r0 = (bits >>  0) & 0xFFFu;
            const uint64_t r1 = (bits >> 12) & 0xFFFu;
            const uint64_t r2 = (bits >> 24) & 0xFFFu;
            const uint64_t r3 = (bits >> 36) & 0xFFFu;
            const uint64_t flipped = r3 | (r2 << 12) | (r1 << 24) | (r0 << 36);
            for (int i = 0; i < 6; ++i) p6[i] = static_cast<uint8_t>((flipped >> (i * 8)) & 0xFF);
        }

        inline void flipBlockInPlace(uint8_t* b, uint32_t glFmt) {
            const bool isDXT1 = (glFmt == GL_COMPRESSED_RGB_S3TC_DXT1_EXT ||
                                  glFmt == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT);
            const bool isDXT3 = (glFmt == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT);
            const bool isDXT5 = (glFmt == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT);
            const bool isBC4  = (glFmt == GL_COMPRESSED_RED_RGTC1);
            const bool isBC5  = (glFmt == GL_COMPRESSED_RG_RGTC2);

            if (isDXT1) {
                // bytes 4-7 are the 2-bit row indices: row0=b[4], row3=b[7]
                std::swap(b[4], b[7]);
                std::swap(b[5], b[6]);
            } else if (isDXT3) {
                // bytes 0-7: 4-bit explicit alpha, 2 bytes per pixel-row
                std::swap(b[0], b[6]); std::swap(b[1], b[7]);
                std::swap(b[2], b[4]); std::swap(b[3], b[5]);
                // bytes 8-15: DXT1 colour block
                std::swap(b[12], b[15]);
                std::swap(b[13], b[14]);
            } else if (isDXT5) {
                // bytes 0-1: alpha0/alpha1; bytes 2-7: 3-bit index rows
                flipBC4Bits(b + 2);
                // bytes 8-15: DXT1 colour block
                std::swap(b[12], b[15]);
                std::swap(b[13], b[14]);
            } else if (isBC4) {
                flipBC4Bits(b + 2);
            } else if (isBC5) {
                flipBC4Bits(b + 2);      // R channel
                flipBC4Bits(b + 8 + 2);  // G channel
            }
            // BC6H / BC7: not handled; those formats are already rare and the
            // decoders for their internal row layout are non-trivial.
        }

        // Flip an entire mip level in-place.
        void flipMipY(uint8_t* data, uint32_t w, uint32_t h, uint32_t glFmt) {
            const uint32_t bb      = blockBytes(glFmt);
            const uint32_t blocksX = (w + 3u) / 4u;
            const uint32_t blocksY = (h + 3u) / 4u;
            const uint32_t rowBytes = blocksX * bb;

            // Swap pairs of block-rows from outside in, flipping each block.
            for (uint32_t top = 0, bot = blocksY - 1; top < bot; ++top, --bot) {
                uint8_t* rowT = data + top * rowBytes;
                uint8_t* rowB = data + bot * rowBytes;
                for (uint32_t col = 0; col < blocksX; ++col) {
                    uint8_t* blkT = rowT + col * bb;
                    uint8_t* blkB = rowB + col * bb;
                    // Flip internal rows, then swap the two blocks.
                    flipBlockInPlace(blkT, glFmt);
                    flipBlockInPlace(blkB, glFmt);
                    for (uint32_t byte = 0; byte < bb; ++byte)
                        std::swap(blkT[byte], blkB[byte]);
                }
            }
            // Middle row (odd blocksY): just flip internal rows.
            if (blocksY % 2 == 1) {
                uint8_t* mid = data + (blocksY / 2) * rowBytes;
                for (uint32_t col = 0; col < blocksX; ++col)
                    flipBlockInPlace(mid + col * bb, glFmt);
            }
        }

        std::shared_ptr<Texture> parseDDS(const uint8_t* raw, size_t size) {
            if (size < 4 + sizeof(DDSHeader)) {
                std::cerr << "[DDSLoader] File too small\n";
                return nullptr;
            }

            uint32_t magic = 0;
            std::memcpy(&magic, raw, 4);
            if (magic != DDS_MAGIC) {
                std::cerr << "[DDSLoader] Invalid DDS magic\n";
                return nullptr;
            }

            DDSHeader hdr{};
            std::memcpy(&hdr, raw + 4, sizeof(DDSHeader));
            if (hdr.dwSize != 124) {
                std::cerr << "[DDSLoader] Unexpected DDS header size\n";
                return nullptr;
            }

            size_t dataOffset = 4 + sizeof(DDSHeader);

            // Determine GL compressed format.
            uint32_t glFmt = 0;
            if (hdr.ddspf.dwFlags & DDPF_FOURCC) {
                const uint32_t fcc = hdr.ddspf.dwFourCC;
                if      (fcc == makeFourCC('D','X','T','1')) glFmt = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
                else if (fcc == makeFourCC('D','X','T','3')) glFmt = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
                else if (fcc == makeFourCC('D','X','T','5')) glFmt = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
                else if (fcc == makeFourCC('A','T','I','1')) glFmt = GL_COMPRESSED_RED_RGTC1;
                else if (fcc == makeFourCC('A','T','I','2')) glFmt = GL_COMPRESSED_RG_RGTC2;
                else if (fcc == makeFourCC('D','X','1','0')) {
                    if (size < dataOffset + sizeof(DDSHeaderDXT10)) {
                        std::cerr << "[DDSLoader] Truncated DX10 header\n";
                        return nullptr;
                    }
                    DDSHeaderDXT10 dx10{};
                    std::memcpy(&dx10, raw + dataOffset, sizeof(DDSHeaderDXT10));
                    dataOffset += sizeof(DDSHeaderDXT10);
                    glFmt = dxgiToGL(dx10.dxgiFormat);
                }
            }

            if (glFmt == 0) {
                std::cerr << "[DDSLoader] Unsupported DDS format (FourCC=0x"
                          << std::hex << hdr.ddspf.dwFourCC << std::dec << ")\n";
                return nullptr;
            }

            const uint32_t bb       = blockBytes(glFmt);
            const uint32_t mipCount = std::max(1u, hdr.dwMipMapCount);

            // Parse each mip level into a separate Image.
            std::vector<Image> mips;
            mips.reserve(mipCount);

            for (uint32_t i = 0; i < mipCount; ++i) {
                const uint32_t w = std::max(1u, hdr.dwWidth  >> i);
                const uint32_t h = std::max(1u, hdr.dwHeight >> i);
                const uint32_t bytes = mipDataSize(w, h, bb);

                if (dataOffset + bytes > size) {
                    std::cerr << "[DDSLoader] Truncated mip level " << i << "\n";
                    break;
                }

                std::vector<unsigned char> buf(bytes);
                std::memcpy(buf.data(), raw + dataOffset, bytes);
                dataOffset += bytes;

                // Flip Y to match stb_image's flipY=true convention so DDS
                // textures are consistent with all other texture formats.
                flipMipY(buf.data(), w, h, glFmt);

                Image::CompressedFormat cf{};
                cf.format = glFmt;
                mips.emplace_back(std::move(buf), w, h,cf);
            }

            if (mips.empty()) return nullptr;

            // Level 0 goes into the base image; remaining levels into mipmaps().
            auto tex = Texture::create(mips[0]);
            for (size_t i = 1; i < mips.size(); ++i)
                tex->mipmaps().push_back(std::move(mips[i]));

            tex->generateMipmaps = false;
            tex->format = Format::RGBA;
            tex->needsUpdate();
            return tex;
        }

    }// namespace

    struct DDSLoader::Impl {};

    DDSLoader::DDSLoader() : pimpl_(std::make_unique<Impl>()) {}
    DDSLoader::~DDSLoader() = default;

    std::shared_ptr<Texture> DDSLoader::load(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "[DDSLoader] Cannot open: " << path << "\n";
            return nullptr;
        }
        const auto fileSize = file.tellg();
        file.seekg(0);
        std::vector<uint8_t> data(static_cast<size_t>(fileSize));
        file.read(reinterpret_cast<char*>(data.data()), fileSize);
        file.close();

        auto tex = parseDDS(data.data(), data.size());
        if (tex) tex->name = path.stem().string();
        return tex;
    }

    std::shared_ptr<Texture> DDSLoader::loadFromMemory(const std::vector<unsigned char>& data) {
        return parseDDS(data.data(), data.size());
    }

}// namespace threepp
