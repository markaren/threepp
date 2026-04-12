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
                case 74: return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;        // BC2_UNORM
                case 77: return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;        // BC3_UNORM
                case 80: return GL_COMPRESSED_RED_RGTC1;                  // BC4_UNORM
                case 83: return GL_COMPRESSED_RG_RGTC2;                   // BC5_UNORM
                case 95: return GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;   // BC6H_UF16
                case 98: return GL_COMPRESSED_RGBA_BPTC_UNORM;           // BC7_UNORM
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

                mips.emplace_back(std::move(buf), w, h, glFmt);
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
