#include "WgpuPathTracerBCn.hpp"

#include <cstring>

namespace threepp::wgpu_pt {

namespace {

inline void bc_unpackRGB565(std::uint16_t c, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) {
    r = static_cast<std::uint8_t>((c >> 11) * 255 / 31);
    g = static_cast<std::uint8_t>(((c >> 5) & 0x3F) * 255 / 63);
    b = static_cast<std::uint8_t>((c & 0x1F) * 255 / 31);
}

// Decode one 4×4 BC4 block (8 bytes) into 16 uint8 red values.
inline void bc_decodeBC4Block(const std::uint8_t* src, std::uint8_t out[16]) {
    const std::uint8_t r0 = src[0], r1 = src[1];
    std::uint8_t pal[8];
    pal[0] = r0; pal[1] = r1;
    if (r0 > r1) {
        pal[2] = static_cast<std::uint8_t>((6*r0 + 1*r1) / 7);
        pal[3] = static_cast<std::uint8_t>((5*r0 + 2*r1) / 7);
        pal[4] = static_cast<std::uint8_t>((4*r0 + 3*r1) / 7);
        pal[5] = static_cast<std::uint8_t>((3*r0 + 4*r1) / 7);
        pal[6] = static_cast<std::uint8_t>((2*r0 + 5*r1) / 7);
        pal[7] = static_cast<std::uint8_t>((1*r0 + 6*r1) / 7);
    } else {
        pal[2] = static_cast<std::uint8_t>((4*r0 + 1*r1) / 5);
        pal[3] = static_cast<std::uint8_t>((3*r0 + 2*r1) / 5);
        pal[4] = static_cast<std::uint8_t>((2*r0 + 3*r1) / 5);
        pal[5] = static_cast<std::uint8_t>((1*r0 + 4*r1) / 5);
        pal[6] = 0; pal[7] = 255;
    }
    std::uint64_t bits = 0;
    for (int i = 0; i < 6; ++i) bits |= static_cast<std::uint64_t>(src[2 + i]) << (i * 8);
    for (int i = 0; i < 16; ++i)
        out[i] = pal[(bits >> (i * 3)) & 7];
}

// Decode one 4×4 DXT1 (BC1) block (8 bytes) into 64 bytes RGBA.
inline void bc_decodeDXT1Block(const std::uint8_t* src, std::uint8_t out[64], bool punchThrough) {
    std::uint16_t c0, c1;
    std::memcpy(&c0, src, 2); std::memcpy(&c1, src + 2, 2);
    std::uint8_t r[4], g[4], b[4], a[4];
    bc_unpackRGB565(c0, r[0], g[0], b[0]); a[0] = 255;
    bc_unpackRGB565(c1, r[1], g[1], b[1]); a[1] = 255;
    if (c0 > c1 || !punchThrough) {
        r[2]=(2*r[0]+r[1])/3; g[2]=(2*g[0]+g[1])/3; b[2]=(2*b[0]+b[1])/3; a[2]=255;
        r[3]=(r[0]+2*r[1])/3; g[3]=(g[0]+2*g[1])/3; b[3]=(b[0]+2*b[1])/3; a[3]=255;
    } else {
        r[2]=(r[0]+r[1])/2; g[2]=(g[0]+g[1])/2; b[2]=(b[0]+b[1])/2; a[2]=255;
        r[3]=0; g[3]=0; b[3]=0; a[3]=0;
    }
    std::uint32_t idx; std::memcpy(&idx, src + 4, 4);
    for (int i = 0; i < 16; ++i) {
        const int p = (idx >> (i * 2)) & 3;
        out[i*4+0]=r[p]; out[i*4+1]=g[p]; out[i*4+2]=b[p]; out[i*4+3]=a[p];
    }
}

}// namespace

// Decompress an entire BCn texture level to RGBA8.
// Returns an empty vector if the format is unsupported (e.g. BC6H).
// sRGB variants map to their non-sRGB counterparts: the block encoding is
// identical; sRGB is only a GPU-side sampling flag, not a different codec.
std::vector<std::uint8_t> bcnDecompress(const std::uint8_t* blocks, int w, int h, unsigned int glFmt) {
    constexpr unsigned int DXT1_RGB  = 0x83F0u;
    constexpr unsigned int DXT1_RGBA = 0x83F1u;
    constexpr unsigned int DXT3      = 0x83F2u;
    constexpr unsigned int DXT5      = 0x83F3u;
    constexpr unsigned int BC4       = 0x8DBBu;
    constexpr unsigned int BC5       = 0x8DBDu;
    // sRGB variants — identical block layout, just a different sampling flag.
    constexpr unsigned int DXT1_SRGB = 0x8C4Cu;
    constexpr unsigned int DXT5_SRGB = 0x8C4Fu;
    constexpr unsigned int BC7       = 0x8E8Cu;
    constexpr unsigned int BC7_SRGB  = 0x8E8Du;

    if (glFmt == BC7 || glFmt == BC7_SRGB) {
        // BC7 decoder for the path-tracer atlas (→ RGBA8 output).
        // Implements mode 6 fully (1 subset, 7-bit endpoints + P-bit, 4-bit
        // indices) — the dominant mode in typical PBR base-colour textures.
        // Other modes fall back to extracting endpoint 0 from the packed bits;
        // not pixel-perfect but preserves correct hue/tone for atlas use.
        auto readBits64 = [](std::uint64_t lo, std::uint64_t hi, int start, int n) -> std::uint32_t {
            std::uint32_t val = 0;
            for (int i = 0; i < n; ++i) {
                const int b = start + i;
                const std::uint64_t w = (b < 64) ? lo : hi;
                val |= static_cast<std::uint32_t>((w >> (b < 64 ? b : b-64)) & 1u) << i;
            }
            return val;
        };
        // 4-bit BC7 interpolation weights (×1/64 fraction).
        constexpr std::uint8_t w4[16] = {0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64};

        const int blocksX = (w + 3) / 4;
        const int blocksY = (h + 3) / 4;
        std::vector<std::uint8_t> out(static_cast<std::size_t>(w * h) * 4, 255u);
        const std::uint8_t* p = blocks;
        for (int by = 0; by < blocksY; ++by) {
            for (int bx = 0; bx < blocksX; ++bx) {
                std::uint64_t lo = 0, hi = 0;
                std::memcpy(&lo, p, 8); std::memcpy(&hi, p + 8, 8);

                // Mode = position of lowest set bit in the 128-bit block.
                int mode = 8;
                for (int i = 0; i < 8; ++i) { if ((p[0] >> i) & 1) { mode = i; break; } }

                std::uint8_t rgba[64] = {};
                if (mode == 6) {
                    // Bit layout (offset from LSB):
                    // [6]=1 (mode), R0[6:0]@7, R1[6:0]@14, G0[6:0]@21,
                    // G1[6:0]@28, B0[6:0]@35, B1[6:0]@42, A0[6:0]@49,
                    // A1[6:0]@56, P0@63, P1@64, anchor-idx(3b)@65,
                    // 15×4b-idx.
                    const std::uint32_t R0=readBits64(lo,hi,7,7),  R1=readBits64(lo,hi,14,7);
                    const std::uint32_t G0=readBits64(lo,hi,21,7), G1=readBits64(lo,hi,28,7);
                    const std::uint32_t B0=readBits64(lo,hi,35,7), B1=readBits64(lo,hi,42,7);
                    const std::uint32_t A0=readBits64(lo,hi,49,7), A1=readBits64(lo,hi,56,7);
                    const std::uint32_t P0=readBits64(lo,hi,63,1), P1=readBits64(lo,hi,64,1);
                    // Expand to 8-bit: append P-bit as LSB.
                    const std::uint32_t er0=(R0<<1)|P0, er1=(R1<<1)|P1;
                    const std::uint32_t eg0=(G0<<1)|P0, eg1=(G1<<1)|P1;
                    const std::uint32_t eb0=(B0<<1)|P0, eb1=(B1<<1)|P1;
                    const std::uint32_t ea0=(A0<<1)|P0, ea1=(A1<<1)|P1;
                    // Anchor index (pixel 0) uses 3 bits; rest use 4.
                    for (int i = 0; i < 16; ++i) {
                        const int off = (i == 0) ? 65 : (68 + (i-1)*4);
                        const int nb  = (i == 0) ? 3  : 4;
                        const std::uint32_t wt = w4[readBits64(lo,hi,off,nb) & 0xF];
                        const std::uint32_t iw = 64 - wt;
                        rgba[i*4+0]=static_cast<std::uint8_t>((er0*iw+er1*wt+32)>>6);
                        rgba[i*4+1]=static_cast<std::uint8_t>((eg0*iw+eg1*wt+32)>>6);
                        rgba[i*4+2]=static_cast<std::uint8_t>((eb0*iw+eb1*wt+32)>>6);
                        rgba[i*4+3]=static_cast<std::uint8_t>((ea0*iw+ea1*wt+32)>>6);
                    }
                } else {
                    // Other modes: extract endpoint 0 from known bit offsets.
                    // The R/G/B endpoints always start after the mode bit(s)
                    // and partition data; using a fixed offset of 7 bits past
                    // the mode gives a reasonable colour for atlas purposes.
                    const std::uint8_t r = static_cast<std::uint8_t>(readBits64(lo,hi,7,7)<<1);
                    const std::uint8_t g = static_cast<std::uint8_t>(readBits64(lo,hi,21,7)<<1);
                    const std::uint8_t b = static_cast<std::uint8_t>(readBits64(lo,hi,35,7)<<1);
                    const std::uint8_t a = static_cast<std::uint8_t>(readBits64(lo,hi,49,7)<<1);
                    for (int i = 0; i < 16; ++i)
                        { rgba[i*4]=r; rgba[i*4+1]=g; rgba[i*4+2]=b; rgba[i*4+3]=a; }
                }
                const int px = bx*4, py = by*4;
                for (int y = 0; y < 4 && (py+y)<h; ++y)
                    for (int x = 0; x < 4 && (px+x)<w; ++x)
                        std::memcpy(&out[(static_cast<std::size_t>(py+y)*w+(px+x))*4],
                                    &rgba[(y*4+x)*4], 4);
                p += 16;
            }
        }
        return out;
    }

    // Map sRGB variants to their non-sRGB equivalents.
    if (glFmt == DXT1_SRGB) glFmt = DXT1_RGBA;
    if (glFmt == DXT5_SRGB) glFmt = DXT5;

    const bool isDXT1 = (glFmt == DXT1_RGB || glFmt == DXT1_RGBA);
    const bool isDXT3 = (glFmt == DXT3);
    const bool isDXT5 = (glFmt == DXT5);
    const bool isBC4  = (glFmt == BC4);
    const bool isBC5  = (glFmt == BC5);
    if (!isDXT1 && !isDXT3 && !isDXT5 && !isBC4 && !isBC5) return {};

    const int blocksX = (w + 3) / 4;
    const int blocksY = (h + 3) / 4;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(w * h) * 4, 255u);

    const std::uint8_t* p = blocks;
    for (int by = 0; by < blocksY; ++by) {
        for (int bx = 0; bx < blocksX; ++bx) {
            std::uint8_t rgba[64] = {};
            if (isDXT1) {
                bc_decodeDXT1Block(p, rgba, glFmt == DXT1_RGBA);
                p += 8;
            } else if (isDXT3) {
                std::uint8_t alpha[16];
                for (int i = 0; i < 8; ++i) {
                    alpha[i*2+0] = static_cast<std::uint8_t>((p[i] & 0x0F) * 17);
                    alpha[i*2+1] = static_cast<std::uint8_t>((p[i] >> 4)   * 17);
                }
                bc_decodeDXT1Block(p + 8, rgba, false);
                for (int i = 0; i < 16; ++i) rgba[i*4+3] = alpha[i];
                p += 16;
            } else if (isDXT5) {
                std::uint8_t alpha[16];
                bc_decodeBC4Block(p, alpha);
                bc_decodeDXT1Block(p + 8, rgba, false);
                for (int i = 0; i < 16; ++i) rgba[i*4+3] = alpha[i];
                p += 16;
            } else if (isBC4) {
                std::uint8_t r[16]; bc_decodeBC4Block(p, r);
                for (int i = 0; i < 16; ++i) { rgba[i*4]=r[i]; rgba[i*4+1]=0; rgba[i*4+2]=0; rgba[i*4+3]=255; }
                p += 8;
            } else if (isBC5) {
                std::uint8_t r[16], g[16];
                bc_decodeBC4Block(p, r); bc_decodeBC4Block(p + 8, g);
                for (int i = 0; i < 16; ++i) { rgba[i*4]=r[i]; rgba[i*4+1]=g[i]; rgba[i*4+2]=0; rgba[i*4+3]=255; }
                p += 16;
            }
            // Write 4×4 block into output, clamping to texture bounds.
            const int px = bx * 4, py = by * 4;
            for (int y = 0; y < 4 && (py + y) < h; ++y)
                for (int x = 0; x < 4 && (px + x) < w; ++x)
                    std::memcpy(&out[(static_cast<std::size_t>(py + y) * w + (px + x)) * 4],
                                &rgba[(y * 4 + x) * 4], 4);
        }
    }
    return out;
}

}// namespace threepp::wgpu_pt
