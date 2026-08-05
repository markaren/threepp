
#include "threepp/loaders/exr/PizDecode.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// PIZ, in two halves:
//
//   1. a 2D wavelet transform over each channel's whole block, which turns
//      smooth image data into mostly-small coefficients, and
//   2. Huffman coding of those coefficients, with a run-length escape.
//
// A per-file "bitmap" records which of the 65536 possible 16-bit values actually
// occur, so the Huffman alphabet can be compacted to just those before coding.
//
// The layout below is the file format, not a design choice: this follows
// OpenEXR's ImfWav.cpp / ImfHuf.cpp closely on purpose, down to the shift
// constants and the odd-pixel special cases. Rewriting it into something tidier
// would mean inventing a different format. What IS ours: every read is bounds
// checked and every failure returns false instead of throwing or trusting the
// header, since the input is an untrusted file.

namespace {

    constexpr int USHORT_RANGE = 1 << 16;
    constexpr int BITMAP_SIZE = USHORT_RANGE >> 3;

    constexpr int HUF_ENCBITS = 16;// literal (value) bit length
    constexpr int HUF_DECBITS = 14;// decoding table lookup width
    constexpr int HUF_ENCSIZE = (1 << HUF_ENCBITS) + 1;
    constexpr int HUF_DECSIZE = 1 << HUF_DECBITS;
    constexpr int HUF_DECMASK = HUF_DECSIZE - 1;

    // Code-length values that instead mean "N symbols in a row have no code".
    constexpr int SHORT_ZEROCODE_RUN = 59;
    constexpr int LONG_ZEROCODE_RUN = 63;
    constexpr int SHORTEST_LONG_RUN = 2 + LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN;

    // A symbol's code and its bit length share one 64-bit word.
    constexpr uint64_t hufLength(uint64_t code) { return code & 63; }
    constexpr uint64_t hufCode(uint64_t code) { return code >> 6; }

    uint16_t readU16(const unsigned char* p) {
        return static_cast<uint16_t>(p[0] | (p[1] << 8));
    }

    uint32_t readU32(const unsigned char* p) {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    }

    // ---------------------------------------------------------------- bitmap

    // The bitmap lists which 16-bit values occur anywhere in the block. Turning
    // it back into a lookup table both restores the original values and yields
    // maxValue, which decides whether the wavelet ran in its 14- or 16-bit mode.
    uint16_t reverseLutFromBitmap(const unsigned char bitmap[], uint16_t lut[]) {

        int k = 0;
        for (int i = 0; i < USHORT_RANGE; ++i) {
            if (i == 0 || (bitmap[i >> 3] & (1 << (i & 7)))) {
                lut[k++] = static_cast<uint16_t>(i);
            }
        }
        const int n = k - 1;
        while (k < USHORT_RANGE) lut[k++] = 0;

        return static_cast<uint16_t>(n);
    }

    // --------------------------------------------------------------- wavelet

    // 14-bit mode: values fit in a short, so the lifting steps can use plain
    // signed arithmetic.
    void wdec14(uint16_t l, uint16_t h, uint16_t& a, uint16_t& b) {

        const int16_t ls = static_cast<int16_t>(l);
        const int16_t hs = static_cast<int16_t>(h);

        const int hi = hs;
        const int ai = ls + (hi & 1) + (hi >> 1);

        a = static_cast<uint16_t>(static_cast<int16_t>(ai));
        b = static_cast<uint16_t>(static_cast<int16_t>(ai - hi));
    }

    // 16-bit mode: the full range is in play, so the same lifting is done modulo
    // 2^16 with a half-range offset.
    void wdec16(uint16_t l, uint16_t h, uint16_t& a, uint16_t& b) {

        constexpr int MOD_MASK = (1 << 16) - 1;
        constexpr int A_OFFSET = 1 << 15;

        const int m = l;
        const int d = h;
        const int bb = (m - (d >> 1)) & MOD_MASK;
        const int aa = (d + bb - A_OFFSET) & MOD_MASK;

        b = static_cast<uint16_t>(bb);
        a = static_cast<uint16_t>(aa);
    }

    void wav2Decode(uint16_t* in, int nx, int ox, int ny, int oy, uint16_t mx) {

        const bool w14 = mx < (1 << 14);
        const int n = (nx > ny) ? ny : nx;
        int p = 1;
        int p2;

        // Largest power of two that still fits the smaller dimension — the level
        // the encoder stopped at, and so the one decoding starts from.
        while (p <= n) p <<= 1;
        p >>= 1;
        p2 = p;
        p >>= 1;

        while (p >= 1) {

            uint16_t* py = in;
            uint16_t* ey = in + oy * (ny - p2);
            const int oy1 = oy * p;
            const int oy2 = oy * p2;
            const int ox1 = ox * p;
            const int ox2 = ox * p2;
            uint16_t i00, i01, i10, i11;

            for (; py <= ey; py += oy2) {

                uint16_t* px = py;
                uint16_t* ex = py + ox * (nx - p2);

                for (; px <= ex; px += ox2) {

                    uint16_t* p01 = px + ox1;
                    uint16_t* p10 = px + oy1;
                    uint16_t* p11 = p10 + ox1;

                    if (w14) {
                        wdec14(*px, *p10, i00, i10);
                        wdec14(*p01, *p11, i01, i11);
                        wdec14(i00, i01, *px, *p01);
                        wdec14(i10, i11, *p10, *p11);
                    } else {
                        wdec16(*px, *p10, i00, i10);
                        wdec16(*p01, *p11, i01, i11);
                        wdec16(i00, i01, *px, *p01);
                        wdec16(i10, i11, *p10, *p11);
                    }
                }

                // Odd number of pixels in x: the last column pairs vertically only.
                if (nx & p) {
                    uint16_t* p10 = px + oy1;
                    if (w14) wdec14(*px, *p10, i00, *p10);
                    else wdec16(*px, *p10, i00, *p10);
                    *px = i00;
                }
            }

            // Odd number of pixels in y: the last row pairs horizontally only.
            if (ny & p) {

                uint16_t* px = py;
                uint16_t* ex = py + ox * (nx - p2);

                for (; px <= ex; px += ox2) {
                    uint16_t* p01 = px + ox1;
                    if (w14) wdec14(*px, *p01, i00, *p01);
                    else wdec16(*px, *p01, i00, *p01);
                    *px = i00;
                }
            }

            p2 = p;
            p >>= 1;
        }
    }

    // --------------------------------------------------------------- huffman

    // Bits arrive MSB-first: `c` is the accumulator and `lc` how many of its low
    // bits are live. `ok` latches the first read past the end.
    struct BitReader {

        const unsigned char* p;
        const unsigned char* end;
        uint64_t c{0};
        int lc{0};
        bool ok{true};

        bool getChar() {
            if (p >= end) {
                ok = false;
                return false;
            }
            c = (c << 8) | *p++;
            lc += 8;
            return true;
        }

        uint64_t getBits(int nBits) {
            while (lc < nBits) {
                if (!getChar()) return 0;
            }
            lc -= nBits;
            return (c >> lc) & ((1ull << nBits) - 1);
        }
    };

    // Assign canonical codes to the lengths just read: shorter codes first, each
    // length starting where the previous one left off.
    void hufCanonicalCodeTable(std::vector<uint64_t>& hcode) {

        uint64_t n[59]{};

        for (int i = 0; i < HUF_ENCSIZE; ++i) n[hcode[i]] += 1;

        uint64_t c = 0;
        for (int i = 58; i > 0; --i) {
            const uint64_t nc = (c + n[i]) >> 1;
            n[i] = c;
            c = nc;
        }

        for (int i = 0; i < HUF_ENCSIZE; ++i) {
            const int l = static_cast<int>(hcode[i]);
            if (l > 0) hcode[i] = static_cast<uint64_t>(l) | (n[l]++ << 6);
        }
    }

    // The code-length table itself is packed: 6 bits per symbol, with two escape
    // values standing in for runs of unused symbols.
    bool hufUnpackEncTable(BitReader& r, int im, int iM, std::vector<uint64_t>& hcode) {

        for (; im <= iM; im++) {

            if (!r.ok) return false;

            const uint64_t l = hcode[im] = r.getBits(6);

            if (l == LONG_ZEROCODE_RUN) {

                const int zerun = static_cast<int>(r.getBits(8)) + SHORTEST_LONG_RUN;
                if (im + zerun > iM + 1) return false;
                for (int i = 0; i < zerun; ++i) hcode[im++] = 0;
                im--;

            } else if (l >= SHORT_ZEROCODE_RUN) {

                const int zerun = static_cast<int>(l) - SHORT_ZEROCODE_RUN + 2;
                if (im + zerun > iM + 1) return false;
                for (int i = 0; i < zerun; ++i) hcode[im++] = 0;
                im--;
            }
        }

        if (!r.ok) return false;

        hufCanonicalCodeTable(hcode);
        return true;
    }

    // One entry per HUF_DECBITS-bit prefix. Codes that fit the prefix resolve in
    // a single lookup (len/lit); longer ones land in `p`, a short list of
    // candidate symbols the decoder then matches bit-exactly.
    struct HufDec {
        int len{0};
        int lit{0};
        std::vector<int> p;
    };

    bool hufBuildDecTable(const std::vector<uint64_t>& hcode, int im, int iM, std::vector<HufDec>& hdecod) {

        for (; im <= iM; im++) {

            const uint64_t c = hufCode(hcode[im]);
            const int l = static_cast<int>(hufLength(hcode[im]));

            if (l > 0 && (c >> l) != 0) return false;

            if (l > HUF_DECBITS) {

                const uint64_t index = c >> (l - HUF_DECBITS);
                if (index >= static_cast<uint64_t>(HUF_DECSIZE)) return false;
                HufDec& pl = hdecod[static_cast<size_t>(index)];
                if (pl.len) return false;
                pl.lit++;
                pl.p.push_back(im);

            } else if (l) {

                const uint64_t start = c << (HUF_DECBITS - l);
                const uint64_t count = 1ull << (HUF_DECBITS - l);
                if (start + count > static_cast<uint64_t>(HUF_DECSIZE)) return false;

                for (uint64_t i = 0; i < count; ++i) {
                    HufDec& pl = hdecod[static_cast<size_t>(start + i)];
                    if (pl.len || !pl.p.empty()) return false;
                    pl.len = l;
                    pl.lit = im;
                }
            }
        }
        return true;
    }

    // Emit one decoded symbol. `rlc` is the run-length escape: it is followed by
    // an 8-bit repeat count for the PREVIOUS symbol, which is how PIZ pays
    // almost nothing for the long runs of zero coefficients the wavelet leaves.
    bool getCode(int po, int rlc, BitReader& r, uint16_t*& out, uint16_t* ob, uint16_t* oe) {

        if (po == rlc) {

            if (r.lc < 8 && !r.getChar()) return false;
            r.lc -= 8;

            unsigned int cs = static_cast<unsigned char>(r.c >> r.lc);
            if (out + cs > oe) return false;
            if (out <= ob) return false;

            const uint16_t s = out[-1];
            while (cs-- > 0) *out++ = s;
            return true;
        }

        if (out < oe) {
            *out++ = static_cast<uint16_t>(po);
            return true;
        }
        return false;
    }

    bool hufDecode(const std::vector<uint64_t>& hcode, const std::vector<HufDec>& hdecod,
                   const unsigned char* in, int ni, int rlc, size_t no, uint16_t* out) {

        uint16_t* const ob = out;
        uint16_t* const oe = out + no;

        BitReader r{in, in + (ni + 7) / 8};

        while (r.p < r.end) {

            if (!r.getChar()) return false;

            while (r.lc >= HUF_DECBITS) {

                const HufDec& pl = hdecod[static_cast<size_t>((r.c >> (r.lc - HUF_DECBITS)) & HUF_DECMASK)];

                if (pl.len) {

                    r.lc -= pl.len;
                    if (!getCode(pl.lit, rlc, r, out, ob, oe)) return false;

                } else {

                    if (pl.p.empty()) return false;

                    int j = 0;
                    for (; j < pl.lit; j++) {

                        const int l = static_cast<int>(hufLength(hcode[pl.p[j]]));
                        while (r.lc < l && r.p < r.end) r.getChar();

                        if (r.lc >= l) {
                            if (hufCode(hcode[pl.p[j]]) == ((r.c >> (r.lc - l)) & ((1ull << l) - 1))) {
                                r.lc -= l;
                                if (!getCode(pl.p[j], rlc, r, out, ob, oe)) return false;
                                break;
                            }
                        }
                    }
                    if (j == pl.lit) return false;
                }
            }
        }

        // The final byte is padded; drop the bits that were never written.
        const int i = (8 - ni) & 7;
        r.c >>= i;
        r.lc -= i;

        while (r.lc > 0) {

            const HufDec& pl = hdecod[static_cast<size_t>((r.c << (HUF_DECBITS - r.lc)) & HUF_DECMASK)];
            if (!pl.len) return false;

            r.lc -= pl.len;
            if (!getCode(pl.lit, rlc, r, out, ob, oe)) return false;
        }

        return out - ob == static_cast<ptrdiff_t>(no);
    }

    bool hufUncompress(const unsigned char* in, size_t inSize, uint16_t* out, size_t no) {

        if (inSize == 0) return no == 0;
        if (inSize < 20) return false;

        const int im = static_cast<int>(readU32(in));
        const int iM = static_cast<int>(readU32(in + 4));
        // in + 8 is the packed table's length, which the bit reader's own bound
        // covers; in + 16 is reserved.
        const uint32_t nBits = readU32(in + 12);

        if (im < 0 || im >= HUF_ENCSIZE || iM < 0 || iM >= HUF_ENCSIZE || im > iM) return false;

        const unsigned char* ptr = in + 20;
        const size_t avail = inSize - 20;
        if ((nBits + 7ull) / 8 > avail) return false;

        std::vector<uint64_t> hcode(HUF_ENCSIZE, 0);
        BitReader table{ptr, in + inSize};
        if (!hufUnpackEncTable(table, im, iM, hcode)) return false;

        const size_t consumed = static_cast<size_t>(table.p - ptr);
        if (consumed > avail) return false;
        if ((nBits + 7ull) / 8 > avail - consumed) return false;

        std::vector<HufDec> hdecod(HUF_DECSIZE);
        if (!hufBuildDecTable(hcode, im, iM, hdecod)) return false;

        return hufDecode(hcode, hdecod, table.p, static_cast<int>(nBits), iM, no, out);
    }

}// namespace

bool threepp::detail::pizDecode(const unsigned char* in, std::size_t inSize,
                                const std::vector<int>& shortsPerSample,
                                int width, int lines,
                                unsigned char* out, std::size_t outSize) {

    if (width <= 0 || lines <= 0 || shortsPerSample.empty()) return false;

    size_t totalShorts = 0;
    for (int s : shortsPerSample) {
        totalShorts += static_cast<size_t>(width) * lines * s;
    }
    if (totalShorts * sizeof(uint16_t) != outSize) return false;

    // --- range compression bitmap ---
    if (inSize < 4) return false;
    const uint16_t minNonZero = readU16(in);
    const uint16_t maxNonZero = readU16(in + 2);
    size_t pos = 4;

    if (maxNonZero >= BITMAP_SIZE) return false;

    std::vector<unsigned char> bitmap(BITMAP_SIZE, 0);
    if (minNonZero <= maxNonZero) {
        const size_t n = static_cast<size_t>(maxNonZero) - minNonZero + 1;
        if (n > inSize - pos) return false;
        std::memcpy(bitmap.data() + minNonZero, in + pos, n);
        pos += n;
    }

    std::vector<uint16_t> lut(USHORT_RANGE);
    const uint16_t maxValue = reverseLutFromBitmap(bitmap.data(), lut.data());

    // --- huffman ---
    if (inSize - pos < 4) return false;
    const int32_t length = static_cast<int32_t>(readU32(in + pos));
    pos += 4;
    if (length < 0 || static_cast<size_t>(length) > inSize - pos) return false;

    std::vector<uint16_t> tmp(totalShorts);
    if (!hufUncompress(in + pos, static_cast<size_t>(length), tmp.data(), totalShorts)) {
        std::cerr << "[EXRLoader] corrupt PIZ chunk (huffman)" << std::endl;
        return false;
    }

    // --- wavelet, per channel, per 16-bit component of a sample ---
    size_t offset = 0;
    for (int s : shortsPerSample) {
        for (int j = 0; j < s; ++j) {
            wav2Decode(tmp.data() + offset + j, width, s, lines, width * s, maxValue);
        }
        offset += static_cast<size_t>(width) * lines * s;
    }

    // --- undo the range compression ---
    for (auto& v : tmp) v = lut[v];

    // --- channel-planar to scanline-interleaved ---
    std::vector<size_t> cursor(shortsPerSample.size());
    size_t start = 0;
    for (size_t c = 0; c < shortsPerSample.size(); ++c) {
        cursor[c] = start;
        start += static_cast<size_t>(width) * lines * shortsPerSample[c];
    }

    unsigned char* o = out;
    for (int y = 0; y < lines; ++y) {
        for (size_t c = 0; c < shortsPerSample.size(); ++c) {
            const size_t n = static_cast<size_t>(width) * shortsPerSample[c];
            std::memcpy(o, tmp.data() + cursor[c], n * sizeof(uint16_t));
            o += n * sizeof(uint16_t);
            cursor[c] += n;
        }
    }

    return true;
}
