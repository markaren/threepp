
#include "threepp/utils/Base64.hpp"

#include <array>

namespace {

    constexpr std::array<int, 256> buildDecodeTable() {
        std::array<int, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 26; ++i) {
            t['A' + i] = i;
            t['a' + i] = i + 26;
        }
        for (int i = 0; i < 10; ++i) t['0' + i] = i + 52;
        t['+'] = 62;
        t['/'] = 63;
        t['='] = 0;
        return t;
    }

    constexpr auto kDecodeTable = buildDecodeTable();

    constexpr char kEncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

}// namespace

std::vector<uint8_t> threepp::utils::base64Decode(const std::string& encoded) {

    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : encoded) {
        if (c == '\n' || c == '\r' || c == ' ') continue;
        int d = kDecodeTable[c];
        if (d == -1) break;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::string threepp::utils::base64Encode(const uint8_t* data, size_t size) {

    std::string out;
    out.reserve((size + 2) / 3 * 4);

    size_t i = 0;
    for (; i + 2 < size; i += 3) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(kEncodeTable[(n >> 18) & 0x3F]);
        out.push_back(kEncodeTable[(n >> 12) & 0x3F]);
        out.push_back(kEncodeTable[(n >> 6) & 0x3F]);
        out.push_back(kEncodeTable[n & 0x3F]);
    }

    const size_t rem = size - i;
    if (rem == 1) {
        const uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(kEncodeTable[(n >> 18) & 0x3F]);
        out.push_back(kEncodeTable[(n >> 12) & 0x3F]);
        out.append("==");
    } else if (rem == 2) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(kEncodeTable[(n >> 18) & 0x3F]);
        out.push_back(kEncodeTable[(n >> 12) & 0x3F]);
        out.push_back(kEncodeTable[(n >> 6) & 0x3F]);
        out.push_back('=');
    }

    return out;
}
