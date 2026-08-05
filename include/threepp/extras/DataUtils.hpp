// Half-float conversion, ported from three.js r129 src/extras/DataUtils.js
// (three.js commit d39d82999f). The call sites read the same way:
//
//     const uint16_t h = DataUtils::toHalfFloat(value);   // JS: DataUtils.toHalfFloat(value)
//
// The algorithm itself is the one r129 credits to
// http://gamedev.stackexchange.com/questions/17326/ — faster than the OpenEXR
// implementation and, unlike a plain truncation, it rounds. Sign, zero,
// underflow-to-zero, the denormal range and the round-to-nearest tie handling
// are ported unchanged, so a value that survives the trip in three.js survives
// it here bit for bit.
//
// ONE DELIBERATE DEVIATION, in the exponent-overflow branch. r129 writes
//
//     bits |= ( ( e == 255 ) ? 0 : 1 ) && ( x & 0x007fffff );
//
// which in JavaScript evaluates to `x & 0x007fffff` whenever e != 255 — up to
// 23 bits ORed into what is about to be stored as a 16-bit value. It does not
// do what the comment above it in r129 says it does, and the results are not
// merely imprecise: 70000.0f comes out as NEGATIVE infinity (the mantissa
// carries a 1 into what becomes the sign bit after truncation), and 1e20f
// comes out as a NaN. The line reads like a transcription slip of the C
// original, where `&&` yields 0 or 1 rather than its right operand.
//
// This port implements what that comment describes, which is also what every
// later three.js does (current three.js sidesteps the whole branch by clamping
// the input to ±65504 first):
//
//     NaN in   -> NaN out          (a mantissa bit is kept, so it stays NaN)
//     Inf in   -> Inf out
//     overflow -> Inf, signed      (|value| >= 65520)
//
// Everything below 65504 is untouched by the deviation and matches r129
// exactly. fromHalfFloat has no counterpart in r129 at all; it is an
// extension, added because a conversion you cannot invert is a conversion you
// cannot test.

#ifndef THREEPP_DATAUTILS_HPP
#define THREEPP_DATAUTILS_HPP

#include <cstdint>
#include <cstring>

namespace threepp {

    namespace DataUtils {

        // float32 -> float16, returned as the raw 16 bits.
        [[nodiscard]] inline std::uint16_t toHalfFloat(float val) {

            std::uint32_t u;
            std::memcpy(&u, &val, sizeof(u));
            const auto x = static_cast<std::int32_t>(u);

            std::int32_t bits = (x >> 16) & 0x8000; /* Get the sign */
            std::int32_t m = (x >> 12) & 0x07ff;    /* Keep one extra bit for rounding */
            const std::int32_t e = (x >> 23) & 0xff;/* Using int is faster here */

            /* If zero, or denormal, or exponent underflows too much for a denormal
             * half, return signed zero. */
            if (e < 103) return static_cast<std::uint16_t>(bits);

            /* If NaN, return NaN. If Inf or exponent overflow, return Inf. */
            if (e > 142) {

                bits |= 0x7c00;
                /* If exponent was 0xff and one mantissa bit was set, it means NaN,
                 * not Inf, so make sure we set one mantissa bit too. */
                if (e == 255 && (x & 0x007fffff)) bits |= 1;
                return static_cast<std::uint16_t>(bits);
            }

            /* If exponent underflows but not too much, return a denormal */
            if (e < 113) {

                m |= 0x0800;
                /* Extra rounding may overflow and set mantissa to 0 and exponent
                 * to 1, which is OK. */
                bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
                return static_cast<std::uint16_t>(bits);
            }

            bits |= ((e - 112) << 10) | (m >> 1);
            /* Extra rounding. An overflow will set mantissa to 0 and increment
             * the exponent, which is OK. */
            bits += m & 1;
            return static_cast<std::uint16_t>(bits);
        }

        // float16 -> float32. Not in r129 (nor in any three.js of that era):
        // an extension, and the only way to state what toHalfFloat promises as
        // a testable round trip. Exact — every half is representable as a
        // float — so this direction never rounds.
        [[nodiscard]] inline float fromHalfFloat(std::uint16_t half) {

            const std::uint32_t sign = (half & 0x8000u) << 16;
            const std::uint32_t exp = (half >> 10) & 0x1fu;
            const std::uint32_t mant = half & 0x03ffu;

            std::uint32_t bits;

            if (exp == 0) {

                if (mant == 0) {

                    bits = sign;// signed zero

                } else {

                    // Denormal half, normal float: shift the mantissa up until
                    // the implicit leading 1 appears, and pay for it in the
                    // exponent.
                    std::uint32_t m = mant;
                    std::uint32_t e = 127 - 15 + 1;
                    while ((m & 0x0400u) == 0) {

                        m <<= 1;
                        --e;
                    }
                    m &= 0x03ffu;
                    bits = sign | (e << 23) | (m << 13);
                }

            } else if (exp == 0x1f) {

                bits = sign | 0x7f800000u | (mant << 13);// Inf / NaN

            } else {

                bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            }

            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }

    }// namespace DataUtils

}// namespace threepp

#endif//THREEPP_DATAUTILS_HPP
