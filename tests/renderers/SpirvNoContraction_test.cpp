// SpirvNoContraction on a hand-built module and on the backend's own embedded
// shaders. CPU only.

#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/vulkan/SpirvNoContraction.hpp"
#include "threepp/renderers/vulkan/SpirvReflect.hpp"

#include "threepp/renderers/vulkan/shaders/bloom_down.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/deferred_shade.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer.vert.spv.h"

#include <set>

using namespace threepp::vulkan;

namespace {

    constexpr uint32_t op(uint32_t wordCount, uint32_t opcode) { return (wordCount << 16) | opcode; }

    constexpr uint32_t kNoContraction = 42;

    // A compute shader with float scalar, vector and matrix arithmetic, one
    // integer add, and one float result the source already declared precise.
    //   %13 = a * b, %14 = %13 + c, %15 = %14 - c (precise), %16 = int add,
    //   %17 = matrix * vector, %18 = dot, %19 = -%17
    const std::vector<uint32_t> kModule = {
            0x07230203, 0x00010600, 0, 20, 0,
            op(2, 17), 1,                       // OpCapability Shader
            op(3, 14), 0, 1,                    // OpMemoryModel Logical GLSL450
            op(5, 15), 5, 11, 0x6e69616d, 0,    // OpEntryPoint GLCompute %11 "main"
            op(6, 16), 11, 17, 1, 1, 1,         // OpExecutionMode %11 LocalSize 1 1 1
            op(3, 71), 15, kNoContraction,      // OpDecorate %15 NoContraction
            op(2, 19), 1,                       // %1 = OpTypeVoid
            op(3, 33), 2, 1,                    // %2 = OpTypeFunction %1
            op(3, 22), 3, 32,                   // %3 = OpTypeFloat 32
            op(4, 21), 4, 32, 1,                // %4 = OpTypeInt 32 1
            op(4, 23), 5, 3, 4,                 // %5 = OpTypeVector %3 4
            op(4, 24), 6, 5, 4,                 // %6 = OpTypeMatrix %5 4
            op(4, 43), 3, 7, 0x3f800000,        // %7 = OpConstant %3 1.0
            op(4, 43), 4, 8, 2,                 // %8 = OpConstant %4 2
            op(7, 44), 5, 9, 7, 7, 7, 7,        // %9 = OpConstantComposite %5
            op(7, 44), 6, 10, 9, 9, 9, 9,       // %10 = OpConstantComposite %6
            op(5, 54), 1, 11, 0, 2,             // %11 = OpFunction %1 None %2
            op(2, 248), 12,                     // %12 = OpLabel
            op(5, 133), 3, 13, 7, 7,            // %13 = OpFMul %3 %7 %7
            op(5, 129), 3, 14, 13, 7,           // %14 = OpFAdd %3 %13 %7
            op(5, 131), 3, 15, 14, 7,           // %15 = OpFSub %3 %14 %7
            op(5, 128), 4, 16, 8, 8,            // %16 = OpIAdd %4 %8 %8
            op(5, 145), 5, 17, 10, 9,           // %17 = OpMatrixTimesVector %5 %10 %9
            op(5, 148), 3, 18, 9, 9,            // %18 = OpDot %3 %9 %9
            op(4, 127), 5, 19, 17,              // %19 = OpFNegate %5 %17
            op(1, 253),                         // OpReturn
            op(1, 56),                          // OpFunctionEnd
    };

    bool isPreamble(uint32_t opcode) {
        return (opcode >= 2 && opcode <= 7) || opcode == 10 || opcode == 11 || (opcode >= 14 && opcode <= 17) ||
               (opcode >= 71 && opcode <= 75) || opcode == 330 || opcode == 331 || opcode == 332 ||
               opcode == 5632 || opcode == 5633;
    }

    struct Parsed {
        bool wellFormed = false;
        std::multiset<uint32_t> noContraction;// ids decorated NoContraction (repeats kept)
        std::set<uint32_t> floatResults;      // results of float arithmetic
        size_t typesStart = 0;                // word index of the first non-preamble instruction
        size_t lastAnnotationEnd = 0;         // word index past the last OpDecorate
    };

    // Walks a module the way a consumer would: every instruction's word count
    // must land exactly on the end, ids must be under the bound, and every
    // decoration must sit before the first type declaration.
    Parsed parse(const std::vector<uint32_t>& w) {
        Parsed p;
        if (w.size() < 5 || w[0] != 0x07230203) return p;
        const uint32_t bound = w[3];
        std::set<uint32_t> floatTypes;
        bool inPreamble = true;
        size_t i = 5;
        while (i < w.size()) {
            const uint32_t wc = w[i] >> 16, opcode = w[i] & 0xFFFFu;
            if (wc == 0 || i + wc > w.size()) return p;
            if (inPreamble && !isPreamble(opcode)) {
                inPreamble = false;
                p.typesStart = i;
            }
            if (opcode == 71) {
                if (!inPreamble || w[i + 1] >= bound) return p;
                if (w[i + 2] == kNoContraction) p.noContraction.insert(w[i + 1]);
                p.lastAnnotationEnd = i + wc;
            } else if (opcode == 22) {
                floatTypes.insert(w[i + 1]);
            } else if ((opcode == 23 || opcode == 24) && floatTypes.count(w[i + 2])) {
                floatTypes.insert(w[i + 1]);
            } else if ((opcode == 127 || opcode == 129 || opcode == 131 || opcode == 133 || opcode == 136 ||
                        (opcode >= 140 && opcode <= 146) || opcode == 148) &&
                       floatTypes.count(w[i + 1])) {
                if (w[i + 2] >= bound) return p;
                p.floatResults.insert(w[i + 2]);
            }
            i += wc;
        }
        p.wellFormed = i == w.size();
        return p;
    }

    void checkDecorated(const std::vector<uint32_t>& in, const std::vector<uint32_t>& out,
                        const NoContractionStats& st) {
        REQUIRE(out.size() == in.size() + 3 * st.added);
        for (int k = 0; k < 5; ++k) CHECK(out[k] == in[k]);// header, bound included

        const auto before = parse(in);
        const auto after = parse(out);
        REQUIRE(before.wellFormed);
        REQUIRE(after.wellFormed);
        CHECK(after.lastAnnotationEnd <= after.typesStart);
        CHECK(after.floatResults == before.floatResults);
        CHECK(st.floatResults == before.floatResults.size());
        CHECK(st.alreadyDecorated == before.noContraction.size());
        // Every float arithmetic result is decorated, exactly once.
        for (const uint32_t id : after.floatResults) CHECK(after.noContraction.count(id) == 1);
        // Nothing else gained the decoration.
        CHECK(after.noContraction.size() == before.noContraction.size() + st.added);

        // Applying it again finds nothing to add.
        NoContractionStats again;
        CHECK(addNoContraction(out.data(), out.size(), &again) == out);
        CHECK(again.added == 0);
    }

}// namespace

TEST_CASE("hand-built module: float results gain NoContraction, integer and precise ones do not") {
    NoContractionStats st;
    const auto out = addNoContraction(kModule.data(), kModule.size(), &st);
    CHECK(st.floatResults == 6);
    CHECK(st.alreadyDecorated == 1);
    CHECK(st.added == 5);
    checkDecorated(kModule, out, st);

    const auto after = parse(out);
    CHECK(after.noContraction == std::multiset<uint32_t>{13, 14, 15, 17, 18, 19});
    CHECK(after.noContraction.count(16) == 0);// OpIAdd
    // The new decorations follow the existing one, before %1 = OpTypeVoid.
    CHECK(after.typesStart == 24 + 15);
}

TEST_CASE("not SPIR-V, or truncated: returned unchanged") {
    const std::vector<uint32_t> notSpirv = {1, 2, 3, 4, 5, 6};
    CHECK(addNoContraction(notSpirv.data(), notSpirv.size()) == notSpirv);
    const std::vector<uint32_t> truncated(kModule.begin(), kModule.end() - 3);
    NoContractionStats st;
    CHECK(addNoContraction(truncated.data(), truncated.size(), &st) == truncated);
    CHECK(st.added == 0);
}

TEST_CASE("embedded shaders: decorated, still well-formed, same bindings") {
    const std::vector<std::vector<uint32_t>> modules = {
            {std::begin(kBloomDownCompSpv), std::end(kBloomDownCompSpv)},
            {std::begin(kDeferredShadeCompSpv), std::end(kDeferredShadeCompSpv)},
            {std::begin(kGbufferVertSpv), std::end(kGbufferVertSpv)},
    };
    for (const auto& in : modules) {
        NoContractionStats st;
        const auto out = addNoContraction(in.data(), in.size(), &st);
        CHECK(st.added > 0);
        checkDecorated(in, out, st);

        const auto a = reflectSpirvBindings(in.data(), in.size());
        const auto b = reflectSpirvBindings(out.data(), out.size());
        REQUIRE(a.size() == b.size());
        for (size_t k = 0; k < a.size(); ++k) {
            CHECK(a[k].set == b[k].set);
            CHECK(a[k].binding == b[k].binding);
            CHECK(a[k].kind == b[k].kind);
        }
    }
}
