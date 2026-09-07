// The SH basis: is it the real orthonormal one, does it still agree with
// threepp's own SphericalHarmonis3, and does the shader still use the same
// numbers as the C++ table?
//
// The orthonormality check is the load-bearing one. It validates all sixteen
// constants — including band 3, which SphericalHarmonis3 has nothing to say
// about — against the definition of the basis rather than against any
// implementation, which is the only kind of check available when the reference
// implementation is one you are not allowed to read.

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/SphericalHarmonics3.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatSH.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Approx;

namespace {

    // Fibonacci sphere: a low-discrepancy, deterministic set of directions.
    std::vector<Vector3> sphereSamples(int n) {

        std::vector<Vector3> out;
        out.reserve(static_cast<size_t>(n));

        const float golden = math::PI * (3.f - std::sqrt(5.f));
        for (int i = 0; i < n; ++i) {

            const float y = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
            const float r = std::sqrt(std::max(0.f, 1.f - y * y));
            const float theta = golden * static_cast<float>(i);
            out.emplace_back(std::cos(theta) * r, y, std::sin(theta) * r);
        }

        return out;
    }

}// namespace


TEST_CASE("SplatSH: the basis is orthonormal over the sphere") {

    // integral over the sphere of Y_i * Y_j == delta_ij. Averaging over uniform
    // directions and multiplying by the sphere's area 4*pi approximates it.
    const auto dirs = sphereSamples(40000);

    constexpr int N = 16;
    double gram[N][N] = {};

    for (const auto& d : dirs) {

        float basis[N];
        splats::shBasis(d, splats::MAX_SH_DEGREE, basis);

        for (int i = 0; i < N; ++i) {

            for (int j = 0; j < N; ++j) gram[i][j] += static_cast<double>(basis[i]) * basis[j];
        }
    }

    const double weight = 4.0 * 3.14159265358979323846 / static_cast<double>(dirs.size());

    for (int i = 0; i < N; ++i) {

        for (int j = 0; j < N; ++j) {

            INFO("Gram(" << i << ", " << j << ")");
            CHECK(gram[i][j] * weight == Approx(i == j ? 1.0 : 0.0).margin(2e-3));
        }
    }
}

TEST_CASE("SplatSH: bands 0-2 match threepp's SphericalHarmonis3, up to sign") {

    // SphericalHarmonis3 writes the same constants with every sign positive,
    // and to six decimals. Splat coefficients are trained against the
    // (-1)^|m| convention, so the two must agree in magnitude and are expected
    // to differ in sign on the odd-|m| terms. This pins both halves of that:
    // any magnitude drift fails, and the sign pattern is asserted explicitly
    // rather than left as folklore.
    SphericalHarmonis3 reference;

    const auto dirs = sphereSamples(97);

    for (const auto& dir : dirs) {

        float mine[16];
        splats::shBasis(dir, 2, mine);

        for (int i = 0; i < 9; ++i) {

            // Isolate basis function i by setting only coefficient i.
            std::vector<Vector3> coeffs(9, Vector3{});
            coeffs[i].set(1.f, 0.f, 0.f);
            reference.set(coeffs);

            Vector3 out;
            reference.getAt(dir, out);

            INFO("basis " << i << " at " << dir);
            CHECK(std::abs(out.x) == Approx(std::abs(mine[i])).margin(2e-6f));
        }
    }

    // The sign pattern itself: + for even |m|, - for odd.
    const Vector3 probe = Vector3{0.3f, 0.5f, 0.81f}.normalize();
    float mine[16];
    splats::shBasis(probe, 2, mine);

    const int oddM[] = {1, 3, 5, 7};// l=1 m=+-1, l=2 m=+-1
    const int evenM[] = {0, 2, 4, 6, 8};

    for (int i : oddM) {

        std::vector<Vector3> coeffs(9, Vector3{});
        coeffs[i].set(1.f, 0.f, 0.f);
        reference.set(coeffs);
        Vector3 out;
        reference.getAt(probe, out);

        INFO("odd-|m| basis " << i);
        CHECK(out.x * mine[i] < 0.f);// opposite signs
    }

    for (int i : evenM) {

        std::vector<Vector3> coeffs(9, Vector3{});
        coeffs[i].set(1.f, 0.f, 0.f);
        reference.set(coeffs);
        Vector3 out;
        reference.getAt(probe, out);

        INFO("even-|m| basis " << i);
        CHECK(out.x * mine[i] > 0.f);// same sign
    }
}

TEST_CASE("SplatSH: the shader carries the same constants as the C++ table") {

    // One source of truth, enforced textually: the GLSL is a string literal,
    // so the only thing that can be checked from C++ is that the decimal
    // representations still appear in it. That is enough — a change to either
    // side that does not change the other fails here.
    // Whitespace-stripped, so the assertions below are about the maths and not
    // about how the file happens to be formatted.
    std::string vs;
    for (char ch : SplatCloud::vertexShaderSource()) {

        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') vs += ch;
    }

    const std::vector<std::string> constants{
            "0.28209479177387814", // SH_C0
            "0.4886025119029199",  // SH_C1
            "1.0925484305920792",  // SH_C2 |m|=2 and |m|=1
            "0.31539156525252005", // SH_C2 m=0
            "0.5462742152960396",  // SH_C2 m=+2
            "0.5900435899266435",  // SH_C3 |m|=3
            "2.890611442640554",   // SH_C3 |m|=2 (the xyz one)
            "0.4570457994644658",  // SH_C3 |m|=1
            "0.3731763325901154",  // SH_C3 m=0
            "1.445305721320277"};  // SH_C3 m=+2

    for (const auto& c : constants) {

        INFO("shader is missing the constant " << c);
        CHECK(vs.find(c) != std::string::npos);
    }

    // ... and the negations, which is how the (-1)^|m| convention shows up in
    // the GLSL. If someone "cleans up" the signs, this fails.
    for (const auto& c : {"-1.0925484305920792", "-0.5900435899266435",
                          "-0.4570457994644658",
                          "-SH_C1*y", "+SH_C1*z", "-SH_C1*x"}) {

        INFO("shader is missing " << c);
        CHECK(vs.find(c) != std::string::npos);
    }

    // The +0.5 evaluation offset lives on both sides too.
    CHECK(vs.find("c+0.5") != std::string::npos);
    CHECK(splats::SH_COLOR_OFFSET == Approx(0.5f));
}

TEST_CASE("SplatSH: a DC-only splat renders as the colour it was given") {

    const Vector3 wanted{0.8f, 0.25f, 0.1f};
    const auto dc = splats::dcFromColor(wanted);

    float coeffs[16 * 3] = {};
    coeffs[0] = dc.x;
    coeffs[1] = dc.y;
    coeffs[2] = dc.z;

    // Degree 0, and degree 3 with every higher band zero: same answer, from
    // every direction.
    for (int degree : {0, 3}) {

        for (const auto& dir : sphereSamples(31)) {

            const auto rgb = splats::evalSh(coeffs, degree, dir);
            INFO("degree " << degree << " dir " << dir);
            CHECK(rgb.x == Approx(wanted.x).margin(1e-6f));
            CHECK(rgb.y == Approx(wanted.y).margin(1e-6f));
            CHECK(rgb.z == Approx(wanted.z).margin(1e-6f));
        }
    }
}

TEST_CASE("SplatSH: higher bands make the colour view-dependent, and it never goes negative") {

    float coeffs[16 * 3] = {};
    const auto dc = splats::dcFromColor(Vector3{0.5f, 0.5f, 0.5f});
    coeffs[0] = dc.x;
    coeffs[1] = dc.y;
    coeffs[2] = dc.z;
    coeffs[1 * 3 + 0] = 3.f;// a strong band-1 term on red

    const auto a = splats::evalSh(coeffs, 1, Vector3{0, 1, 0});
    const auto b = splats::evalSh(coeffs, 1, Vector3{0, -1, 0});

    CHECK(a.x != Approx(b.x));
    CHECK(std::min(a.x, b.x) >= 0.f);// clamped, never negative
    CHECK(a.y == Approx(b.y));       // green untouched
}

TEST_CASE("SplatSH: coefficient counts follow (degree+1)^2") {

    CHECK(splats::shCoeffCount(0) == 1);
    CHECK(splats::shCoeffCount(1) == 4);
    CHECK(splats::shCoeffCount(2) == 9);
    CHECK(splats::shCoeffCount(3) == 16);
}
