
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/constants.hpp"

TEST_CASE("constants") {

    REQUIRE(threepp::CullFaceNone == 0);
    REQUIRE(threepp::CullFaceBack == 1);
    REQUIRE(threepp::CullFaceFront == 2);
    REQUIRE(threepp::CullFaceFrontBack == 3);

    REQUIRE(threepp::NoBlending == 0);
    REQUIRE(threepp::NormalBlending == 1);
    REQUIRE(threepp::AdditiveBlending == 2);
    REQUIRE(threepp::SubtractiveBlending == 3);
    REQUIRE(threepp::MultiplyBlending == 4);
    REQUIRE(threepp::CustomBlending == 5);
    REQUIRE(threepp::AddEquation == 100);
    REQUIRE(threepp::SubtractEquation == 101);
    REQUIRE(threepp::ReverseSubtractEquation == 102);
    REQUIRE(threepp::MinEquation == 103);
    REQUIRE(threepp::MaxEquation == 104);
    REQUIRE(threepp::ZeroFactor == 200);
    REQUIRE(threepp::OneFactor == 201);
    REQUIRE(threepp::SrcColorFactor == 202);
    REQUIRE(threepp::OneMinusSrcColorFactor == 203);
    REQUIRE(threepp::SrcAlphaFactor == 204);
    REQUIRE(threepp::OneMinusSrcAlphaFactor == 205);
    REQUIRE(threepp::DstAlphaFactor == 206);
    REQUIRE(threepp::OneMinusDstAlphaFactor == 207);
    REQUIRE(threepp::DstColorFactor == 208);
    REQUIRE(threepp::OneMinusDstColorFactor == 209);
    REQUIRE(threepp::SrcAlphaSaturateFactor == 210);
    REQUIRE(threepp::NeverDepth == 0);
    REQUIRE(threepp::AlwaysDepth == 1);
    REQUIRE(threepp::LessDepth == 2);
    REQUIRE(threepp::LessEqualDepth == 3);
    REQUIRE(threepp::EqualDepth == 4);
    REQUIRE(threepp::GreaterEqualDepth == 5);
    REQUIRE(threepp::GreaterDepth == 6);
    REQUIRE(threepp::NotEqualDepth == 7);
    REQUIRE(threepp::MultiplyOperation == 0);
    REQUIRE(threepp::MixOperation == 1);
    REQUIRE(threepp::AddOperation == 2);

    REQUIRE(threepp::UVMapping == 300);
    REQUIRE(threepp::CubeReflectionMapping == 301);
    REQUIRE(threepp::CubeRefractionMapping == 302);
    REQUIRE(threepp::EquirectangularReflectionMapping == 303);
    REQUIRE(threepp::EquirectangularRefractionMapping == 304);
    REQUIRE(threepp::CubeUVReflectionMapping == 306);

    REQUIRE(threepp::NearestMipMapNearestFilter == 1004);
    REQUIRE(threepp::NearestMipMapLinearFilter == 1005);
    REQUIRE(threepp::LinearFilter == 1006);
    REQUIRE(threepp::LinearMipMapNearestFilter == 1007);
    REQUIRE(threepp::LinearMipMapLinearFilter == 1008);
    REQUIRE(threepp::UnsignedByteType == 1009);
    REQUIRE(threepp::ByteType == 1010);
    REQUIRE(threepp::ShortType == 1011);
    REQUIRE(threepp::UnsignedShortType == 1012);
    REQUIRE(threepp::IntType == 1013);
    REQUIRE(threepp::UnsignedIntType == 1014);
    REQUIRE(threepp::FloatType == 1015);
    REQUIRE(threepp::HalfFloatType == 1016);
    REQUIRE(threepp::UnsignedShort4444Type == 1017);
    REQUIRE(threepp::UnsignedShort5551Type == 1018);
    REQUIRE(threepp::UnsignedInt248Type == 1020);
    REQUIRE(threepp::AlphaFormat == 1021);
    REQUIRE(threepp::RGBAFormat == 1023);
    REQUIRE(threepp::LuminanceFormat == 1024);
    REQUIRE(threepp::LuminanceAlphaFormat == 1025);
    REQUIRE(threepp::DepthFormat == 1026);
    REQUIRE(threepp::DepthStencilFormat == 1027);

    REQUIRE(threepp::LoopOnce == 2200);
    REQUIRE(threepp::LoopRepeat == 2201);
    REQUIRE(threepp::LoopPingPong == 2202);
    REQUIRE(threepp::InterpolateDiscrete == 2300);
    REQUIRE(threepp::InterpolateLinear == 2301);
    REQUIRE(threepp::InterpolateSmooth == 2302);
    REQUIRE(threepp::ZeroCurvatureEnding == 2400);
    REQUIRE(threepp::ZeroSlopeEnding == 2401);
    REQUIRE(threepp::WrapAroundEnding == 2402);
    REQUIRE(threepp::TrianglesDrawMode == 0);
    REQUIRE(threepp::TriangleStripDrawMode == 1);
    REQUIRE(threepp::TriangleFanDrawMode == 2);

    REQUIRE(threepp::BasicDepthPacking == 3200);
    REQUIRE(threepp::RGBADepthPacking == 3201);
}
