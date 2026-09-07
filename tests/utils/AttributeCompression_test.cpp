
#include <catch2/catch_test_macros.hpp>

#include "threepp/core/AttributeView.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    std::shared_ptr<BufferGeometry> makeQuad() {

        auto geometry = BufferGeometry::create();

        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   std::vector<float>{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(
                                                 std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1}, 3));
        geometry->setAttribute("uv", FloatBufferAttribute::create(
                                             std::vector<float>{0, 0, 1, 0, 1, 1, 0, 1}, 2));
        geometry->setAttribute("color", FloatBufferAttribute::create(
                                                std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, 3));
        return geometry;
    }

}// namespace

TEST_CASE("compressAttributes narrows normal, uv and color but never position") {

    auto geometry = makeQuad();

    const size_t saved = compressAttributes(*geometry);

    // normal 48->24, uv 32->16, color 48->12
    CHECK(saved == (48 - 24) + (32 - 16) + (48 - 12));

    CHECK(geometry->getAttribute("normal")->type() == AttributeType::Int16);
    CHECK(geometry->getAttribute("uv")->type() == AttributeType::UInt16);
    CHECK(geometry->getAttribute("color")->type() == AttributeType::UInt8);

    // Positions keep their dynamic range.
    CHECK(geometry->getAttribute("position")->type() == AttributeType::Float);

    // itemSize and count survive the conversion.
    CHECK(geometry->getAttribute("normal")->itemSize() == 3);
    CHECK(geometry->getAttribute("normal")->count() == 4);
    CHECK(geometry->getAttribute("normal")->normalized());
}

TEST_CASE("compressed values round-trip back through FloatAttributeView") {

    auto geometry = makeQuad();
    compressAttributes(*geometry);

    FloatAttributeView normals(geometry->getAttribute("normal"));
    FloatAttributeView uvs(geometry->getAttribute("uv"));
    FloatAttributeView colors(geometry->getAttribute("color"));

    REQUIRE(normals.size() == 12);
    REQUIRE(uvs.size() == 8);
    REQUIRE(colors.size() == 12);

    // int16 snorm resolves ~3e-5; uint16 unorm ~1.5e-5; uint8 unorm ~4e-3.
    for (size_t i = 0; i < normals.size(); ++i) {
        const float expected = (i % 3 == 2) ? 1.f : 0.f;
        CHECK(std::abs(normals[i] - expected) < 1e-4f);
    }
    CHECK(std::abs(uvs[2] - 1.f) < 1e-4f);
    CHECK(std::abs(uvs[3] - 0.f) < 1e-4f);
    CHECK(std::abs(colors[0] - 1.f) < 5e-3f);
    CHECK(std::abs(colors[1] - 0.f) < 5e-3f);
}

// Tiled/atlas UVs run outside [0,1]; unorm16 would clamp them onto the texture
// edge and visibly break the mapping, so those must be left as float.
TEST_CASE("compressAttributes refuses UVs outside the unit range") {

    auto geometry = makeQuad();
    geometry->setAttribute("uv", FloatBufferAttribute::create(
                                         std::vector<float>{0, 0, 4, 0, 4, 4, 0, 4}, 2));

    compressAttributes(*geometry);

    CHECK(geometry->getAttribute("uv")->type() == AttributeType::Float);
    // The others still compress.
    CHECK(geometry->getAttribute("normal")->type() == AttributeType::Int16);
}

TEST_CASE("compressAttributes honours the per-attribute opt-outs") {

    auto geometry = makeQuad();

    AttributeCompression what;
    what.normal = false;
    what.color = false;

    compressAttributes(*geometry, what);

    CHECK(geometry->getAttribute("normal")->type() == AttributeType::Float);
    CHECK(geometry->getAttribute("color")->type() == AttributeType::Float);
    CHECK(geometry->getAttribute("uv")->type() == AttributeType::UInt16);
}

TEST_CASE("compressAttributes is idempotent") {

    auto geometry = makeQuad();

    const size_t first = compressAttributes(*geometry);
    const size_t second = compressAttributes(*geometry);

    CHECK(first > 0);
    CHECK(second == 0);
    CHECK(geometry->getAttribute("normal")->type() == AttributeType::Int16);
}

// BufferGeometry::copy used to throw runtime_error("TODO") for anything that was
// neither float nor unsigned int; a compressed geometry is exactly that case.
TEST_CASE("a compressed geometry can still be cloned") {

    auto geometry = makeQuad();
    compressAttributes(*geometry);

    auto clone = geometry->clone();

    REQUIRE(clone != nullptr);
    CHECK(clone->getAttribute("normal")->type() == AttributeType::Int16);
    CHECK(clone->getAttribute("uv")->type() == AttributeType::UInt16);
    CHECK(clone->getAttribute("color")->type() == AttributeType::UInt8);
    CHECK(clone->getAttribute("normal")->count() == 4);
}

TEST_CASE("mergeBufferGeometries keeps narrow attributes narrow") {

    auto a = makeQuad();
    auto b = makeQuad();
    compressAttributes(*a);
    compressAttributes(*b);

    auto merged = mergeBufferGeometries(std::vector<BufferGeometry*>{a.get(), b.get()});

    REQUIRE(merged != nullptr);
    CHECK(merged->getAttribute("position")->type() == AttributeType::Float);
    CHECK(merged->getAttribute("normal")->type() == AttributeType::Int16);
    CHECK(merged->getAttribute("uv")->type() == AttributeType::UInt16);
    CHECK(merged->getAttribute("color")->type() == AttributeType::UInt8);
    CHECK(merged->getAttribute("normal")->count() == 8);
    CHECK(merged->getAttribute("normal")->normalized());
}

TEST_CASE("mergeBufferGeometries refuses mixed scalar types per attribute") {

    auto a = makeQuad();
    auto b = makeQuad();
    compressAttributes(*a);// a narrow, b float

    auto merged = mergeBufferGeometries(std::vector<BufferGeometry*>{a.get(), b.get()});

    CHECK(merged == nullptr);
}

TEST_CASE("mergeVertices widens narrow attributes instead of failing") {

    auto geometry = makeQuad();
    compressAttributes(*geometry);

    auto welded = mergeVertices(*geometry);

    REQUIRE(welded != nullptr);
    // No duplicate vertices in the quad, so the count survives.
    CHECK(welded->getAttribute("position")->count() == 4);

    // Narrow sources come back as float with the values denormalized and the
    // normalized flag cleared.
    const auto* normal = welded->getAttribute("normal");
    REQUIRE(normal);
    CHECK(normal->type() == AttributeType::Float);
    CHECK_FALSE(normal->normalized());

    FloatAttributeView view(normal);
    CHECK(std::abs(view[2] - 1.f) < 1e-4f);// z component of the +Z normal
    CHECK(std::abs(view[0]) < 1e-4f);
}
