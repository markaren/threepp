
#include <catch2/catch_test_macros.hpp>

#include "threepp/core/AttributeView.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/math/Matrix4.hpp"

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

using namespace threepp;

TEST_CASE("BufferAttribute reports count from array size and itemSize") {

    auto attr = FloatBufferAttribute::create(std::vector<float>{0, 1, 2, 3, 4, 5}, 3);

    CHECK(attr->itemSize() == 3);
    CHECK(attr->count() == 2);
}

TEST_CASE("BufferAttribute accessors address elements by itemSize") {

    auto attr = FloatBufferAttribute::create(std::vector<float>{1, 2, 3, 4, 5, 6}, 3);

    CHECK(attr->getX(0) == 1);
    CHECK(attr->getY(0) == 2);
    CHECK(attr->getZ(0) == 3);
    CHECK(attr->getX(1) == 4);
    CHECK(attr->getZ(1) == 6);

    attr->setXYZ(0, 7, 8, 9);
    CHECK(attr->getX(0) == 7);
    CHECK(attr->getZ(0) == 9);
    // Writing element 0 must not disturb element 1.
    CHECK(attr->getX(1) == 4);
}

// copyAt was never instantiated, and its body ended in `return &this;` — taking
// the address of a prvalue, which is ill-formed. The first caller would have been
// a hard compile error, so this test is as much a compile check as a behaviour one.
TEST_CASE("BufferAttribute copyAt copies one element between attributes") {

    auto dst = FloatBufferAttribute::create(std::vector<float>{0, 0, 0, 0, 0, 0}, 3);
    auto src = FloatBufferAttribute::create(std::vector<float>{1, 2, 3, 4, 5, 6}, 3);

    auto& returned = dst->copyAt(1, *src, 0);

    CHECK(&returned == dst.get());// returns *this, not the address of `this`

    // Element 1 of dst now holds element 0 of src; element 0 is untouched.
    CHECK(dst->getX(0) == 0);
    CHECK(dst->getX(1) == 1);
    CHECK(dst->getY(1) == 2);
    CHECK(dst->getZ(1) == 3);
}

// applyMatrix4 and friends used to write through two shared `inline static`
// scratch vectors, so transforming two attributes concurrently produced
// interleaved garbage. The scratch is function-local now; this pins that.
TEST_CASE("BufferAttribute transforms are safe on concurrent attributes") {

    constexpr int kVerts = 512;
    constexpr int kThreads = 4;

    std::vector<std::unique_ptr<FloatBufferAttribute>> attrs;
    for (int t = 0; t < kThreads; ++t) {
        std::vector<float> data;
        data.reserve(kVerts * 3);
        for (int i = 0; i < kVerts; ++i) {
            data.insert(data.end(), {static_cast<float>(t), static_cast<float>(t), static_cast<float>(t)});
        }
        attrs.push_back(FloatBufferAttribute::create(std::move(data), 3));
    }

    // Each thread translates its own attribute by a distinct offset.
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&attrs, t] {
            Matrix4 m;
            m.setPosition(static_cast<float>(100 * t), 0, 0);
            attrs[t]->applyMatrix4(m);
        });
    }
    for (auto& th : threads) th.join();

    for (int t = 0; t < kThreads; ++t) {
        const float expectedX = static_cast<float>(t) + static_cast<float>(100 * t);
        for (int i = 0; i < kVerts; ++i) {
            REQUIRE(attrs[t]->getX(i) == expectedX);
            REQUIRE(attrs[t]->getY(i) == static_cast<float>(t));
        }
    }
}

// --- narrow attribute types -------------------------------------------------

TEST_CASE("Narrow attributes report their scalar type and byte footprint") {

    auto normals = Int16BufferAttribute::create(std::vector<std::int16_t>{32767, 0, 0, 0, 32767, 0}, 3, true);
    auto uvs = Uint16BufferAttribute::create(std::vector<std::uint16_t>{0, 0, 65535, 65535}, 2, true);
    auto colors = Uint8BufferAttribute::create(std::vector<std::uint8_t>{255, 128, 0}, 3, true);

    CHECK(normals->type() == AttributeType::Int16);
    CHECK(uvs->type() == AttributeType::UInt16);
    CHECK(colors->type() == AttributeType::UInt8);

    CHECK(normals->count() == 2);
    CHECK(normals->normalized());

    // The whole point: a 2-vertex normal attribute costs 12 bytes, not 24.
    CHECK(normals->byteLength() == 12);
    CHECK(uvs->byteLength() == 8);
    CHECK(colors->byteLength() == 3);

    CHECK(bytesPerElement(AttributeType::Float) == 4);
    CHECK(bytesPerElement(AttributeType::Int16) == 2);
    CHECK(bytesPerElement(AttributeType::UInt8) == 1);
}

TEST_CASE("denormalize follows the UNORM/SNORM conversion rules") {

    CHECK(denormalize(255.f, AttributeType::UInt8) == 1.f);
    CHECK(denormalize(0.f, AttributeType::UInt8) == 0.f);
    CHECK(denormalize(65535.f, AttributeType::UInt16) == 1.f);
    CHECK(denormalize(32767.f, AttributeType::Int16) == 1.f);

    // SNORM clamps the extra negative code point to -1 rather than -1.000030.
    CHECK(denormalize(-32768.f, AttributeType::Int16) == -1.f);
    CHECK(denormalize(-127.f, AttributeType::Int8) == -1.f);

    // Float attributes pass through untouched.
    CHECK(denormalize(7.5f, AttributeType::Float) == 7.5f);
}

TEST_CASE("cloneUntyped preserves the scalar type instead of widening") {

    auto narrow = Uint16BufferAttribute::create(std::vector<std::uint16_t>{1, 2, 3, 4}, 2, true);

    auto copy = narrow->cloneUntyped();

    REQUIRE(copy != nullptr);
    CHECK(copy->type() == AttributeType::UInt16);
    CHECK(copy->byteLength() == narrow->byteLength());
    CHECK(copy->normalized());
    CHECK(copy->itemSize() == 2);
}

TEST_CASE("FloatAttributeView is zero-copy for float attributes") {

    auto attr = FloatBufferAttribute::create(std::vector<float>{1, 2, 3, 4, 5, 6}, 3);

    FloatAttributeView view(attr.get());

    REQUIRE(static_cast<bool>(view));
    CHECK_FALSE(view.widened());
    // Points straight at the attribute's own storage — no allocation, no copy.
    CHECK(view.data() == attr->array().data());
    CHECK(view.size() == 6);
    CHECK(view.itemSize() == 3);
    CHECK(view.count() == 2);
}

TEST_CASE("FloatAttributeView widens and denormalizes narrow attributes") {

    auto colors = Uint8BufferAttribute::create(std::vector<std::uint8_t>{255, 128, 0}, 3, true);

    FloatAttributeView view(colors.get());

    REQUIRE(static_cast<bool>(view));
    CHECK(view.widened());
    CHECK(view.size() == 3);
    CHECK(view[0] == 1.f);
    CHECK(view[1] == 128.f / 255.f);
    CHECK(view[2] == 0.f);
}

TEST_CASE("FloatAttributeView leaves un-normalized narrow values raw") {

    auto indices = Uint16BufferAttribute::create(std::vector<std::uint16_t>{3, 7, 11, 0}, 4, false);

    FloatAttributeView view(indices.get());

    REQUIRE(static_cast<bool>(view));
    CHECK(view[0] == 3.f);
    CHECK(view[1] == 7.f);
    CHECK(view[2] == 11.f);
}

TEST_CASE("FloatAttributeView survives being moved") {

    auto colors = Uint8BufferAttribute::create(std::vector<std::uint8_t>{255, 0, 255}, 3, true);

    FloatAttributeView original(colors.get());
    FloatAttributeView moved(std::move(original));

    // The widened storage moved with it; data() must follow, not dangle.
    REQUIRE(static_cast<bool>(moved));
    CHECK(moved.widened());
    CHECK(moved[0] == 1.f);
    CHECK(moved[2] == 1.f);
}

TEST_CASE("FloatAttributeView of a null attribute is empty") {

    FloatAttributeView view(nullptr);

    CHECK_FALSE(static_cast<bool>(view));
    CHECK(view.size() == 0);
}
