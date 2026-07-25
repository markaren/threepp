
#include <catch2/catch_test_macros.hpp>

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/math/Matrix4.hpp"

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
