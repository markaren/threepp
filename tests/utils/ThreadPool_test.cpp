
#include <catch2/catch_test_macros.hpp>

#include "threepp/utils/ThreadPool.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace threepp;

TEST_CASE("ThreadPool runs every item once") {

    ThreadPool pool(3);
    REQUIRE(pool.concurrency() == 4);

    for (const int n : {0, 1, 7, 64, 1000, 100003}) {
        std::vector<int> hits(n, 0);
        pool.parallelFor(0, n, [&](int i) { ++hits[i]; });
        for (int i = 0; i < n; ++i) REQUIRE(hits[i] == 1);
    }

    // a range that does not start at zero, and an explicit grain
    std::vector<int> hits(50, 0);
    pool.parallelFor(10, 40, [&](int i) { ++hits[i]; }, 4);
    for (int i = 0; i < 50; ++i) CHECK(hits[i] == (i >= 10 && i < 40 ? 1 : 0));
}

TEST_CASE("ThreadPool gives the same result on any number of threads") {

    const int n = 20000;
    std::vector<double> serial(n), parallel(n);
    const auto fill = [](std::vector<double>& out) {
        return [&out](int i) { out[i] = std::sqrt(static_cast<double>(i)) * 1.5 + i % 7; };
    };
    ThreadPool none(0);
    none.parallelFor(0, n, fill(serial));
    ThreadPool many(7);
    many.parallelFor(0, n, fill(parallel));
    CHECK(serial == parallel);
}

TEST_CASE("ThreadPool survives many short loops in a row") {

    // the frame of a simulation: tens of loops of a few hundred items, back to back, thousands of times
    ThreadPool pool(5);
    std::vector<int> data(300, 0);
    for (int round = 0; round < 5000; ++round) {
        pool.parallelFor(0, 300, [&](int i) { data[i] += 1; });
    }
    for (int v : data) REQUIRE(v == 5000);
}

TEST_CASE("ThreadPool runs nested loops, three deep") {

    ThreadPool pool(5);
    std::vector<std::vector<std::vector<int>>> cells(12, std::vector<std::vector<int>>(10, std::vector<int>(40, 0)));
    for (int round = 0; round < 50; ++round) {
        pool.parallelFor(0, 12, [&](int a) {
            pool.parallelFor(0, 10, [&](int b) {
                pool.parallelFor(0, 40, [&](int c) { ++cells[a][b][c]; }, 1);
            }, 1);
        }, 1);
    }
    for (const auto& plane : cells)
        for (const auto& row : plane)
            for (int v : row) REQUIRE(v == 50);
}

TEST_CASE("ThreadPool takes nested loops from several threads at once") {

    // the renderer's shape: a few threads each run a loop whose items open loops of their own
    ThreadPool pool(6);
    constexpr int callers = 4;
    std::vector<std::vector<int>> out(callers, std::vector<int>(64 * 32, 0));
    std::vector<std::thread> threads;
    for (int t = 0; t < callers; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < 100; ++round) {
                pool.parallelFor(0, 64, [&](int a) {
                    pool.parallelFor(0, 32, [&](int b) { ++out[t][a * 32 + b]; }, 4);
                }, 1);
            }
        });
    }
    for (auto& th : threads) th.join();
    for (const auto& v : out)
        for (int x : v) REQUIRE(x == 100);
}

TEST_CASE("ThreadPool runs a loop in order when every slot is taken") {

    // more loops open at once than the pool has slots: the extra ones run on their callers' threads
    ThreadPool pool(3);
    std::vector<int> hits(ThreadPool::kSlots * 4 * 8, 0);
    pool.parallelFor(0, ThreadPool::kSlots * 4, [&](int a) {
        pool.parallelFor(0, 8, [&](int b) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            ++hits[a * 8 + b];
        }, 1);
    }, 1);
    for (int v : hits) REQUIRE(v == 1);
}

TEST_CASE("ThreadPool takes loops from two threads at once") {

    ThreadPool pool(4);
    std::vector<int> a(5000, 0), b(5000, 0);
    std::thread other([&] {
        for (int round = 0; round < 200; ++round) pool.parallelFor(0, 5000, [&](int i) { ++a[i]; });
    });
    for (int round = 0; round < 200; ++round) pool.parallelFor(0, 5000, [&](int i) { ++b[i]; });
    other.join();
    for (int i = 0; i < 5000; ++i) {
        REQUIRE(a[i] == 200);
        REQUIRE(b[i] == 200);
    }
}

TEST_CASE("ThreadPool passes on an exception from the calling thread and stays usable") {

    ThreadPool pool(2);
    const auto caller = std::this_thread::get_id();
    CHECK_THROWS_AS(pool.parallelFor(0, 1000, [&](int) {
        if (std::this_thread::get_id() == caller) throw std::runtime_error("from the caller");
    }, 1),
                    std::runtime_error);

    std::vector<int> hits(100, 0);
    pool.parallelFor(0, 100, [&](int i) { ++hits[i]; });
    for (int v : hits) REQUIRE(v == 1);
}

TEST_CASE("ThreadPool wakes its workers after they went to sleep") {

    ThreadPool pool(3);
    std::vector<int> hits(4000, 0);
    pool.parallelFor(0, 4000, [&](int i) { ++hits[i]; });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));// well past the workers' spin
    pool.parallelFor(0, 4000, [&](int i) { ++hits[i]; }, 1);
    for (int v : hits) REQUIRE(v == 2);
}

TEST_CASE("ThreadPool::shared is one pool") {

    CHECK(&ThreadPool::shared() == &ThreadPool::shared());
    std::vector<int> hits(1000, 0);
    ThreadPool::shared().parallelFor(0, 1000, [&](int i) { ++hits[i]; });
    for (int v : hits) REQUIRE(v == 1);
}
