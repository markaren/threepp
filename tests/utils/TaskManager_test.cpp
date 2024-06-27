
#include <catch2/catch_test_macros.hpp>

#include "threepp/utils/TaskManager.hpp"

#include <thread>

using namespace threepp;


TEST_CASE("TaskManager") {

    TaskManager manager;

    bool task1 = false;
    manager.invokeLater([&task1]() { task1 = true; });

    REQUIRE(!task1);
    manager.handleTasks();
    REQUIRE(task1);
}

TEST_CASE("Timed task") {

    TaskManager manager;

    bool task = false;
    manager.invokeLater([&task]() { task = true; }, 0.2);
    manager.handleTasks();
    REQUIRE(!task);
    std::this_thread::sleep_for(std::chrono::milliseconds(210));
    manager.handleTasks();
    REQUIRE(task);
}

TEST_CASE("Nested tasks") {

    TaskManager manager;

    bool task1 = false;
    bool task2 = false;
    bool task3 = false;
    bool task4 = false;
    manager.invokeLater([&]() {
        task1 = true;
        manager.invokeLater([&] {
            task2 = true;
            manager.invokeLater([&] {
                task3 = true;
            });// nested-nested task
        });    // nested task
        manager.invokeLater([&] {
            task4 = true;
        },
                            0);// nested-nested task
    });
    REQUIRE(!task1);
    REQUIRE(!task2);
    REQUIRE(!task3);
    REQUIRE(!task4);
    manager.handleTasks();
    REQUIRE(task1);
    REQUIRE(task2);
    REQUIRE(task3);
    REQUIRE(!task4);
    manager.handleTasks();
    REQUIRE(task4);
}

TEST_CASE("Threaded") {

    TaskManager manager;

    for (int i = 0; i < 100; i++) {
        bool task1 = false;
        bool task2 = false;
        manager.handleTasks();
        std::thread t1([&]() {
            manager.invokeLater([&]() {
                task1 = true;
            });
        });
        manager.handleTasks();
        std::thread t2([&]() {
            manager.invokeLater([&]() {
                task2 = true;
            });
        });

        t1.join();
        t2.join();

        manager.handleTasks();
        REQUIRE(task1);
        REQUIRE(task2);
    }
}
