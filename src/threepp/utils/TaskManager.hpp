
#ifndef THREEPP_TASKMANAGER_HPP
#define THREEPP_TASKMANAGER_HPP

#include <chrono>
#include <functional>
#include <queue>
#include <utility>

namespace threepp::utils {

    using Task = std::function<void()>;

    class TaskManager {

    public:
        inline void handleTasks() {

            if (previousTime < 0) {
                previousTime = getCurrentTimeInSeconds();// first invocation
            }

            const auto currentTime = getCurrentTimeInSeconds();
            const auto deltaTime = currentTime - previousTime;
            time += deltaTime;
            previousTime = currentTime;

            while (!tasks_.empty()) {
                auto& task = tasks_.top();
                if (task.second < time) {
                    task.first();
                    tasks_.pop();
                } else {
                    break;
                }
            }
        }

        void invokeLater(const Task& task, double delay = 0) {

            tasks_.emplace(task, time + delay);
        }

    private:
        double time{};
        double previousTime = -1;

        using TimedTask = std::pair<std::function<void()>, double>;

        struct CustomComparator {
            bool operator()(const TimedTask& l, const TimedTask& r) const { return l.second > r.second; }
        };

        std::priority_queue<TimedTask, std::vector<TimedTask>, CustomComparator> tasks_;

        static double getCurrentTimeInSeconds() {

            using Clock = std::chrono::high_resolution_clock;

            const auto now = Clock::now();
            const auto duration = now.time_since_epoch();

            return std::chrono::duration<double>(duration).count();
        }
    };

}// namespace threepp::utils

#endif//THREEPP_TASKMANAGER_HPP
