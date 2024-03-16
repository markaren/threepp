
#ifndef THREEPP_TASKMANAGER_HPP
#define THREEPP_TASKMANAGER_HPP

#include <functional>
#include <queue>
#include <utility>

namespace threepp::utils {

    using Task = std::pair<std::function<void()>, float>;

    class TaskManager {

    public:
        inline void handleTasks(float currentTime) {
            while (!tasks_.empty()) {
                auto& task = tasks_.top();
                if (task.second < currentTime) {
                    task.first();
                    tasks_.pop();
                } else {
                    break;
                }
            }
        }

        void invokeLater(const std::function<void()>& f, float tNow, float tPassed = 0) {

            tasks_.emplace(f, tNow + tPassed);
        }

    private:
        struct CustomComparator {
            bool operator()(const Task& l, const Task& r) const { return l.second > r.second; }
        };

        std::priority_queue<Task, std::vector<Task>, CustomComparator> tasks_;
    };

}// namespace threepp::utils

#endif//THREEPP_TASKMANAGER_HPP
