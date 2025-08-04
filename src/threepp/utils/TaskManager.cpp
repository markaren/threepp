
#include "threepp/utils/TaskManager.hpp"

#include <mutex>
#include <queue>
#include <utility>
#include <chrono>

using namespace threepp;

using Task = std::function<void()>;

struct TaskManager::Impl {

    void handleTasks() {

        if (previousTime_ < 0) {
            previousTime_ = getCurrentTimeInSeconds();// first invocation
        }

        const auto currentTime = getCurrentTimeInSeconds();
        const auto deltaTime = currentTime - previousTime_;
        time_ += deltaTime;
        previousTime_ = currentTime;

        std::unique_lock lock(m_);
        if (tasks_.empty()) return;

        while (!tasks_.empty()) {
            if (tasks_.top().second < time_) {
                auto task = tasks_.top();
                tasks_.pop();
                lock.unlock();
                task.first();
                lock.lock();
            } else {
                break;
            }
        }
    }

    void invokeLater(const Task& task, double delay) {

        std::lock_guard lock(m_);
        tasks_.emplace(task, time_ + delay);
    }

private:
    std::mutex m_;

    double time_{};
    double previousTime_{-1};

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

TaskManager::TaskManager()
    : pimpl_(std::make_unique<Impl>()) {}

void TaskManager::handleTasks() {
    pimpl_->handleTasks();
}

void TaskManager::invokeLater(const Task& task, double delay) {
    pimpl_->invokeLater(task, delay);
}

TaskManager::~TaskManager() = default;
