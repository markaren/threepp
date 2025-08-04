
#ifndef THREEPP_TASKMANAGER_HPP
#define THREEPP_TASKMANAGER_HPP

#include <memory>
#include <functional>

namespace threepp {

    class TaskManager {

    public:
        TaskManager();

        void handleTasks();

        void invokeLater(const std::function<void()>& task, double delay = -1);

        ~TaskManager();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TASKMANAGER_HPP
