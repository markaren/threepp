
#ifndef THREEPP_TIMER_HPP
#define THREEPP_TIMER_HPP

#include <chrono>

namespace threepp {

    // https://codereview.stackexchange.com/questions/48872/measuring-execution-time-in-c
    template<typename TimeT = std::chrono::milliseconds>
    struct measure {
        template<typename F, typename... Args>
        static typename TimeT::rep execution(F func, Args&&... args) {
            auto start = std::chrono::system_clock::now();

            // Now call the function with all the parameters you need.
            func(std::forward<Args>(args)...);

            auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);

            return duration.count();
        }
    };

}// namespace threepp


#endif//THREEPP_TIMER_HPP
