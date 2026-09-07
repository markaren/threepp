
#ifndef THREEPP_MONITOR_HPP
#define THREEPP_MONITOR_HPP

#include "threepp/canvas/WindowSize.hpp"

namespace threepp::monitor {

    //the size of the Monitor
    [[nodiscard]] WindowSize monitorSize(int monitor = 0);

    // query the dpi scale of the Monitor
    [[nodiscard]] std::pair<float, float> contentScale(int monitor = 0);

}

#endif //THREEPP_MONITOR_HPP
