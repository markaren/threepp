
#ifndef THREEPP_MONITOR_HPP
#define THREEPP_MONITOR_HPP

#include "threepp/canvas/WindowSize.hpp"

#include <utility>

namespace threepp::monitor {

    //the size of the Monitor
    [[nodiscard]] WindowSize monitorSize(int monitor = 0);

    // query the dpi scale of the Monitor
    [[nodiscard]] std::pair<float, float> contentScale(int monitor = 0);

    // the number of connected Monitors (1 in the browser)
    [[nodiscard]] int monitorCount();

    // the top-left corner of the Monitor in screen coordinates, the space
    // Canvas::setPosition works in
    [[nodiscard]] std::pair<int, int> monitorPosition(int monitor = 0);

}

#endif //THREEPP_MONITOR_HPP
