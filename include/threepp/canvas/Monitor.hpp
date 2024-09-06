
#ifndef MONITOR_HPP
#define MONITOR_HPP

#include "threepp/canvas/WindowSize.hpp"

namespace threepp::monitor {

    //the size of the (primary) Monitor
    [[nodiscard]] WindowSize monitorSize();

    // query the dpi scale of the (primary) Monitor
    [[nodiscard]] std::pair<float, float> contentScale();

}

#endif //MONITOR_HPP
