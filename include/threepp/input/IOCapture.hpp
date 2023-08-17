
#ifndef THREEPP_IOCAPTURE_HPP
#define THREEPP_IOCAPTURE_HPP

#include <functional>

namespace threepp {

    using MouseCaptureCallback = std::function<bool(void)>;
    using ScrollCaptureCallback = std::function<bool(void)>;
    using KeyboardCaptureCallback = std::function<bool(void)>;

    struct IOCapture {
        MouseCaptureCallback preventMouseEvent = [] { return false; };
        ScrollCaptureCallback preventScrollEvent = [] { return false; };
        KeyboardCaptureCallback preventKeyboardEvent = [] { return false; };
    };

}// namespace threepp

#endif//THREEPP_IOCAPTURE_HPP
