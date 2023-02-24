
#ifndef THREEPP_CAMERAVIEW_HPP
#define THREEPP_CAMERAVIEW_HPP

namespace threepp {

    struct CameraView {

        bool enabled{};
        int fullWidth{};
        int fullHeight{};
        int offsetX{};
        int offsetY{};
        int width{};
        int height{};
    };

}// namespace threepp

#endif//THREEPP_CAMERAVIEW_HPP
