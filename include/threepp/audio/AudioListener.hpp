
#ifndef THREEPP_AUDIOLISTENER_HPP
#define THREEPP_AUDIOLISTENER_HPP

#include "threepp/core/Object3D.hpp"

namespace threepp {

    class AudioListener: public Object3D {

    public:
        [[nodiscard]] std::string type() const override {

            return "AudioListener";
        }
    };

}

#endif//THREEPP_AUDIOLISTENER_HPP
