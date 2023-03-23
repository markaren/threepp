// https://github.com/mrdoob/three.js/blob/r129/examples/js/objects/Sky.js

#ifndef THREEPP_SKY_HPP
#define THREEPP_SKY_HPP

#include "threepp/objects/Mesh.hpp"

namespace threepp {

    /**
     * Based on "A Practical Analytic Model for Daylight"
     * aka The Preetham Model, the de facto standard analytic skydome model
     * https://www.researchgate.net/publication/220720443_A_Practical_Analytic_Model_for_Daylight
     *
     * First implemented by Simon Wallner
     * http://www.simonwallner.at/projects/atmospheric-scattering
     *
     * Improved by Martin Upitis
     * http://blenderartists.org/forum/showthread.php?245954-preethams-sky-impementation-HDR
     *
     * Three.js integration by zz85 http://twitter.com/blurspline
    */
    class Sky: public Mesh {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<Sky> create();

    private:
        Sky();
    };

}// namespace threepp

#endif//THREEPP_SKY_HPP
