// https://github.com/mrdoob/three.js/blob/r129/src/core/Layers.js

#ifndef THREEPP_LAYERS_HPP
#define THREEPP_LAYERS_HPP

namespace threepp {

    class Layers {

    public:
        Layers() : mask_(1 | 0) {}

        void set(unsigned int channel) {

            this->mask_ = 1 << channel | 0;
        }

        void enable(unsigned int channel) {

            this->mask_ |= 1 << channel | 0;
        }

        void enableAll() {

            this->mask_ = 0xffffffff | 0;
        }

        void toggle(unsigned int channel) {

            this->mask_ ^= 1 << channel | 0;
        }

        void disable(unsigned int channel) {

            this->mask_ &= ~(1 << channel | 0);
        }

        void disableAll() {

            this->mask_ = 0;
        }

        bool test(Layers &layers) const {

            return (this->mask_ & layers.mask_) != 0;
        }

        [[nodiscard]] bool isEnabled(unsigned int channel) const {

            return (mask_ & (1 << channel | 0)) != 0;
        }


    private:
        unsigned int mask_;

        friend class Object3D;
    };

}// namespace threepp

#endif//THREEPP_LAYERS_HPP
