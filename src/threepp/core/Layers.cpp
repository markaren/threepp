
#include "threepp/core/Layers.hpp"

using namespace threepp;


Layers::Layers(): mask_(1 | 0) {}

void Layers::set(unsigned int channel) {

    this->mask_ = 1 << channel | 0;
}

void Layers::enable(unsigned int channel) {

    this->mask_ |= 1 << channel | 0;
}

void Layers::enableAll() {

    this->mask_ = 0xffffffff | 0;
}

void Layers::toggle(unsigned int channel) {

    this->mask_ ^= 1 << channel | 0;
}

void Layers::disable(unsigned int channel) {

    this->mask_ &= ~(1 << channel | 0);
}

void Layers::disableAll() {

    this->mask_ = 0;
}

bool Layers::test(const Layers& layers) const {

    return (this->mask_ & layers.mask_) != 0;
}

bool Layers::isEnabled(unsigned int channel) const {

    return (mask_ & (1 << channel | 0)) != 0;
}

unsigned int Layers::mask() const {
    return mask_;
}
