
#ifndef THREEPP_PROPERTYMIXER_HPP
#define THREEPP_PROPERTYMIXER_HPP

#include "threepp/animation/PropertyBinding.hpp"

#include <string>

#include <memory>

namespace threepp {

    class PropertyMixer {

    public:
        PropertyMixer(const PropertyBinding& binding, const std::string& typeName, int valueSize);

        ~PropertyMixer();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_PROPERTYMIXER_HPP
