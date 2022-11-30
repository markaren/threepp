
#ifndef THREEPP_INSTANCEOF_HPP
#define THREEPP_INSTANCEOF_HPP

#include <memory>

namespace {

    template<typename Base, typename T>
    inline bool instanceof(T *object) {

        return dynamic_cast<Base*>(object) != nullptr;
    }

    template<typename Base, typename T>
    inline bool instanceof(std::shared_ptr<T> object) {

        return std::dynamic_pointer_cast<Base>(object) != nullptr;
    }

}

#endif//THREEPP_INSTANCEOF_HPP
