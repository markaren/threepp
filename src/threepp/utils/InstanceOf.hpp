
#ifndef THREEPP_INSTANCEOF_HPP
#define THREEPP_INSTANCEOF_HPP

namespace {

    template<typename Base, typename T>
    inline bool instanceof(T *object) {

        return dynamic_cast<Base*>(object) != nullptr;
    }

}

#endif//THREEPP_INSTANCEOF_HPP
