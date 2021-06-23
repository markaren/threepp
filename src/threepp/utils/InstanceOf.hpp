
#ifndef THREEPP_INSTANCEOF_HPP
#define THREEPP_INSTANCEOF_HPP

namespace {

    template<typename Base, typename T>
    inline bool instanceof(const T*) {
        return std::is_base_of<Base, T>::value;
    }

}

#endif//THREEPP_INSTANCEOF_HPP
