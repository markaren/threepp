
#ifndef THREEPP_UNIFORMUTILS_HPP
#define THREEPP_UNIFORMUTILS_HPP

#include "threepp/renderers/gl/GLTextures.hpp"

#include <vector>

namespace {

    std::vector<std::vector<float>> arrayCacheF32;
    std::vector<std::vector<int>> arrayCacheI32;

    void ensureCapacity(std::vector<float>& v, unsigned int size) {

        while (v.size() < size) {
            v.emplace_back();
        }
    }

    template<class ArrayLike>
    std::vector<float>& flatten(const ArrayLike& array, int nBlocks, int blockSize) {

        const auto n = nBlocks * blockSize;
        arrayCacheF32.resize(n + 1);
        auto& r = arrayCacheF32[n];

        if (r.empty()) r.resize(n + 1);

        if (nBlocks != 0) {

            int offset = 0;
            for (int i = 0; i < nBlocks; ++i) {

                array[i].toArray(r, offset);
                offset += blockSize;
            }
        }

        return r;
    }

    template<class ArrayLike>
    std::vector<float>& flattenP(const ArrayLike& array, int nBlocks, int blockSize) {

        const auto n = nBlocks * blockSize;
        arrayCacheF32.resize(n + 1);
        auto& r = arrayCacheF32[n];

        if (r.empty()) r.resize(n + 1);

        if (nBlocks != 0) {

            int offset = 0;
            for (int i = 0; i < nBlocks; ++i) {

                array[i]->toArray(r, offset);
                offset += blockSize;
            }
        }

        return r;
    }

    template<class ArrayLike1, class ArrayLike2>
    bool arraysEqual(const ArrayLike1& a, const ArrayLike2& b) {

        if (a.size() != b.size()) return false;

        for (unsigned i = 0, l = a.size(); i < l; ++i) {

            if (a[i] != b[i]) return false;
        }

        return true;
    }

    template<class ArrayLike1, class ArrayLike2>
    void copyArray(ArrayLike1& a, const ArrayLike2& b) {

        for (unsigned i = 0, l = b.size(); i < l; ++i) {

            a[i] = b[i];
        }
    }

    // Texture unit allocation

    std::vector<int>& allocTexUnits(threepp::gl::GLTextures& textures, size_t n) {

        while (n >= arrayCacheI32.size()) {
            arrayCacheI32.emplace_back(arrayCacheI32.size());
        }

        auto& r = arrayCacheI32[n];

        //        if (r.empty()) {
        //
        //            r.resize(n);
        //            arrayCacheI32[n] = r;
        //        }

        for (size_t i = 0; i != n; ++i) {

            r[i] = textures.allocateTextureUnit();
        }

        return r;
    }

}// namespace

#endif//THREEPP_UNIFORMUTILS_HPP
