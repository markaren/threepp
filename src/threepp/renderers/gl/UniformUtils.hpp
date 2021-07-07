
#ifndef THREEPP_UNIFORMUTILS_HPP
#define THREEPP_UNIFORMUTILS_HPP

#include "GLTextures.hpp"

#include <vector>

namespace {

    threepp::Texture emptyTexture;

    std::vector<std::vector<float>> arrayCacheF32;
    std::vector<std::vector<int>> arrayCacheI32;

    std::array<float, 16> mat4array;
    std::array<float, 9> mat3array;
    std::array<float, 4> mat2array;

    template<class ArrayLike>
    std::vector<float> &flatten(const ArrayLike &array, int nBlocks, int blockSize) {

        const auto firstElem = array[0];

        if (!isnan(firstElem)) return array;

        const auto n = nBlocks * blockSize;
        auto &r = arrayCacheF32[n];

        if (r.empty()) {

            r.resize(n);
            arrayCacheF32[n] = r;
        }

        if (nBlocks != 0) {

            firstElem.toArray(r, 0);

            for (int i = 1, offset = 0; i != nBlocks; ++i) {

                offset += blockSize;
                array[i].toArray(r, offset);
            }
        }

        return r;
    }

    template<class ArrayLike1, class ArrayLike2>
    bool arraysEqual(const ArrayLike1 &a, const ArrayLike2 &b) {

        if (a.size() != b.size()) return false;

        for (int i = 0, l = a.size(); i < l; i++) {

            if (a[i] != b[i]) return false;
        }

        return true;
    }

    template<class ArrayLike1, class ArrayLike2>
    void copyArray(ArrayLike1 &a, const ArrayLike2 &b) {

        for (int i = 0, l = (int) b.size(); i < l; i++) {

            a[i] = b[i];
        }
    }

    // Texture unit allocation

    std::vector<int> &allocTexUnits(threepp::gl::GLTextures &textures, int n) {

        auto &r = arrayCacheI32[n];

        if (r.empty()) {

            r.resize(n);
            arrayCacheI32[n] = r;
        }

        for (int i = 0; i != n; ++i) {

            r[i] = textures.allocateTextureUnit();
        }

        return r;
    }

}

#endif//THREEPP_UNIFORMUTILS_HPP
