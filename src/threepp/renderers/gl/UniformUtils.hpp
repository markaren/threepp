
#ifndef THREEPP_UNIFORMUTILS_HPP
#define THREEPP_UNIFORMUTILS_HPP

#include "threepp/renderers/gl/GLTextures.hpp"

#include <vector>

namespace threepp::gl {

    inline std::vector<std::vector<float>> arrayCacheF32;
    inline std::vector<std::vector<int>> arrayCacheI32;

    inline void ensureCapacity(std::vector<float>& v, unsigned int size) {

        while (v.size() < size) {
            v.emplace_back();
        }
    }

    // The scratch array for n floats. The cache only grows: resizing it down
    // to a shorter array's length freed the scratch of every longer one, which
    // the next long array then allocated again.
    inline std::vector<float>& arrayCacheSlotF32(int n) {

        if (arrayCacheF32.size() <= static_cast<size_t>(n)) arrayCacheF32.resize(static_cast<size_t>(n) + 1);

        return arrayCacheF32[n];
    }

    template<class ArrayLike>
    std::vector<float>& flatten(const ArrayLike& array, int nBlocks, int blockSize) {

        const auto n = nBlocks * blockSize;
        auto& r = arrayCacheSlotF32(n);

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
        auto& r = arrayCacheSlotF32(n);

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

    inline std::vector<int>& allocTexUnits(GLTextures& textures, size_t n) {

        while (n >= arrayCacheI32.size()) {
            arrayCacheI32.emplace_back(arrayCacheI32.size());
        }

        auto& r = arrayCacheI32[n];

        for (size_t i = 0; i != n; ++i) {

            r[i] = textures.allocateTextureUnit();
        }

        return r;
    }

}// namespace threepp::gl

#endif//THREEPP_UNIFORMUTILS_HPP
