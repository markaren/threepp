
#include "GLUniforms.hpp"

#include "GLTextures.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;
using namespace threepp::gl;

#include <array>
#include <vector>

namespace {

    Texture emptyTexture;

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

    template<class T>
    bool arraysEqual(const T &a, const T &b) {

        if (a.size() != b.size()) return false;

        for (int i = 0, l = a.size(); i < l; i++) {

            if (a[i] != b[i]) return false;
        }

        return true;
    }

    template<class ArrayLike>
    void copyArray(const ArrayLike &a, ArrayLike &b) {

        for (int i = 0, l = b.size(); i < l; i++) {

            a[i] = b[i];
        }
    }

    // Texture unit allocation

    std::vector<int> &allocTexUnits(GLTextures &textures, int n) {

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

}// namespace
