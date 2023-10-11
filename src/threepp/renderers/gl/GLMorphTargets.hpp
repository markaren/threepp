
#ifndef THREEPP_GLMORPHTARGETS_HPP
#define THREEPP_GLMORPHTARGETS_HPP

#include "GLProgram.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

#include <algorithm>
#include <unordered_map>
#include <limits>

namespace {

    bool numericalSort(const std::vector<int>& a, const std::vector<int>& b) {
        return a[0] < b[0];
    }

    bool absNumericalSort(const std::vector<int>& a, const std::vector<int>& b) {
        return std::abs(b[1]) < std::abs(a[1]);
    }

}

namespace threepp::gl {

    class GLMorphTargets {

    public:
        std::unordered_map<unsigned int, std::vector<std::vector<int>>> influencesList;
        std::vector<float> morphInfluences;

        std::vector<std::vector<int>> workInfluences;

        GLMorphTargets() {

            for (int i = 0; i < 8; i++) {

                auto& influences = workInfluences.emplace_back();
                influences.insert(influences.begin(), {static_cast<float>(i), 0});
            }
        }

        void update(Object3D* object, BufferGeometry* geometry, Material* material, GLProgram& program) {

            std::vector<float> objectInfluences;
            if (auto objectWithMorphTargetInfluences = dynamic_cast<ObjectWithMorphTargetInfluences*>(object)) {
                objectInfluences = objectWithMorphTargetInfluences->morphTargetInfluences;
            }

            auto length = objectInfluences.size();

            std::vector<std::vector<int>> influences;

            if (influencesList.count(geometry->id)) {

                influences = influencesList.at(geometry->id);

            } else {

                for (int i = 0; i < length; i++) {

                    auto& list = influences.emplace_back();
                    list.insert(list.begin(), {i, 0});
                }

                influencesList[geometry->id] = influences;
            }

            // Collect influences

            for ( unsigned i = 0; i < length; i ++ ) {

                auto& influence = influences[ i ];

                influence[ 0 ] = i;
                influence[ 1 ] = objectInfluences[ i ];

            }

            std::sort(influences.begin(), influences.end(), absNumericalSort);

            for ( unsigned i = 0; i < 8; i ++ ) {

                if ( i < length && influences[ i ][ 1 ] ) {

                    workInfluences[ i ][ 0 ] = influences[ i ][ 0 ];
                    workInfluences[ i ][ 1 ] = influences[ i ][ 1 ];

                } else {

                    workInfluences[ i ][ 0 ] = std::numeric_limits<int>::max();
                    workInfluences[ i ][ 1 ] = 0;

                }

            }

            std::sort(workInfluences.begin(), workInfluences.end(), absNumericalSort);
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLMORPHTARGETS_HPP
