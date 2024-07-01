
#ifndef THREEPP_GLMORPHTARGETS_HPP
#define THREEPP_GLMORPHTARGETS_HPP

#include "GLProgram.hpp"
#include "GLUniforms.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/materials.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>

namespace {

    using Influence = std::pair<size_t, float>;

    const auto MAX_SAFE_INTEGER = std::numeric_limits<unsigned int>::max();

    bool numericalSort(const Influence& a, const Influence& b) {
        return a.first < b.first;
    }

    bool absNumericalSort(const Influence& a, const Influence& b) {
        return std::abs(b.second) < std::abs(a.second);
    }

}// namespace

namespace threepp::gl {

    class GLMorphTargets {

    public:
        std::unordered_map<unsigned int, std::vector<Influence>> influencesList;
        std::vector<float> morphInfluences;

        std::vector<Influence> workInfluences;

        GLMorphTargets(): morphInfluences(8) {

            for (int i = 0; i < 8; i++) {

                workInfluences.emplace_back(i, 0.f);
            }
        }

        void update(Object3D* object, BufferGeometry* geometry, Material* material, GLProgram* program) {

            std::vector<float> objectInfluences;
            if (auto objectWithMorphTargetInfluences = dynamic_cast<ObjectWithMorphTargetInfluences*>(object)) {
                objectInfluences = objectWithMorphTargetInfluences->morphTargetInfluences();
            }

            auto length = objectInfluences.size();

            std::vector<Influence> influences;

            if (influencesList.contains(geometry->id)) {

                influences = influencesList.at(geometry->id);

            } else {

                for (unsigned i = 0; i < length; i++) {

                    influences.emplace_back(i, 0.f);
                }

                influencesList[geometry->id] = influences;
            }

            // Collect influences

            for (unsigned i = 0; i < length; i++) {

                auto& influence = influences.at(i);

                influence.first = i;
                influence.second = objectInfluences[i];
            }

            std::ranges::stable_sort(influences, absNumericalSort);

            for (unsigned i = 0; i < 8; i++) {

                if (i < length && influences.at(i).second > 0) {

                    workInfluences[i].first = influences.at(i).first;
                    workInfluences[i].second = influences.at(i).second;

                } else {

                    workInfluences[i].first = MAX_SAFE_INTEGER;
                    workInfluences[i].second = 0;
                }
            }

            std::stable_sort(workInfluences.begin(), workInfluences.end(), numericalSort);

            std::vector<std::shared_ptr<BufferAttribute>>* morphTargets = nullptr;
            std::vector<std::shared_ptr<BufferAttribute>>* morphNormals = nullptr;
            if (auto m = material->as<MaterialWithMorphTargets>()) {
                if (m->morphTargets) {
                    morphTargets = geometry->getMorphAttribute("position");
                }
                if (m->morphNormals) {
                    morphNormals = geometry->getMorphAttribute("normal");
                }
            }

            float morphInfluencesSum = 0;

            for (int i = 0; i < 8; i++) {

                auto& influence = workInfluences.at(i);
                auto index = influence.first;
                auto value = influence.second;

                std::string morphTarget_i = "morphTarget" + std::to_string(i);
                std::string morphNormal_i = "morphNormal" + std::to_string(i);

                if (index != MAX_SAFE_INTEGER && value > 0) {

                    if (morphTargets && geometry->getAttribute(morphTarget_i) != (*morphTargets)[index].get()) {

                        auto attr = morphTargets->at(index);
                        geometry->setAttribute(morphTarget_i, attr);
                    }

                    if (morphNormals && geometry->getAttribute(morphNormal_i) != (*morphNormals)[index].get()) {

                        auto attr = morphNormals->at(index);
                        geometry->setAttribute(morphNormal_i, attr);
                    }

                    morphInfluences[i] = value;
                    morphInfluencesSum += value;

                } else {

                    if (morphTargets && geometry->hasAttribute(morphTarget_i) == true) {

                        geometry->deleteAttribute(morphTarget_i);
                    }

                    if (morphNormals && geometry->hasAttribute(morphNormal_i) == true) {

                        geometry->deleteAttribute(morphNormal_i);
                    }

                    morphInfluences[i] = 0;
                }
            }

            // GLSL shader uses formula baseinfluence * base + sum(target * influence)
            // This allows us to switch between absolute morphs and relative morphs without changing shader code
            // When baseinfluence = 1 - sum(influence), the above is equivalent to sum((target - base) * influence)
            float morphBaseInfluence = geometry->morphTargetsRelative ? 1.f : 1.f - morphInfluencesSum;

            program->getUniforms()->setValue("morphTargetBaseInfluence", morphBaseInfluence);
            program->getUniforms()->setValue("morphTargetInfluences", morphInfluences);
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLMORPHTARGETS_HPP
