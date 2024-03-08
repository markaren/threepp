

#ifndef THREEPP_LODRAYCAST_HPP
#define THREEPP_LODRAYCAST_HPP

#include "threepp/core/Raycaster.hpp"
#include "threepp/math/MathUtils.hpp"

#include "LODControl.hpp"
#include "MapNode.hpp"

namespace threepp {

    class LODRaycast: public LODControl {

    public:

        int subdivisionRays = 1;

        float thresholdUp = 0.6;

        float thresholdDown = 0.15;

        Raycaster raycaster;

        Vector2 mouse;

        bool powerDistance = false;

        bool scaleDistance = true;

        float getLevel(...) {
            return 0;
        }

        float getLevel(MapNode* node) {
            return node->getLevel();
        }

        void updateLOD(MapView& view, Camera& camera, GLRenderer& renderer, Object3D& scene) override {
            std::vector<Intersection> intersects;
            
            for (unsigned t = 0; t < this->subdivisionRays; t++)
            {
                // Generate random point in viewport
                this->mouse.set(math::randFloat() * 2 - 1, math::randFloat() * 2 - 1);

                // Check intersection
                this->raycaster.setFromCamera(this->mouse, camera);
                intersects = this->raycaster.intersectObjects(view.children, true);
            }

            for (unsigned i = 0; i < intersects.size(); i++)
            {
                auto* obj = intersects[i].object;

                if (auto node = dynamic_cast<MapNode*>(obj)) {

                    auto distance = intersects[i].distance;

                    if (this->powerDistance) {
                        distance = std::pow(distance * 2, node->getLevel());
                    }

                    if (this->scaleDistance) {
                        // Get scale from transformation matrix directly
                        const auto& matrix = node->matrixWorld->elements;
                        const auto vector = Vector3(matrix[0], matrix[1], matrix[2]);
                        distance = vector.length() / distance;
                    }

                    if (distance > this->thresholdUp) {
                        node->subdivide();
                    } else if (distance < this->thresholdDown && node->parentNode) {
                        node->parentNode->simplify();
                    }
                }
            }
        }
    };

}

#endif//THREEPP_LODRAYCAST_HPP
