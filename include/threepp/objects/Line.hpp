// https://github.com/mrdoob/three.js/blob/r129/src/objects/Line.js

#ifndef THREEPP_LINE_HPP
#define THREEPP_LINE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace threepp {

    class Line : public Object3D {

    public:
        BufferGeometry* geometry() override {

            return geometry_.get();
        }

        Material* material() override {

            return material_.get();
        }

        std::vector<Material*> materials() override {
            return {material_.get()};
        }

        virtual void computeLineDistances() {

            // we assume non-indexed geometry

            if (!geometry_->hasIndex()) {

                const auto positionAttribute = geometry_->getAttribute<float>("position");
                std::vector<float> lineDistances{0};

                for (int i = 1, l = positionAttribute->count(); i < l; i++) {

                    positionAttribute->setFromBufferAttribute(_start, i - 1);
                    positionAttribute->setFromBufferAttribute(_end, i);

                    lineDistances[i] = lineDistances[i - 1];
                    lineDistances[i] += _start.distanceTo(_end);
                }

                geometry_->setAttribute("lineDistance", TypedBufferAttribute<float>::create(lineDistances, 1));

            } else {

                std::cerr << "THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry." << std::endl;
            }
        }

        static std::shared_ptr<Line> create(const std::shared_ptr<BufferGeometry>& geometry, std::shared_ptr<Material> material) {

            return std::shared_ptr<Line>(new Line(geometry, std::move((material))));
        }

    protected:
        std::shared_ptr<BufferGeometry> geometry_;
        std::shared_ptr<Material> material_;

        inline static Vector3 _start;
        inline static Vector3 _end;

        Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : geometry_(std::move(geometry)), material_(std::move(material)) {}
    };

}// namespace threepp

#endif//THREEPP_LINE_HPP
