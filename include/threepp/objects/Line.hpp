// https://github.com/mrdoob/three.js/blob/r129/src/objects/Line.js

#ifndef THREEPP_LINE_HPP
#define THREEPP_LINE_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

#include <memory>
#include <utility>
#include <vector>
#include <iostream>

namespace threepp {

    class Line : public Object3D {

    public:
        Line(const Line &) = delete;

        const std::shared_ptr<BufferGeometry> &geometry() const {
            return geometry_;
        }
        const std::shared_ptr<Material> &material() const {
            return material_;
        }

        Line &computeLineDistances() {

            // we assume non-indexed geometry

            if (!geometry_->getIndex().empty()) {

                const auto positionAttribute = geometry_->getAttribute<float>("position");
                std::vector<float> lineDistances{0};

                for (int i = 1, l = positionAttribute.count(); i < l; i++) {

                    _start.fromBufferAttribute(positionAttribute, i - 1);
                    _end.fromBufferAttribute(positionAttribute, i);

                    lineDistances[i] = lineDistances[i - 1];
                    lineDistances[i] += _start.distanceTo(_end);
                }

                geometry_->setAttribute("lineDistance", BufferAttribute<float>(lineDistances, 1));

            } else {

                std::cerr << "THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry." << std::endl;
            }


            return *this;
        }

        std::string type() const override {
            return "Line";
        }

    protected:
        Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : geometry_(std::move(geometry)), material_(std::move(material)) {}

    private:
        std::shared_ptr<BufferGeometry> geometry_;
        std::shared_ptr<Material> material_;

        inline static Vector3 _start;
        inline static Vector3 _end;

    };

}// namespace threepp

#endif//THREEPP_LINE_HPP
