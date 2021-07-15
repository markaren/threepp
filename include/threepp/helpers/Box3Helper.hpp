// https://github.com/mrdoob/three.js/blob/dev/src/helpers/Box3Helper.js

#ifndef THREEPP_BOX3HELPER_HPP
#define THREEPP_BOX3HELPER_HPP

#include "threepp/objects/LineSegments.hpp"

namespace threepp {

    class Box3Helper : public LineSegments {

    public:

        void updateMatrixWorld(bool force) override {
            if (box.isEmpty()) return;

            box.getCenter(this->position);

            box.getSize(this->scale);

            this->scale.multiplyScalar(0.5);

            LineSegments::updateMatrixWorld(force);
        }

        static std::shared_ptr<Box3Helper> create(const Box3 &box, unsigned int color = 0xffff00) {

            return std::shared_ptr<Box3Helper>(new Box3Helper(box, color));
        }

    protected:
        Box3Helper(const Box3 &box, unsigned int color)
            : LineSegments(BufferGeometry::create(), LineBasicMaterial::create()), box(box) {

            std::vector<int> indices{0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};

            std::vector<float> positions{1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1};

            auto lineMaterial = dynamic_cast<LineBasicMaterial *>(material_.get());
            lineMaterial->color.setHex(color);
            lineMaterial->toneMapped = false;

            geometry_->setIndex(IntBufferAttribute::create(indices, 1));

            geometry_->setAttribute("position", FloatBufferAttribute::create(positions, 3));

            geometry_->computeBoundingSphere();
        }

    private:
        const Box3& box;
    };

}// namespace threepp

#endif//THREEPP_BOX3HELPER_HPP
