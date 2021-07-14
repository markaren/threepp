//
// Created by LarsIvar on 14.07.2021.
//

#ifndef THREEPP_PLANEHELPER_HPP
#define THREEPP_PLANEHELPER_HPP

#include "threepp/objects/Line.hpp"

namespace threepp {

    class PlaneHelper : public Line {

    public:
        Plane plane;
        float size;

        void updateMatrixWorld(bool force) override {

            auto scale = -plane.constant;
            if (std::abs(scale) < 1e-8) scale = 1e-8f;

            this->scale.set(0.5f * this->size, 0.5f * this->size, scale);

            this->children[0]->material()->side = (scale < 0) ? BackSide : FrontSide; // renderer flips side when determinant < 0; flipping not wanted here

            this->lookAt(this->plane.normal);

            Object3D::updateMatrixWorld(force);
        }

        static std::shared_ptr<PlaneHelper> create(const Plane &plane, float size = 1, unsigned int hex = 0xffff00) {

            return std::shared_ptr<PlaneHelper>(new PlaneHelper(plane, size, hex));
        }

    protected:
        PlaneHelper(const Plane &plane, float size, unsigned int hex)
            : Line(BufferGeometry::create(), LineBasicMaterial::create()), plane(plane), size(size) {

            std::vector<float> positions{1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0};

            geometry_->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            geometry_->computeBoundingSphere();

            auto material = dynamic_cast<LineBasicMaterial *>(material_.get());
            material->color.setHex(hex);
            material->toneMapped = false;

            std::vector<float> positions2{1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1};

            auto geometry2 = BufferGeometry::create();
            geometry2->setAttribute("position", FloatBufferAttribute::create(positions2, 3));
            geometry2->computeBoundingSphere();

            auto material2 = MeshBasicMaterial::create();
            material2->color.setHex(hex);
            material2->opacity = 0.2f;
            material2->transparent = true;
            material2->depthWrite = false;
            material2->toneMapped = false;

            mesh_ = Mesh::create(geometry2, material2);
            this->add(mesh_);
        }

    private:
        std::shared_ptr<Mesh> mesh_;
    };

}// namespace threepp

#endif//THREEPP_PLANEHELPER_HPP
