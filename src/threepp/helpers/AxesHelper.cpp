
#include "threepp/helpers/AxesHelper.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;

AxesHelper::AxesHelper(float size): LineSegments(BufferGeometry::create(), LineBasicMaterial::create()) {

    std::vector<float> vertices{
            0, 0, 0, size, 0, 0,
            0, 0, 0, 0, size, 0,
            0, 0, 0, 0, 0, size};

    std::vector<float> colors{
            1, 0, 0, 1, 0.6f, 0,
            0, 1, 0, 0.6f, 1, 0,
            0, 0, 1, 0, 0.6f, 1};

    geometry_->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    geometry_->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    materials_.front()->vertexColors = true;
}

std::shared_ptr<AxesHelper> AxesHelper::create(float size) {

    return std::shared_ptr<AxesHelper>(new AxesHelper(size));
}
