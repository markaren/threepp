
#include "threepp/helpers/BoxHelper.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;


BoxHelper::BoxHelper(Object3D& object, const Color& color)
    : LineSegments(BufferGeometry::create(), LineBasicMaterial::create()),
      object(&object) {

    std::vector<unsigned int> indices{0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<float> positions(8 * 3);

    geometry_->setIndex(indices);
    geometry_->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    material()->setValues({{"color", color}, {"toneMapped", false}});

    this->matrixAutoUpdate = false;

    this->update();
}


std::string BoxHelper::type() const {

    return "BoxHelper";
}

void BoxHelper::update() {

    static Box3 _box;

    _box.setFromObject(*this->object);

    if (_box.isEmpty()) return;

    const auto& min = _box.min();
    const auto& max = _box.max();

    /*
			5____4
		1/___0/|
		| 6__|_7
		2/___3/

		0: max.x, max.y, max.z
		1: min.x, max.y, max.z
		2: min.x, min.y, max.z
		3: max.x, min.y, max.z
		4: max.x, max.y, min.z
		5: min.x, max.y, min.z
		6: min.x, min.y, min.z
		7: max.x, min.y, min.z
		*/

    const auto position = this->geometry_->getAttribute<float>("position");
    auto& array = position->array();

    // clang-format off
    array[ 0 ] = max.x; array[ 1 ] = max.y; array[ 2 ] = max.z;
    array[ 3 ] = min.x; array[ 4 ] = max.y; array[ 5 ] = max.z;
    array[ 6 ] = min.x; array[ 7 ] = min.y; array[ 8 ] = max.z;
    array[ 9 ] = max.x; array[ 10 ] = min.y; array[ 11 ] = max.z;
    array[ 12 ] = max.x; array[ 13 ] = max.y; array[ 14 ] = min.z;
    array[ 15 ] = min.x; array[ 16 ] = max.y; array[ 17 ] = min.z;
    array[ 18 ] = min.x; array[ 19 ] = min.y; array[ 20 ] = min.z;
    array[ 21 ] = max.x; array[ 22 ] = min.y; array[ 23 ] = min.z;
    // clang-format on

    position->needsUpdate();

    this->geometry_->computeBoundingSphere();
}

std::shared_ptr<BoxHelper> BoxHelper::create(Object3D& object, const Color& color) {

    return std::shared_ptr<BoxHelper>(new BoxHelper(object, color));
}

BoxHelper& BoxHelper::setFromObject(Object3D& object) {

    this->object = &object;
    this->update();

    return *this;
}
