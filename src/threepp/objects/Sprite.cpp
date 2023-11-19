
#include "threepp/objects/Sprite.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"

#include "threepp/core/InterleavedBufferAttribute.hpp"
#include "threepp/core/Raycaster.hpp"

#include "threepp/math/Triangle.hpp"

#include <cmath>
#include <memory>

using namespace threepp;

namespace {

    Vector3 _intersectPoint;
    Vector3 _worldScale;
    Vector3 _mvPosition;

    Vector2 _alignedPosition;
    Vector2 _rotatedPosition;
    Matrix4 _viewWorldMatrix;

    Vector3 _vA;
    Vector3 _vB;
    Vector3 _vC;

    Vector2 _uvA;
    Vector2 _uvB;
    Vector2 _uvC;


    void transformVertex(Vector3& vertexPosition, const Vector3& mvPosition, const Vector2& center, const Vector3& scale, const std::optional<std::pair<float, float>>& sincos) {

        // compute position in camera space
        _alignedPosition.subVectors(vertexPosition, center).addScalar(0.5f).multiply(scale);

        // to check if rotation is not zero
        if (sincos) {

            float sin = sincos->first;
            float cos = sincos->second;

            _rotatedPosition.x = (cos * _alignedPosition.x) - (sin * _alignedPosition.y);
            _rotatedPosition.y = (sin * _alignedPosition.x) + (cos * _alignedPosition.y);

        } else {

            _rotatedPosition.copy(_alignedPosition);
        }


        vertexPosition.copy(mvPosition);
        vertexPosition.x += _rotatedPosition.x;
        vertexPosition.y += _rotatedPosition.y;

        // transform to world space
        vertexPosition.applyMatrix4(_viewWorldMatrix);
    }
}// namespace


Sprite::Sprite(const std::shared_ptr<SpriteMaterial>& material)
    : material(material),
      _geometry(new BufferGeometry()) {

    std::vector<float> float32Array{
            -0.5f, -0.5f, 0.f, 0.f, 0.f,
            0.5f, -0.5f, 0.f, 1.f, 0.f,
            0.5f, 0.5f, 0.f, 1.f, 1.f,
            -0.5f, 0.5f, 0.f, 0.f, 1.f};

    auto interleavedBuffer = InterleavedBuffer::create(float32Array, 5);
    _geometry->setIndex(std::vector<int>{0, 1, 2, 0, 2, 3});
    _geometry->setAttribute("position", std::make_unique<InterleavedBufferAttribute>(interleavedBuffer, 3, 0, false));
    _geometry->setAttribute("uv", std::make_unique<InterleavedBufferAttribute>(interleavedBuffer, 3, 0, false));
}

std::string Sprite::type() const {

    return "Sprite";
}

BufferGeometry* Sprite::geometry() {

    return _geometry.get();
}

std::shared_ptr<Sprite> Sprite::create(const std::shared_ptr<SpriteMaterial>& material) {

    return std::make_shared<Sprite>(material);
}

void Sprite::raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) {

    if (!raycaster.camera) {

        throw std::runtime_error("THREE.Sprite: 'Raycaster.camera' needs to be set in order to raycast against sprites.");
    }

    _worldScale.setFromMatrixScale(*this->matrixWorld);

    _viewWorldMatrix.copy(*raycaster.camera->matrixWorld);
    this->modelViewMatrix.multiplyMatrices(raycaster.camera->matrixWorldInverse, *this->matrixWorld);

    _mvPosition.setFromMatrixPosition(this->modelViewMatrix);

    if (raycaster.camera->is<PerspectiveCamera>() && !this->material->sizeAttenuation) {

        _worldScale.multiplyScalar(-_mvPosition.z);
    }

    float rotation = material->rotation;
    std::optional<std::pair<float, float>> sincos;

    if (rotation != 0) {
        sincos = std::make_pair(std::sin(rotation), std::cos(rotation));
    }

    transformVertex(_vA.set(-0.5f, -0.5f, 0.f), _mvPosition, center, _worldScale, sincos);
    transformVertex(_vB.set(0.5f, -0.5f, 0.f), _mvPosition, center, _worldScale, sincos);
    transformVertex(_vC.set(0.5f, 0.5f, 0.f), _mvPosition, center, _worldScale, sincos);

    _uvA.set(0, 0);
    _uvB.set(1, 0);
    _uvC.set(1, 1);

    // check first triangle
    auto intersect = raycaster.ray.intersectTriangle(_vA, _vB, _vC, false, _intersectPoint);

    if (!intersect) {

        // check second triangle
        transformVertex(_vB.set(-0.5f, 0.5f, 0.f), _mvPosition, center, _worldScale, sincos);
        _uvB.set(0, 1);

        intersect = raycaster.ray.intersectTriangle(_vA, _vC, _vB, false, _intersectPoint);
        if (!intersect) {

            return;
        }
    }

    auto distance = raycaster.ray.origin.distanceTo(_intersectPoint);

    if (distance < raycaster.near || distance > raycaster.far) return;

    Intersection intersection{};
    intersection.distance = distance;
    intersection.object = this;
    intersection.point = _intersectPoint;
    intersection.uv = Vector2();
    Triangle::getUV(_intersectPoint, _vA, _vB, _vC, _uvA, _uvB, _uvC, *intersection.uv);
    intersects.emplace_back(intersection);
}
