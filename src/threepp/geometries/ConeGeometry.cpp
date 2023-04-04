
#include "threepp/geometries/ConeGeometry.hpp"

using namespace threepp;

ConeGeometry::ConeGeometry(const ConeGeometry::Params& params)
    : CylinderGeometry(CylinderGeometry::Params(
              0, params.radius, params.height, params.radialSegments,
              params.heightSegments, params.openEnded, params.thetaStart, params.thetaLength)) {}


std::string ConeGeometry::type() const {

    return "ConeGeometry";
}

std::shared_ptr<ConeGeometry> ConeGeometry::create(const ConeGeometry::Params& params) {

    return std::shared_ptr<ConeGeometry>(new ConeGeometry(params));
}

std::shared_ptr<ConeGeometry> ConeGeometry::create(float radius, float height, unsigned int radialSegments, unsigned int heightSegments, bool openEnded, float thetaStart, float thetaLength) {

    return create(Params(radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength));
}

ConeGeometry::Params::Params(float radius, float height, unsigned int radialSegments, unsigned int heightSegments, bool openEnded, float thetaStart, float thetaLength)
    : radius(radius), height(height),
      radialSegments(radialSegments), heightSegments(heightSegments),
      openEnded(openEnded), thetaStart(thetaStart), thetaLength(thetaLength) {}
