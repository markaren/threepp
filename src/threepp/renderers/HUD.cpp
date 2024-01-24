
#include "threepp/renderers/HUD.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/geometries/TextGeometry.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;

struct HUD::Impl {
    explicit Impl(threepp::GLRenderer& renderer)
        : renderer(&renderer) {}


    void render() {
        renderer->clearDepth();
        renderer->render(scene, camera);
    }

    void addText(const std::string& str) {
        //            TextGeometry::Options opts()
        //        auto geometry = TextGeometry::create(str, )
    }

    Scene scene;
    OrthographicCamera camera;

    threepp::GLRenderer* renderer;

    std::vector<std::shared_ptr<Mesh>> meshes_;
};

HUD::HUD(threepp::GLRenderer& renderer)
    : pimpl_(std::make_unique<Impl>(renderer)) {}

HUD::~HUD() = default;
