
#ifndef THREEPP_BULLETENGINE_HPP
#define THREEPP_BULLETENGINE_HPP


#include <threepp/objects/Mesh.hpp>

#include <memory>


class BulletEngine {

public:
    explicit BulletEngine(float gravity = -9.81f);

    void register_mesh(std::shared_ptr<threepp::Mesh> m, float mass);

    void step(float dt);

    ~BulletEngine();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

#endif//THREEPP_BULLETENGINE_HPP
