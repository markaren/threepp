// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/controls/TransformControls.js

#ifndef THREEPP_TRANSFORMCONTROLS_HPP
#define THREEPP_TRANSFORMCONTROLS_HPP

#include "threepp/core/Object3D.hpp"

#include <memory>

namespace threepp {

    class Camera;
    class PeripheralsEventSource;

    class TransformControls: public Object3D {

    public:

        bool enabled = true;
        bool showX = true;
        bool showY = true;
        bool showZ = true;

        TransformControls(Camera& camera, PeripheralsEventSource& canvas);

        void setSpace(const std::string& space);

        [[nodiscard]] std::string getSpace() const;

        void setMode(const std::string& mode);

        void setSize(float size);

        void setTranslationSnap(std::optional<float> snap);

        void setRotationSnap(std::optional<float> snap);

        void setScaleSnap(std::optional<float> snap);

        [[nodiscard]] bool isDragging() const;

        TransformControls& attach(Object3D& object);

        TransformControls& detach();

        void updateMatrixWorld(bool force) override;

        ~TransformControls() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TRANSFORMCONTROLS_HPP
