
#include "threepp/renderers/gl/GLGeometries.hpp"

#include "threepp/renderers/gl/GLAttributes.hpp"
#include "threepp/renderers/gl/GLBindingStates.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"

#include "threepp/core/InstancedBufferGeometry.hpp"

#include <glad/glad.h>

#include <unordered_map>

using namespace threepp;
using namespace threepp::gl;

struct GLGeometries::Impl {

    struct OnGeometryDispose: EventListener {

        explicit OnGeometryDispose(GLGeometries::Impl* scope)
            : scope_(scope) {}

        void onEvent(Event& event) override {

            auto geometry = static_cast<BufferGeometry*>(event.target);

            if (geometry->hasIndex()) {

                scope_->attributes_.remove(geometry->getIndex());
            }

            for (const auto& [name, value] : geometry->getAttributes()) {

                scope_->attributes_.remove(value.get());
            }

            geometry->removeEventListener("dispose", this);

            scope_->geometries_.erase(geometry);


            if (scope_->wireframeAttributes_.count(geometry)) {

                const auto& attribute = scope_->wireframeAttributes_.at(geometry);

                scope_->attributes_.remove(attribute.get());
                scope_->wireframeAttributes_.erase(geometry);
            }

            scope_->bindingStates_.releaseStatesOfGeometry(geometry);

            if (auto ig = dynamic_cast<InstancedBufferGeometry*>(geometry)) {
                ig->_maxInstanceCount = 0;
            }

            --scope_->info_.memory.geometries;
        }

    private:
        GLGeometries::Impl* scope_;
    };

    GLInfo& info_;
    GLAttributes& attributes_;
    GLBindingStates& bindingStates_;

    OnGeometryDispose onGeometryDispose_;

    std::unordered_map<BufferGeometry*, bool> geometries_;
    std::unordered_map<BufferGeometry*, std::unique_ptr<IntBufferAttribute>> wireframeAttributes_;

    Impl(GLAttributes& attributes, GLInfo& info, GLBindingStates& bindingStates)
        : info_(info),
          attributes_(attributes),
          bindingStates_(bindingStates),
          onGeometryDispose_(this) {}

    void get(Object3D* object, BufferGeometry* geometry) {

        if (geometries_.count(geometry) && geometries_.at(geometry)) return;

        geometry->addEventListener("dispose", &onGeometryDispose_);

        geometries_[geometry] = true;

        ++info_.memory.geometries;
    }

    void update(BufferGeometry* geometry) {

        auto& geometryAttributes = geometry->getAttributes();

        // Updating index buffer in VAO now. See WebGLBindingStates.

        for (auto& [name, attribute] : geometryAttributes) {

            attributes_.update(attribute.get(), GL_ARRAY_BUFFER);
        }

        // morph targets

        auto& morphAttributes = geometry->getMorphAttributes();

        for (auto& [name, value] : morphAttributes) {

            auto& array = morphAttributes.at(name);

            for (const auto& attribute : array) {

                attributes_.update(attribute.get(), GL_ARRAY_BUFFER);
            }
        }
    }

    void updateWireframeAttribute(BufferGeometry* geometry) {

        std::vector<unsigned int> indices;

        const auto geometryIndex = geometry->getIndex();
        const auto geometryPosition = geometry->getAttribute<float>("position");
        unsigned int version = 0;

        if (geometryIndex != nullptr) {

            const auto& array = geometryIndex->array();
            version = geometryIndex->version;

            for (unsigned i = 0, l = array.size(); i < l; i += 3) {

                const auto a = array[i + 0];
                const auto b = array[i + 1];
                const auto c = array[i + 2];

                indices.insert(indices.end(), {a, b, b, c, c, a});
            }

        } else {

            const auto& array = geometryPosition->array();
            version = geometryPosition->version;

            for (unsigned i = 0, l = array.size() / 3 - 1; i < l; i += 3) {

                const auto a = i + 0;
                const auto b = i + 1;
                const auto c = i + 2;

                indices.insert(indices.end(), {a, b, b, c, c, a});
            }
        }

        auto attribute = IntBufferAttribute::create(indices, 1);
        attribute->version = version;

        // Updating index buffer in VAO now. See WebGLBindingStates

        if (wireframeAttributes_.count(geometry)) {
            auto previousAttribute = wireframeAttributes_.at(geometry).get();
            attributes_.remove(previousAttribute);
        }

        wireframeAttributes_[geometry] = std::move(attribute);
    }

    IntBufferAttribute* getWireframeAttribute(BufferGeometry* geometry) {

        if (wireframeAttributes_.count(geometry)) {

            const auto& currentAttribute = wireframeAttributes_.at(geometry);

            if (geometry->hasIndex()) {

                auto geometryIndex = geometry->getIndex();

                // if the attribute is obsolete, create a new one

                if (currentAttribute->version < geometryIndex->version) {

                    updateWireframeAttribute(geometry);
                }
            }

        } else {

            updateWireframeAttribute(geometry);
        }

        return wireframeAttributes_.at(geometry).get();
    }
};


GLGeometries::GLGeometries(GLAttributes& attributes, GLInfo& info, GLBindingStates& bindingStates)
    : pimpl_(std::make_unique<Impl>(attributes, info, bindingStates)) {}

void GLGeometries::get(Object3D* object, BufferGeometry* geometry) {

    pimpl_->get(object, geometry);
}

void GLGeometries::update(BufferGeometry* geometry) {

    pimpl_->update(geometry);
}

void GLGeometries::updateWireframeAttribute(BufferGeometry* geometry) {

    pimpl_->updateWireframeAttribute(geometry);
}

IntBufferAttribute* GLGeometries::getWireframeAttribute(BufferGeometry* geometry) {

    return pimpl_->getWireframeAttribute(geometry);
}

gl::GLGeometries::~GLGeometries() = default;
