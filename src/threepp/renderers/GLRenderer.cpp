
#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/renderers/GLRenderTarget.hpp"

#include "threepp/renderers/gl/GLAttributes.hpp"
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLBindingStates.hpp"
#include "threepp/renderers/gl/GLBufferRenderer.hpp"
#include "threepp/renderers/gl/GLGeometries.hpp"
#include "threepp/renderers/gl/GLMaterials.hpp"
#include "threepp/renderers/gl/GLMorphTargets.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLPrograms.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"
#include "threepp/renderers/gl/GLRenderStates.hpp"
#include "threepp/renderers/gl/GLTextures.hpp"
#include "threepp/renderers/gl/GLUtils.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/core/InstancedBufferGeometry.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/math/Frustum.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"

#include <glad/glad.h>

#include <cmath>


using namespace threepp;


struct GLRenderer::Impl {

    struct OnMaterialDispose: EventListener {

        explicit OnMaterialDispose(GLRenderer::Impl* scope): scope_(scope) {}

        void onEvent(Event& event) override {

            auto material = static_cast<Material*>(event.target);

            material->removeEventListener("dispose", this);

            scope_->deallocateMaterial(material);
        }

    private:
        GLRenderer::Impl* scope_;
    };

    GLRenderer& scope;

    gl::GLState state;


    Scene _emptyScene;

    OnMaterialDispose onMaterialDispose;

    std::shared_ptr<gl::GLRenderList> currentRenderList;
    std::shared_ptr<gl::GLRenderState> currentRenderState;

    std::vector<std::shared_ptr<gl::GLRenderList>> renderListStack;
    std::vector<std::shared_ptr<gl::GLRenderState>> renderStateStack;

    int _currentActiveCubeFace = 0;
    int _currentActiveMipmapLevel = 0;
    GLRenderTarget* _currentRenderTarget = nullptr;
    std::optional<unsigned int> _currentMaterialId;

    Camera* _currentCamera = nullptr;
    Vector4 _currentViewport;
    Vector4 _currentScissor;
    std::optional<bool> _currentScissorTest;

    //

    WindowSize _size;

    int _pixelRatio = 1;

    Vector4 _viewport;
    Vector4 _scissor;
    bool _scissorTest = false;

    std::vector<unsigned int> _currentDrawBuffers;

    // frustum

    Frustum _frustum;

    // clipping

    bool _clippingEnabled = false;
    bool _localClippingEnabled = false;

    // camera matrices cache

    Matrix4 _projScreenMatrix;

    Vector3 _vector3;

    gl::GLInfo _info;

    gl::GLBackground background;
    gl::GLProperties properties;
    gl::GLGeometries geometries;
    gl::GLBindingStates bindingStates;
    gl::GLAttributes attributes;
    gl::GLClipping clipping;
    gl::GLTextures textures;
    gl::GLMaterials materials;
    gl::GLRenderStates renderStates;
    gl::GLRenderLists renderLists;
    gl::GLObjects objects;
    gl::GLMorphTargets morphTargets;
    gl::GLPrograms programCache;

    std::unique_ptr<gl::GLBufferRenderer> bufferRenderer;
    std::unique_ptr<gl::GLIndexedBufferRenderer> indexedBufferRenderer;

    gl::GLShadowMap shadowMap;

    Impl(GLRenderer& scope, WindowSize size, const GLRenderer::Parameters& parameters)
        : scope(scope), _size(size),
          _viewport(0, 0, _size.width, _size.height),
          _scissor(0, 0, _size.width, _size.height),
          background(state, parameters.premultipliedAlpha),
          bufferRenderer(std::make_unique<gl::GLBufferRenderer>(_info)),
          indexedBufferRenderer(std::make_unique<gl::GLIndexedBufferRenderer>(_info)),
          clipping(properties),
          bindingStates(attributes),
          geometries(attributes, _info, bindingStates),
          textures(state, properties, _info),
          objects(geometries, attributes, _info),
          renderLists(properties),
          shadowMap(objects),
          materials(properties),
          programCache(bindingStates, clipping),
          _currentDrawBuffers(GL_BACK),
          onMaterialDispose(this) {}

    void deallocateMaterial(Material* material) {

        releaseMaterialProgramReferences(material);

        properties.materialProperties.remove(material->uuid());
    }

    void releaseMaterialProgramReferences(Material* material) {

        auto& programs = properties.materialProperties.get(material->uuid())->programs;

        if (!programs.empty()) {

            for (auto& [key, program] : programs) {

                programCache.releaseProgram(program);
            }
        }
    }

    void render(Scene* scene, Camera* camera) {

        // update scene graph

        if (scene->autoUpdate) scene->updateMatrixWorld();

        // update camera matrices and frustum

        if (camera->parent == nullptr) camera->updateMatrixWorld();

        //
        //    if ( scene.isScene === true ) scene.onBeforeRender( _this, scene, camera, _currentRenderTarget );

        currentRenderState = renderStates.get(scene, renderStateStack.size());
        currentRenderState->init();

        renderStateStack.emplace_back(currentRenderState);

        _projScreenMatrix.multiplyMatrices(camera->projectionMatrix, camera->matrixWorldInverse);
        _frustum.setFromProjectionMatrix(_projScreenMatrix);

        _localClippingEnabled = scope.localClippingEnabled;
        _clippingEnabled = clipping.init(scope.clippingPlanes, _localClippingEnabled, camera);

        currentRenderList = renderLists.get(scene, renderListStack.size());
        currentRenderList->init();

        renderListStack.emplace_back(currentRenderList);

        projectObject(scene, camera, 0, scope.sortObjects);

        currentRenderList->finish();

        if (scope.sortObjects) {

            currentRenderList->sort();
        }

        //

        if (_clippingEnabled) clipping.beginShadows();

        auto& shadowsArray = currentRenderState->getShadowsArray();

        shadowMap.render(scope, shadowsArray, scene, camera);

        currentRenderState->setupLights();
        currentRenderState->setupLightsView(camera);

        if (_clippingEnabled) clipping.endShadows();

        //

        if (this->_info.autoReset) this->_info.reset();

        //

        background.render(scope, scene);

        // render scene

        auto& opaqueObjects = currentRenderList->opaque;
        auto& transparentObjects = currentRenderList->transparent;
        //
        if (!opaqueObjects.empty()) renderObjects(opaqueObjects, scene, camera);
        if (!transparentObjects.empty()) renderObjects(transparentObjects, scene, camera);

        //

        if (_currentRenderTarget) {

            // Generate mipmap if we're using any kind of mipmap filtering

            textures.updateRenderTargetMipmap(_currentRenderTarget);
        }

        //

        //    if ( scene.isScene === true ) scene.onAfterRender( _this, scene, camera );

        // Ensure depth buffer writing is enabled so it can be cleared on next render

        state.depthBuffer.setTest(true);
        state.depthBuffer.setMask(true);
        state.colorBuffer.setMask(true);
        //
        state.setPolygonOffset(false);

        // finish

        _currentMaterialId = std::nullopt;
        _currentCamera = nullptr;

        renderStateStack.pop_back();

        if (!renderStateStack.empty()) {

            currentRenderState = renderStateStack.back();

        } else {

            currentRenderState = nullptr;
        }

        renderListStack.pop_back();

        if (!renderListStack.empty()) {

            currentRenderList = renderListStack.back();

        } else {

            currentRenderList = nullptr;
        }
    }

    void renderBufferDirect(Camera* camera, Scene* _scene, BufferGeometry* geometry, Material* material, Object3D* object, std::optional<GeometryGroup> group) {

        auto scene = _scene;

        if (scene == nullptr) {
            scene = &_emptyScene;
        }

        bool isMesh = object->is<Mesh>();
        const auto frontFaceCW = (isMesh && object->matrixWorld->determinant() < 0);

        auto program = setProgram(camera, scene, material, object);

        state.setMaterial(material, frontFaceCW);

        //

        auto index = geometry->getIndex();
        const auto& position = geometry->getAttribute<float>("position");

        //

        if (index == nullptr) {

            if (!geometry->hasAttribute("position") || position->count() == 0) return;

        } else if (index->count() == 0) {

            return;
        }

        //

        int rangeFactor = 1;

        auto wireframeMaterial = dynamic_cast<MaterialWithWireframe*>(material);
        bool isWireframeMaterial = wireframeMaterial != nullptr;

        if (isWireframeMaterial && wireframeMaterial->wireframe) {

            index = geometries.getWireframeAttribute(geometry);
            rangeFactor = 2;
        }

        if (auto m = material->as<MaterialWithMorphTargets>()) {
            if (m->morphTargets || m->morphNormals) {
                morphTargets.update(object, geometry, material, program);
            }
        }

        bindingStates.setup(object, material, program, geometry, index);

        gl::BufferRenderer* renderer = bufferRenderer.get();

        if (index != nullptr) {

            const auto& attribute = attributes.get(index);

            renderer = indexedBufferRenderer.get();
            indexedBufferRenderer->setIndex(attribute);
        }

        //

        const auto dataCount = (index != nullptr) ? index->count() : position->count();

        const auto rangeStart = geometry->drawRange.start * rangeFactor;
        const auto rangeCount = geometry->drawRange.count * rangeFactor;

        const auto groupStart = group ? group->start * rangeFactor : 0;
        const auto groupCount = group ? group->count * rangeFactor : std::numeric_limits<int>::max();

        const auto drawStart = std::max(rangeStart, groupStart);
        const auto drawEnd = std::min(dataCount, std::min(rangeStart + rangeCount, groupStart + groupCount)) - 1;

        const auto drawCount = std::max(0, drawEnd - drawStart + 1);

        if (drawCount == 0) return;

        //

        if (isMesh) {

            if (isWireframeMaterial && wireframeMaterial->wireframe) {

                state.setLineWidth(wireframeMaterial->wireframeLinewidth * static_cast<float>(scope.getTargetPixelRatio()));
                renderer->setMode(GL_LINES);

            } else {

                renderer->setMode(GL_TRIANGLES);
            }

        } else if (object->is<Line>()) {

            float lineWidth = 1;
            if (auto lw = material->as<MaterialWithLineWidth>()) {
                lineWidth = lw->linewidth;
            }

            state.setLineWidth(lineWidth * static_cast<float>(scope.getTargetPixelRatio()));

            if (object->is<LineSegments>()) {

                renderer->setMode(GL_LINES);

            } else if (object->is<LineLoop>()) {

                renderer->setMode(GL_LINE_LOOP);

            } else {

                renderer->setMode(GL_LINE_STRIP);
            }

        } else if (object->is<Points>()) {

            renderer->setMode(GL_POINTS);

        } else if (object->is<Sprite>()) {

            renderer->setMode(GL_TRIANGLES);
        }

        if (auto im = object->as<InstancedMesh>()) {

            renderer->renderInstances(drawStart, drawCount, im->count);

        } else if (auto g = dynamic_cast<InstancedBufferGeometry*>(geometry)) {

            const auto instanceCount = std::min(g->instanceCount, g->_maxInstanceCount);

            renderer->renderInstances(drawStart, drawCount, instanceCount);

        } else {

            renderer->render(drawStart, drawCount);
        }
    }

    void projectObject(Object3D* object, Camera* camera, unsigned int groupOrder, bool sortObjects) {
        if (!object->visible) return;

        bool visible = object->layers.test(camera->layers);

        if (visible) {

            if (object->is<Group>()) {

                groupOrder = object->renderOrder;

            } else if (auto lod = object->as<LOD>()) {

                if (lod->autoUpdate) lod->update(*camera);

            } else if (auto light = object->as<Light>()) {

                currentRenderState->pushLight(light);

                if (light->castShadow) {

                    currentRenderState->pushShadow(light);
                }

            } else if (auto sprite = object->as<Sprite>()) {

                if (!object->frustumCulled || _frustum.intersectsSprite(*sprite)) {

                    if (sortObjects) {

                        _vector3.setFromMatrixPosition(*sprite->matrixWorld)
                                .applyMatrix4(_projScreenMatrix);
                    }

                    auto geometry = objects.update(object);
                    auto material = sprite->material;

                    if (material->visible) {

                        currentRenderList->push(object, geometry, material.get(), groupOrder, _vector3.z, std::nullopt);
                    }
                }

            } else if (object->is<Mesh>() || object->is<Line>() || object->is<Points>()) {

                if (auto skinned = object->as<SkinnedMesh>()) {

                    // update skeleton only once in a frame

                    if (skinned->skeleton->frame != _info.render.frame) {

                        skinned->skeleton->update();
                        skinned->skeleton->frame = _info.render.frame;
                    }
                }

                if (!object->frustumCulled || _frustum.intersectsObject(*object)) {

                    if (sortObjects) {

                        _vector3.setFromMatrixPosition(*object->matrixWorld)
                                .applyMatrix4(_projScreenMatrix);
                    }

                    auto geometry = objects.update(object);
                    const auto& materials = object->materials();

                    if (materials.size() > 1) {

                        const auto& groups = geometry->groups;

                        for (const auto& group : groups) {

                            Material* groupMaterial = materials.at(group.materialIndex);

                            if (groupMaterial && groupMaterial->visible) {

                                currentRenderList->push(object, geometry, groupMaterial, groupOrder, _vector3.z, group);
                            }
                        }

                    } else if (materials.front()->visible) {

                        currentRenderList->push(object, geometry, materials.front(), groupOrder, _vector3.z, std::nullopt);
                    }
                }
            }
        }

        for (const auto& child : object->children) {

            projectObject(child, camera, groupOrder, sortObjects);
        }
    }

    void renderObjects(const std::vector<gl::RenderItem*>& renderList, Scene* scene, Camera* camera) {

        auto& overrideMaterial = scene->overrideMaterial;

        for (const auto& renderItem : renderList) {

            auto object = renderItem->object;
            auto geometry = renderItem->geometry;
            auto material = overrideMaterial == nullptr ? renderItem->material : overrideMaterial.get();
            auto group = renderItem->group;

            renderObject(object, scene, camera, geometry, material, group);
        }
    }

    void renderObject(Object3D* object, Scene* scene, Camera* camera, BufferGeometry* geometry, Material* material, std::optional<GeometryGroup> group) {

        if (object->onBeforeRender) {

            object->onBeforeRender.value()(&scope, scene, camera, geometry, material, group);
        }

        object->modelViewMatrix.multiplyMatrices(camera->matrixWorldInverse, *object->matrixWorld);
        object->normalMatrix.getNormalMatrix(object->modelViewMatrix);

        renderBufferDirect(camera, scene, geometry, material, object, group);

        if (object->onAfterRender) {

            object->onAfterRender.value()(&scope, scene, camera, geometry, material, group);
        }
    }

    gl::GLProgram* getProgram(Material* material, Scene* scene, Object3D* object) {

        //    bool isScene = instanceof <Scene>(scene);
        //
        //    if (!isScene) scene = &_emptyScene;// scene could be a Mesh, Line, Points, ...

        auto materialProperties = properties.materialProperties.get(material->uuid());

        auto& lights = currentRenderState->getLights();
        auto& shadowsArray = currentRenderState->getShadowsArray();

        auto lightsStateVersion = lights.state.version;

        auto parameters = gl::GLPrograms::getParameters(scope, clipping, material, lights.state, shadowsArray.size(), scene, object);
        auto programCacheKey = gl::GLPrograms::getProgramCacheKey(scope, parameters);

        auto& programs = materialProperties->programs;

        // always update environment and fog - changing these trigger an getProgram call, but it's possible that the program doesn't change

        materialProperties->environment = material->as<MeshStandardMaterial>() ? scene->environment : nullptr;
        materialProperties->fog = scene->fog;
        //    materialProperties.envMap = cubemaps.get( material.envMap || materialProperties.environment );

        if (programs.empty()) {

            // new material

            material->addEventListener("dispose", &onMaterialDispose);
        }

        std::shared_ptr<gl::GLProgram> program = nullptr;

        if (programs.count(programCacheKey)) {

            program = programs.at(programCacheKey);

            if (materialProperties->currentProgram == program && materialProperties->lightsStateVersion == lightsStateVersion) {

                updateCommonMaterialProperties(material, parameters);

                return program.get();
            }

        } else {

            parameters.uniforms = gl::GLPrograms::getUniforms(material);

            // material.onBuild( parameters, this );

            // material.onBeforeCompile( parameters, this );

            program = programCache.acquireProgram(scope, parameters, programCacheKey);
            programs[programCacheKey] = program;

            materialProperties->uniforms = parameters.uniforms;
        }

        auto& uniforms = *materialProperties->uniforms;

        if (!material->is<ShaderMaterial>() && !material->is<RawShaderMaterial>() || material->clipping) {

            uniforms["clippingPlanes"] = clipping.uniform;
        }

        updateCommonMaterialProperties(material, parameters);

        // store the light setup it was created for

        materialProperties->needsLights = materialNeedsLights(material);
        materialProperties->lightsStateVersion = lightsStateVersion;

        if (materialProperties->needsLights) {

            // wire up the material to this renderer's lighting state

            uniforms.at("ambientLightColor").setValue(lights.state.ambient);
            uniforms.at("lightProbe").setValue(lights.state.probe);
            uniforms.at("directionalLights").setValue(lights.state.directional);
            uniforms.at("directionalLightShadows").setValue(lights.state.directionalShadow);
            uniforms.at("spotLights").setValue(lights.state.spot);
            uniforms.at("spotLightShadows").setValue(lights.state.spotShadow);
            uniforms.at("pointLights").setValue(lights.state.point);
            uniforms.at("pointLightShadows").setValue(lights.state.pointShadow);
            uniforms.at("hemisphereLights").setValue(lights.state.hemi);

            uniforms.at("directionalShadowMap").setValue(lights.state.directionalShadowMap);
            uniforms.at("directionalShadowMatrix").setValue(lights.state.directionalShadowMatrix);
            uniforms.at("spotShadowMap").setValue(lights.state.spotShadowMap);
            uniforms.at("spotShadowMatrix").setValue(lights.state.spotShadowMatrix);
            uniforms.at("pointShadowMap").setValue(lights.state.pointShadowMap);
            uniforms.at("pointShadowMatrix").setValue(lights.state.pointShadowMatrix);
        }

        auto progUniforms = program->getUniforms();
        auto uniformsList = gl::GLUniforms::seqWithValue(progUniforms->seq, *materialProperties->uniforms);

        materialProperties->currentProgram = program;
        materialProperties->uniformsList = uniformsList;

        return materialProperties->currentProgram.get();
    }

    void updateCommonMaterialProperties(Material* material, gl::ProgramParameters& parameters) {

        auto materialProperties = properties.materialProperties.get(material->uuid());

        materialProperties->outputEncoding = parameters.outputEncoding;
        materialProperties->instancing = parameters.instancing;
        materialProperties->skinning = parameters.skinning;
        materialProperties->numClippingPlanes = parameters.numClippingPlanes;
        materialProperties->numIntersection = parameters.numClipIntersection;
        materialProperties->vertexAlphas = parameters.vertexAlphas;
    }

    gl::GLProgram* setProgram(Camera* camera, Scene* scene, Material* material, Object3D* object) {

        //    bool isScene = instanceof <Scene>(scene);

        //            if (!isScene) scene = _emptyScene;// scene could be a Mesh, Line, Points, ...
        //

        bool isMeshBasicMaterial = material->type() == "MeshBasicMaterial";
        bool isMeshLambertMaterial = material->type() == "MeshLambertMaterial";
        bool isMeshToonMaterial = material->type() == "MeshToonMaterial";
        bool isMeshPhongMaterial = material->type() == "MeshPhongMaterial";
        bool isMeshStandardMaterial = material->type() == "MeshStandardMaterial";
        bool isShadowMaterial = material->type() == "ShadowMaterial";
        bool isShaderMaterial = material->is<ShaderMaterial>();
        bool isEnvMap = material->is<MaterialWithEnvMap>() && material->as<MaterialWithEnvMap>()->envMap;

        textures.resetTextureUnits();

        auto fog = scene->fog;
        auto environment = isMeshStandardMaterial ? scene->environment : nullptr;
        Encoding encoding = (_currentRenderTarget == nullptr) ? scope.outputEncoding : _currentRenderTarget->texture->encoding;
        //                const envMap = cubemaps.get(material.envMap || environment);
        bool vertexAlphas = material->vertexColors &&
                            object->geometry() &&
                            object->geometry()->hasAttribute("color") &&
                            object->geometry()->getAttribute<float>("color")->itemSize() == 4;

        auto materialProperties = properties.materialProperties.get(material->uuid());
        auto& lights = currentRenderState->getLights();

        if (_clippingEnabled) {

            if (_localClippingEnabled || camera != _currentCamera) {

                bool useCache = camera == _currentCamera && material->id == _currentMaterialId.value_or(-1);

                // we might want to call this function with some ClippingGroup
                // object instead of the material, once it becomes feasible
                // (#8465, #8379)
                clipping.setState(material, camera, useCache);
            }
        }

        //

        bool needsProgramChange = false;
        bool isInstancedMesh = object->type() == "InstancedMesh";
        bool isSkinnedMesh = object->type() == "SkinnedMesh";

        if (material->version == materialProperties->version) {

            if (materialProperties->needsLights && (materialProperties->lightsStateVersion != lights.state.version)) {

                needsProgramChange = true;

            } else if (materialProperties->outputEncoding != encoding) {

                needsProgramChange = true;

            } else if (isInstancedMesh && !materialProperties->instancing) {

                needsProgramChange = true;

            } else if (!isInstancedMesh && materialProperties->instancing) {

                needsProgramChange = true;

            } else if (isSkinnedMesh && !materialProperties->skinning) {

                needsProgramChange = true;

            } else if (!isSkinnedMesh && materialProperties->skinning) {

                needsProgramChange = true;

            } else if (false /*materialProperties.envMap != envMap*/) {

                needsProgramChange = true;

            } else if (fog && material->fog && materialProperties->fog && !(fog.value() == materialProperties->fog.value())) {

                needsProgramChange = true;

            } else if (materialProperties->numClippingPlanes &&
                       (materialProperties->numClippingPlanes.value() != clipping.numPlanes ||
                        materialProperties->numIntersection != clipping.numIntersection)) {

                needsProgramChange = true;

            } else if (materialProperties->vertexAlphas != vertexAlphas) {

                needsProgramChange = true;
            }

        } else {

            needsProgramChange = true;
            materialProperties->version = material->version;
        }

        //

        gl::GLProgram* program = materialProperties->currentProgram.get();

        if (needsProgramChange) {

            program = getProgram(material, scene, object);
        }

        bool refreshProgram = false;
        bool refreshMaterial = false;
        bool refreshLights = false;

        auto p_uniforms = program->getUniforms();
        auto& m_uniforms = *materialProperties->uniforms;

        if (state.useProgram(program->program)) {

            refreshProgram = true;
            refreshMaterial = true;
            refreshLights = true;
        }

        if (material->id != _currentMaterialId.value_or(-1)) {

            _currentMaterialId = material->id;

            refreshMaterial = true;
        }

        if (refreshProgram || _currentCamera != camera) {

            p_uniforms->setValue("projectionMatrix", camera->projectionMatrix);

            if (gl::GLCapabilities::instance().logarithmicDepthBuffer) {

                p_uniforms->setValue("logDepthBufFC", 2.f / (std::log(camera->far + 1.f) / math::LN2));
            }

            if (_currentCamera != camera) {

                _currentCamera = camera;

                // lighting uniforms depend on the camera so enforce an update
                // now, in case this material supports lights - or later, when
                // the next material that does gets activated:

                refreshMaterial = true;// set to true on material change
                refreshLights = true;  // remains set until update done
            }

            // load material specific uniforms
            // (shader material also gets them for the sake of genericity)

            if (isShaderMaterial ||
                isMeshPhongMaterial ||
                isMeshToonMaterial ||
                isMeshStandardMaterial ||
                isEnvMap) {

                if (p_uniforms->map.count("cameraPosition")) {

                    auto& uCamPos = p_uniforms->map["cameraPosition"];
                    _vector3.setFromMatrixPosition(*camera->matrixWorld);
                    uCamPos->setValue(_vector3);
                }
            }

            if (isMeshPhongMaterial ||
                isMeshToonMaterial ||
                isMeshLambertMaterial ||
                isMeshBasicMaterial ||
                isMeshStandardMaterial ||
                isShaderMaterial) {

                p_uniforms->setValue("isOrthographic", camera->is<OrthographicCamera>());
            }

            if (isMeshPhongMaterial ||
                isMeshToonMaterial ||
                isMeshLambertMaterial ||
                isMeshBasicMaterial ||
                isMeshStandardMaterial ||
                isShaderMaterial ||
                isShadowMaterial ||
                object->is<SkinnedMesh>()) {

                p_uniforms->setValue("viewMatrix", camera->matrixWorldInverse);
            }
        }

        // skinning uniforms must be set even if material didn't change
        // auto-setting of texture unit for bone texture must go before other textures
        // otherwise textures used for skinning can take over texture units reserved for other material textures

        if (auto skinned = object->as<SkinnedMesh>()) {

            const auto& bindMatrix = skinned->bindMatrix;
            const auto& bindMatrixInverse = skinned->bindMatrixInverse;

            p_uniforms->setValue("bindMatrix", bindMatrix);
            p_uniforms->setValue("bindMatrixInverse", bindMatrixInverse);

            auto& skeleton = skinned->skeleton;

            if (skeleton) {

                if (gl::GLCapabilities::instance().floatVertexTextures) {

                    if (!skeleton->boneTexture) skeleton->computeBoneTexture();

                    p_uniforms->setValue("boneTexture", skeleton->boneTexture.get(), &textures);
                    p_uniforms->setValue("boneTextureSize", skeleton->boneTextureSize);

                } else {

                    const auto& boneMatrices = skeleton->boneMatrices;
                    if (!boneMatrices.empty()) {
                        p_uniforms->setValue("boneMatrices", boneMatrices);
                    }
                }
            }
        }

        if (refreshMaterial || materialProperties->receiveShadow != object->receiveShadow) {

            materialProperties->receiveShadow = object->receiveShadow;
            p_uniforms->setValue("receiveShadow", object->receiveShadow);
        }

        if (refreshMaterial) {

            p_uniforms->setValue("toneMappingExposure", scope.toneMappingExposure);

            // [threepp] hack to solve #162
            if (_clippingEnabled) {
                m_uniforms["clippingPlanes"] = clipping.uniform;
            }

            if (materialProperties->needsLights) {

                // the current material requires lighting info

                // note: all lighting uniforms are always set correctly
                // they simply reference the renderer's state for their
                // values
                //
                // use the current material's .needsUpdate flags to set
                // the GL state when required

                markUniformsLightsNeedsUpdate(m_uniforms, refreshLights);
            }

            // refresh uniforms common to several materials

            if (fog && material->fog) {

                materials.refreshFogUniforms(m_uniforms, *fog);
            }

            materials.refreshMaterialUniforms(m_uniforms, material, _pixelRatio, _size.height);

            gl::GLUniforms::upload(materialProperties->uniformsList, m_uniforms, &textures);
        }

        if (isShaderMaterial) {

            auto m = dynamic_cast<ShaderMaterial*>(material);
            if (m->uniformsNeedUpdate) {

                gl::GLUniforms::upload(materialProperties->uniformsList, m_uniforms, &textures);
                m->uniformsNeedUpdate = false;
            }
        }

        if (material->is<SpriteMaterial>() && object->is<Sprite>()) {

            p_uniforms->setValue("center", object->as<Sprite>()->center);
        }

        // common matrices

        p_uniforms->setValue("modelViewMatrix", object->modelViewMatrix);
        p_uniforms->setValue("normalMatrix", object->normalMatrix);
        p_uniforms->setValue("modelMatrix", *object->matrixWorld);

        return program;
    }

    void markUniformsLightsNeedsUpdate(UniformMap& uniforms, bool value) {
        uniforms.at("ambientLightColor").needsUpdate = value;
        uniforms.at("lightProbe").needsUpdate = value;

        uniforms.at("directionalLights").needsUpdate = value;
        uniforms.at("directionalLightShadows").needsUpdate = value;
        uniforms.at("pointLights").needsUpdate = value;
        uniforms.at("pointLightShadows").needsUpdate = value;
        uniforms.at("spotLights").needsUpdate = value;
        uniforms.at("spotLightShadows").needsUpdate = value;
        uniforms.at("hemisphereLights").needsUpdate = value;
    }

    bool materialNeedsLights(Material* material) {
        bool isMeshLambertMaterial = material->type() == "MeshLambertMaterial";
        bool isMeshToonMaterial = material->type() == "MeshToonMaterial";
        bool isMeshPhongMaterial = material->type() == "MeshPhongMaterial";
        bool isMeshStandardMaterial = material->type() == "MeshStandardMaterial";
        bool isShadowMaterial = material->type() == "ShadowMaterial";
        bool isShaderMaterial = material->is<ShaderMaterial>();
        bool lights = false;

        if (auto materialWithLights = material->as<MaterialWithLights>()) {
            lights = materialWithLights->lights;
        }

        return isMeshLambertMaterial || isMeshToonMaterial || isMeshPhongMaterial ||
               isMeshStandardMaterial || isShadowMaterial ||
               (isShaderMaterial && lights);
    }

    void setRenderTarget(GLRenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {

        _currentRenderTarget = renderTarget;
        _currentActiveCubeFace = activeCubeFace;
        _currentActiveMipmapLevel = activeMipmapLevel;

        if (renderTarget && !properties.renderTargetProperties.get(renderTarget->uuid)->glFramebuffer) {

            textures.setupRenderTarget(renderTarget);
        }

        unsigned int framebuffer = 0;

        if (renderTarget) {

            const auto& texture = renderTarget->texture;

            framebuffer = *properties.renderTargetProperties.get(renderTarget->uuid)->glFramebuffer;

            _currentViewport.copy(renderTarget->viewport);
            _currentScissor.copy(renderTarget->scissor);
            _currentScissorTest = renderTarget->scissorTest;

        } else {

            _currentViewport.copy(_viewport).multiplyScalar(static_cast<float>(_pixelRatio)).floor();
            _currentScissor.copy(_scissor).multiplyScalar(static_cast<float>(_pixelRatio)).floor();
            _currentScissorTest = _scissorTest;
        }

        const bool framebufferBound = state.bindFramebuffer(GL_FRAMEBUFFER, framebuffer) && framebuffer;

        if (framebufferBound && gl::GLCapabilities::instance().drawBuffers) {

            bool needsUpdate = false;

            if (renderTarget) {

                if (_currentDrawBuffers.size() != 1 || _currentDrawBuffers.front() != GL_COLOR_ATTACHMENT0) {

                    _currentDrawBuffers[0] = GL_COLOR_ATTACHMENT0;
                    _currentDrawBuffers.resize(1);

                    needsUpdate = true;
                }

            } else {

                if (_currentDrawBuffers.size() != 1 || _currentDrawBuffers.front() != GL_BACK) {

                    _currentDrawBuffers.clear();
                    _currentDrawBuffers.emplace_back(GL_BACK);

                    needsUpdate = true;
                }
            }

            if (needsUpdate) {

                glDrawBuffers(static_cast<int>(_currentDrawBuffers.size()), _currentDrawBuffers.data());
            }
        }

        state.viewport(_currentViewport);
        state.scissor(_currentScissor);
        state.setScissorTest(_currentScissorTest.value_or(false));
    }

    void copyFramebufferToTexture(const Vector2& position, Texture& texture, int level) {

        const auto levelScale = std::pow(2, -level);
        const auto width = static_cast<int>(texture.image->width * levelScale);
        const auto height = static_cast<int>(texture.image->height * levelScale);

        textures.setTexture2D(texture, 0);

        glCopyTexSubImage2D(GL_TEXTURE_2D, level, 0, 0, static_cast<int>(position.x), static_cast<int>(position.y), width, height);

        state.unbindTexture();
    }

    void readPixels(const Vector2& position, const WindowSize& size, Format format, unsigned char* data) {

        auto glFormat = gl::toGLFormat(format);

        glReadPixels(static_cast<int>(position.x), static_cast<int>(position.y), size.width, size.width, glFormat, GL_UNSIGNED_BYTE, data);
    }

    void setViewport(int x, int y, int width, int height) {

        _viewport.set(static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height));

        state.viewport(_currentViewport.copy(_viewport).multiplyScalar(static_cast<float>(_pixelRatio)).floor());
    }

    void setScissor(int x, int y, int width, int height) {

        _scissor.set((float) x, (float) y, (float) width, (float) height);

        state.scissor(_currentScissor.copy(_scissor).multiplyScalar(static_cast<float>(_pixelRatio)).floor());
    }

    void dispose() {

        renderLists.dispose();
        renderStates.dispose();
        properties.dispose();
        //    cubemaps.dispose();
        objects.dispose();
        bindingStates.dispose();
    }

    void reset() {
        state.reset(_size.width, _size.height);
        bindingStates.reset();
    }

    ~Impl() = default;

    friend struct gl::ProgramParameters;
    friend struct gl::GLShadowMap;
};


GLRenderer::GLRenderer(WindowSize size, const GLRenderer::Parameters& parameters)
    : pimpl_(std::make_unique<Impl>(*this, size, parameters)) {}


const gl::GLInfo& threepp::GLRenderer::info() {

    return pimpl_->_info;
}

gl::GLShadowMap& threepp::GLRenderer::shadowMap() {

    return pimpl_->shadowMap;
}

const gl::GLShadowMap& threepp::GLRenderer::shadowMap() const {

    return pimpl_->shadowMap;
}

gl::GLState& threepp::GLRenderer::state() {

    return pimpl_->state;
}

int GLRenderer::getTargetPixelRatio() const {

    return pimpl_->_pixelRatio;
}

void GLRenderer::setPixelRatio(int value) {

    pimpl_->_pixelRatio = value;
    this->setSize({pimpl_->_size.width, pimpl_->_size.height});
}

WindowSize GLRenderer::getSize() const {

    return pimpl_->_size;
}

void GLRenderer::setSize(WindowSize size) {

    pimpl_->_size = size;

    this->setViewport(0, 0, size.width, size.height);
}

void GLRenderer::getDrawingBufferSize(Vector2& target) const {

    target.set(static_cast<float>(pimpl_->_size.width * pimpl_->_pixelRatio), static_cast<float>(pimpl_->_size.height * pimpl_->_pixelRatio)).floor();
}

void GLRenderer::setDrawingBufferSize(int width, int height, int pixelRatio) {

    pimpl_->_size.width = width;
    pimpl_->_size.height = height;

    pimpl_->_pixelRatio = pixelRatio;

    this->setViewport(0, 0, width, height);
}

void GLRenderer::getCurrentViewport(Vector4& target) const {

    target.copy(pimpl_->_currentViewport);
}

void GLRenderer::getViewport(Vector4& target) const {

    target.copy(pimpl_->_viewport);
}

void GLRenderer::setViewport(const Vector4& v) {

    setViewport(v.x, v.y, v.z, v.w);
}

void GLRenderer::setViewport(int x, int y, int width, int height) {

    pimpl_->setViewport(x, y, width, height);
}

void GLRenderer::getScissor(Vector4& target) {

    target.copy(pimpl_->_scissor);
}

void GLRenderer::setScissor(const Vector4& v) {

    setScissor(v.x, v.y, v.z, v.w);
}

void GLRenderer::setScissor(int x, int y, int width, int height) {

    pimpl_->setScissor(x, y, width, height);
}

bool GLRenderer::getScissorTest() const {

    return pimpl_->_scissorTest;
}

void GLRenderer::setScissorTest(bool boolean) {

    pimpl_->_scissorTest = boolean;

    pimpl_->state.setScissorTest(pimpl_->_scissorTest);
}

void GLRenderer::getClearColor(Color& target) const {

    target.copy(pimpl_->background.getClearColor());
}

void GLRenderer::setClearColor(const Color& color, float alpha) {

    pimpl_->background.setClearColor(color, alpha);
}

float GLRenderer::getClearAlpha() const {

    return pimpl_->background.getClearAlpha();
}

void GLRenderer::setClearAlpha(float clearAlpha) {

    pimpl_->background.setClearAlpha(clearAlpha);
}

void GLRenderer::clear(bool color, bool depth, bool stencil) {

    GLbitfield bits = 0;

    if (color) bits |= GL_COLOR_BUFFER_BIT;
    if (depth) bits |= GL_DEPTH_BUFFER_BIT;
    if (stencil) bits |= GL_STENCIL_BUFFER_BIT;

    glClear(bits);
}

void GLRenderer::clearColor() { clear(true, false, false); }

void GLRenderer::clearDepth() { clear(false, true, false); }

void GLRenderer::clearStencil() { clear(false, false, true); }

void GLRenderer::dispose() {

    pimpl_->dispose();
}

void GLRenderer::render(Scene& scene, Camera& camera) {

    pimpl_->render(&scene, &camera);
}

void GLRenderer::renderBufferDirect(Camera* camera, Scene* scene, BufferGeometry* geometry, Material* material, Object3D* object, std::optional<GeometryGroup> group) {

    pimpl_->renderBufferDirect(camera, scene, geometry, material, object, group);
}

void GLRenderer::setRenderTarget(GLRenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {

    pimpl_->setRenderTarget(renderTarget, activeCubeFace, activeMipmapLevel);
}

void GLRenderer::copyFramebufferToTexture(const Vector2& position, Texture& texture, int level) {

    pimpl_->copyFramebufferToTexture(position, texture, level);
}

void GLRenderer::readPixels(const Vector2& position, const WindowSize& size, Format format, unsigned char* data) {

    pimpl_->readPixels(position, size, format, data);
}

void GLRenderer::resetState() {

    pimpl_->reset();
}

const gl::GLInfo& threepp::GLRenderer::info() const {

    return pimpl_->_info;
}

int threepp::GLRenderer::getActiveCubeFace() const {

    return pimpl_->_currentActiveCubeFace;
}

int threepp::GLRenderer::getActiveMipmapLevel() const {

    return pimpl_->_currentActiveMipmapLevel;
}

GLRenderTarget* threepp::GLRenderer::getRenderTarget() {

    return pimpl_->_currentRenderTarget;
}

GLRenderer::~GLRenderer() = default;
