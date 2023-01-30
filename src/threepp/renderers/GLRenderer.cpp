
#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"

#include <glad/glad.h>

using namespace threepp;

GLRenderer::GLRenderer(Canvas &canvas, const GLRenderer::Parameters &parameters)
    : canvas_(canvas), _size(canvas.getSize()),
      _viewport(0, 0, _size.width, _size.height),
      _scissor(0, 0, _size.width, _size.height),
      state(canvas),
      background(state, parameters.premultipliedAlpha),
      bufferRenderer(new gl::GLBufferRenderer(info)),
      indexedBufferRenderer(new gl::GLIndexedBufferRenderer(info)),
      clipping(properties),
      bindingStates(attributes),
      geometries(attributes, info, bindingStates),
      textures(state, properties, info),
      objects(geometries, attributes, info),
      renderLists(properties),
      shadowMap(objects),
      materials(properties),
      programCache(bindingStates, clipping),
      onMaterialDispose(*this),
      _currentDrawBuffers(GL_BACK) {

    info.programs = &programCache.programs;
}

int GLRenderer::getTargetPixelRatio() const {
    return _pixelRatio;
}

void GLRenderer::setPixelRatio(int value) {
    _pixelRatio = value;
    this->setSize({_size.width, _size.height});
}

WindowSize GLRenderer::getSize() const {
    return _size;
}

void GLRenderer::setSize(WindowSize size) {

    _size = size;

    int canvasWidth = _size.width * _pixelRatio;
    int canvasHeight = _size.height * _pixelRatio;

    canvas_.setSize({canvasWidth, canvasHeight});

    this->setViewport(0, 0, size.width, size.height);
}

void GLRenderer::getDrawingBufferSize(Vector2 &target) const {

    target.set((float) (_size.width * _pixelRatio), (float) (_size.height * _pixelRatio)).floor();
}

void GLRenderer::setDrawingBufferSize(int width, int height, int pixelRatio) {

    _size.width = width;
    _size.height = height;

    _pixelRatio = pixelRatio;

    canvas_.setSize({width * pixelRatio, height * pixelRatio});

    this->setViewport(0, 0, width, height);
}

void GLRenderer::getCurrentViewport(Vector4 &target) const {

    target.copy(_currentViewport);
}

void GLRenderer::getViewport(Vector4 &target) const {

    target.copy(_viewport);
}

void GLRenderer::setViewport(const Vector4 &v) {

    _viewport.copy(v);

    state.viewport(_currentViewport.copy(_viewport).multiplyScalar(static_cast<float>(_pixelRatio)).floor());

    if (textEnabled_) {
        TextHandle::setViewport(static_cast<int>(v.z), static_cast<int>(v.w));
    }
}

void GLRenderer::setViewport(int x, int y, int width, int height) {

    _viewport.set(static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height));

    state.viewport(_currentViewport.copy(_viewport).multiplyScalar(static_cast<float>(_pixelRatio)).floor());

    if (textEnabled_) {
        TextHandle::setViewport(width, height);
    }
}

void GLRenderer::getScissor(Vector4 &target) {

    target.copy(_scissor);
}

void GLRenderer::setScissor(const Vector4 &v) {

    _scissor.copy(v);

    state.scissor(_currentScissor.copy(_scissor).multiplyScalar((float) _pixelRatio).floor());
}


void GLRenderer::setScissor(int x, int y, int width, int height) {

    _scissor.set((float) x, (float) y, (float) width, (float) height);

    state.scissor(_currentScissor.copy(_scissor).multiplyScalar((float) _pixelRatio).floor());
}

bool GLRenderer::getScissorTest() const {

    return _scissorTest;
}

void GLRenderer::setScissorTest(bool boolean) {

    _scissorTest = boolean;

    state.setScissorTest(_scissorTest);
}

void GLRenderer::getClearColor(Color &target) const {

    target.copy(background.getClearColor());
}

void GLRenderer::setClearColor(const Color &color, float alpha) {

    background.setClearColor(color, alpha);
}

float GLRenderer::getClearAlpha() const {

    return background.getClearAlpha();
}

void GLRenderer::setClearAlpha(float clearAlpha) {

    background.setClearAlpha(clearAlpha);
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

    renderLists.dispose();
    renderStates.dispose();
    properties.dispose();
    //    cubemaps.dispose();
    objects.dispose();
    bindingStates.dispose();
}

void GLRenderer::deallocateMaterial(Material &material) {

    releaseMaterialProgramReferences(material);

    properties.materialProperties.remove(material.uuid);
}

void GLRenderer::releaseMaterialProgramReferences(Material &material) {

    auto &programs = properties.materialProperties.get(material.uuid).programs;

    if (!programs.empty()) {

        for (auto &[key, program] : programs) {

            programCache.releaseProgram(program);
        }
    }
}

void GLRenderer::renderBufferDirect(
        Camera *camera,
        Scene *scene,
        BufferGeometry *geometry,
        Material *material,
        Object3D *object,
        std::optional<GeometryGroup> group) {

    if (scene == nullptr) {
        throw std::runtime_error("TODO");
        //scene = &_emptyScene; // renderBufferDirect second parameter used to be fog (could be nullptr)
    }

    bool isMesh = object->as<Mesh>();

    const auto frontFaceCW = (isMesh && object->matrixWorld->determinant() < 0);

    auto program = setProgram(camera, scene, material, object);

    state.setMaterial(material, frontFaceCW);

    //

    auto index = geometry->getIndex();
    const auto &position = geometry->getAttribute<float>("position");

    //

    if (index == nullptr) {

        if (!geometry->hasAttribute("position") || position->count() == 0) return;

    } else if (index->count() == 0) {

        return;
    }

    //

    int rangeFactor = 1;

    auto wireframeMaterial = dynamic_cast<MaterialWithWireframe *>(material);
    bool isWireframeMaterial = wireframeMaterial != nullptr;

    if (isWireframeMaterial && wireframeMaterial->wireframe) {

        index = geometries.getWireframeAttribute(*geometry);
        rangeFactor = 2;
    }

    bindingStates.setup(object, material, program, geometry, index);

    gl::BufferRenderer *renderer = bufferRenderer.get();

    if (index != nullptr) {

        const auto &attribute = attributes.get(index);

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

            state.setLineWidth(wireframeMaterial->wireframeLinewidth * (float) getTargetPixelRatio());
            renderer->setMode(GL_LINES);

        } else {

            renderer->setMode(GL_TRIANGLES);
        }

    } else if (object->as<Line>()) {

        float lineWidth = 1;
        if (isWireframeMaterial) {
            lineWidth = wireframeMaterial->wireframeLinewidth;
        }

        state.setLineWidth(lineWidth * static_cast<float>(getTargetPixelRatio()));

        if (object->as<LineSegments>()) {

            renderer->setMode(GL_LINES);

        } else if (object->as<LineLoop>()) {

            renderer->setMode(GL_LINE_LOOP);

        } else {

            renderer->setMode(GL_LINE_STRIP);
        }

    } else if (object->as<Points>()) {

        renderer->setMode(GL_POINTS);

    } else if (false /*object.isSprite*/) {

        renderer->setMode(GL_TRIANGLES);
    }

    if (object->as<InstancedMesh>()) {

        renderer->renderInstances(drawStart, drawCount, object->as<InstancedMesh>()->count);

    } else if (dynamic_cast<InstancedBufferGeometry *>(geometry)) {

        auto g = dynamic_cast<InstancedBufferGeometry *>(geometry);
        const auto instanceCount = std::min(g->instanceCount, g->_maxInstanceCount);

        renderer->renderInstances(drawStart, drawCount, instanceCount);

    } else {

        renderer->render(drawStart, drawCount);
    }
}

void GLRenderer::render(Scene *scene, Camera *camera) {

    // update scene graph

    if (scene->autoUpdate) scene->updateMatrixWorld();

    // update camera matrices and frustum

    if (camera->parent == nullptr) camera->updateMatrixWorld();

    //
    //    if ( scene.isScene === true ) scene.onBeforeRender( _this, scene, camera, _currentRenderTarget );

    currentRenderState = renderStates.get(scene, (int) renderStateStack.size());
    currentRenderState->init();

    renderStateStack.emplace_back(currentRenderState);

    _projScreenMatrix.multiplyMatrices(camera->projectionMatrix, camera->matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_projScreenMatrix);

    _localClippingEnabled = this->localClippingEnabled;
    _clippingEnabled = clipping.init(this->clippingPlanes, _localClippingEnabled, camera);

    currentRenderList = renderLists.get(scene, renderListStack.size());
    currentRenderList->init();

    renderListStack.emplace_back(currentRenderList);

    projectObject(scene, camera, 0, sortObjects);

    currentRenderList->finish();

    if (sortObjects) {

        currentRenderList->sort();
    }

    //

    if (_clippingEnabled) clipping.beginShadows();

    auto &shadowsArray = currentRenderState->getShadowsArray();

    shadowMap.render(*this, shadowsArray, scene, camera);

    currentRenderState->setupLights();
    currentRenderState->setupLightsView(camera);

    if (_clippingEnabled) clipping.endShadows();

    //

    if (this->info.autoReset) this->info.reset();

    //

    background.render(*this, scene);

    // render scene

    auto &opaqueObjects = currentRenderList->opaque;
    auto &transparentObjects = currentRenderList->transparent;
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

    renderText();
}

void GLRenderer::projectObject(Object3D *object, Camera *camera, unsigned int groupOrder, bool sortObjects) {

    if (!object->visible) return;

    bool visible = object->layers.test(camera->layers);

    if (visible) {

        if (object->as<Group>()) {

            groupOrder = object->renderOrder;

        } else if (object->as<LOD>()) {

            auto lod = object->as<LOD>();
            if (lod->autoUpdate) lod->update(camera);

        } else if (object->as<Light>()) {

            auto light = object->as<Light>();

            currentRenderState->pushLight(light);

            if (light->castShadow) {

                currentRenderState->pushShadow(light);
            }

            //        } else if ( object.isSprite ) {
            //
            //            if ( ! object.frustumCulled || _frustum.intersectsSprite( object ) ) {
            //
            //                if ( sortObjects ) {
            //
            //                    _vector3.setFromMatrixPosition( object.matrixWorld )
            //                            .applyMatrix4( _projScreenMatrix );
            //
            //                }
            //
            //                const geometry = objects.update( object );
            //                const material = object.material;
            //
            //                if ( material.visible ) {
            //
            //                    currentRenderList.push( object, geometry, material, groupOrder, _vector3.z, null );
            //
            //                }
            //
            //            }
            //
            //        } else if ( object.isImmediateRenderObject ) {
            //
            //            if ( sortObjects ) {
            //
            //                _vector3.setFromMatrixPosition( object.matrixWorld )
            //                        .applyMatrix4( _projScreenMatrix );
            //
            //            }
            //
            //            currentRenderList.push( object, null, object.material, groupOrder, _vector3.z, null );

        } else if (object->as<Mesh>() || object->as<Line>() || object->as<Points>()) {

            if (!object->frustumCulled || _frustum.intersectsObject(*object)) {

                if (sortObjects) {

                    _vector3.setFromMatrixPosition(*object->matrixWorld)
                            .applyMatrix4(_projScreenMatrix);
                }

                auto geometry = objects.update(object);
                const auto materials = object->materials();

                if (materials.size() > 1) {

                    const auto &groups = geometry->groups;

                    for (const auto &group : groups) {

                        auto &groupMaterial = materials.at(group.materialIndex);

                        if (groupMaterial && groupMaterial->visible) {

                            currentRenderList->push(object, geometry.get(), groupMaterial.get(), groupOrder, _vector3.z, group);
                        }
                    }

                } else if (materials.front()->visible) {

                    currentRenderList->push(object, geometry.get(), materials.front().get(), groupOrder, _vector3.z, std::nullopt);
                }
            }
        }
    }

    for (const auto &child : object->children) {

        projectObject(child.get(), camera, groupOrder, sortObjects);
    }
}

void GLRenderer::renderObjects(
        const std::vector<gl::RenderItem*> &renderList,
        Scene *scene,
        Camera *camera) {

    auto overrideMaterial = scene->overrideMaterial;

    for (const auto &renderItem : renderList) {

        auto object = renderItem->object;
        auto geometry = renderItem->geometry;
        auto material = overrideMaterial == nullptr ? renderItem->material : overrideMaterial.get();
        auto group = renderItem->group;

        renderObject(object, scene, camera, geometry, material, group);
    }
}

void GLRenderer::renderObject(
        Object3D *object,
        Scene *scene,
        Camera *camera,
        BufferGeometry *geometry,
        Material *material,
        std::optional<GeometryGroup> group) {

    if (object->onBeforeRender) {

        object->onBeforeRender.value()((void *) this, scene, camera, geometry, material, group);
    }

    object->modelViewMatrix.multiplyMatrices(camera->matrixWorldInverse, *object->matrixWorld);
    object->normalMatrix.getNormalMatrix(object->modelViewMatrix);

    if (false /*object.isImmediateRenderObject*/) {

        //        const program = setProgram( camera, scene, material, object );
        //
        //        state.setMaterial( material );
        //
        //        bindingStates.reset();
        //
        //        renderObjectImmediate( object, program );

    } else {

        renderBufferDirect(camera, scene, geometry, material, object, group);
    }

    if (object->onAfterRender) {

        object->onAfterRender.value()(this, scene, camera, geometry, material, group);
    }
}

std::shared_ptr<gl::GLProgram> GLRenderer::getProgram(
        Material *material,
        Scene *scene,
        Object3D *object) {

    //    bool isScene = instanceof <Scene>(scene);
    //
    //    if (!isScene) scene = &_emptyScene;// scene could be a Mesh, Line, Points, ...

    auto &materialProperties = properties.materialProperties.get(material->uuid);

    auto &lights = currentRenderState->getLights();
    auto &shadowsArray = currentRenderState->getShadowsArray();

    auto lightsStateVersion = lights.state.version;

    auto parameters = gl::GLPrograms::getParameters(*this, material, lights.state, static_cast<int>(shadowsArray.size()), scene, object);
    auto programCacheKey = gl::GLPrograms::getProgramCacheKey(*this, parameters);

    auto &programs = materialProperties.programs;

    // always update environment and fog - changing these trigger an getProgram call, but it's possible that the program doesn't change

    materialProperties.environment = material->as<MeshStandardMaterial>() ? scene->environment : nullptr;
    materialProperties.fog = scene->fog;
    //    materialProperties.envMap = cubemaps.get( material.envMap || materialProperties.environment );

    if (programs.empty()) {

        // new material

        material->addEventListener("dispose", &onMaterialDispose);
    }

    std::shared_ptr<gl::GLProgram> program = nullptr;

    if (programs.count(programCacheKey)) {

        program = programs.at(programCacheKey);

        if (materialProperties.currentProgram == program && materialProperties.lightsStateVersion == lightsStateVersion) {

            updateCommonMaterialProperties(material, parameters);

            return program;
        }

    } else {

        parameters.uniforms = gl::GLPrograms::getUniforms(material);

        // material.onBuild( parameters, this );

        // material.onBeforeCompile( parameters, this );

        program = programCache.acquireProgram(*this, parameters, programCacheKey);
        programs[programCacheKey] = program;

        materialProperties.uniforms = parameters.uniforms;
    }

    auto &uniforms = *materialProperties.uniforms;

    if (!material->as<ShaderMaterial>() && !material->as<RawShaderMaterial>() || material->clipping) {

        uniforms["clippingPlanes"] = clipping.uniform;
    }

    updateCommonMaterialProperties(material, parameters);

    // store the light setup it was created for

    materialProperties.needsLights = materialNeedsLights(material);
    materialProperties.lightsStateVersion = lightsStateVersion;

    if (materialProperties.needsLights) {

        // wire up the material to this renderer's lighting state

        uniforms.at("ambientLightColor").setValue(lights.state.ambient);
        uniforms.at("lightProbe").setValue(lights.state.probe);
        uniforms.at("directionalLights").setValue(lights.state.directional);
        uniforms.at("directionalLightShadows").setValue(lights.state.directionalShadow);
        uniforms.at("spotLights").setValue(lights.state.spot);
        uniforms.at("spotLightShadows").setValue(lights.state.spotShadow);
        uniforms.at("rectAreaLights").setValue(lights.state.rectArea);
        //        uniforms.at("ltc_1").setValue(lights.state.rectAreaLTC1);
        //        uniforms.at("ltc_2").setValue(lights.state.rectAreaLTC2);
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
    auto uniformsList = gl::GLUniforms::seqWithValue(progUniforms->seq, *materialProperties.uniforms);

    materialProperties.currentProgram = program;
    materialProperties.uniformsList = uniformsList;

    return materialProperties.currentProgram;
}

void GLRenderer::updateCommonMaterialProperties(Material *material, gl::ProgramParameters &parameters) {

    auto &materialProperties = properties.materialProperties.get(material->uuid);

    materialProperties.outputEncoding = parameters.outputEncoding;
    materialProperties.instancing = parameters.instancing;
    materialProperties.numClippingPlanes = parameters.numClippingPlanes;
    materialProperties.numIntersection = parameters.numClipIntersection;
    materialProperties.vertexAlphas = parameters.vertexAlphas;
}


std::shared_ptr<gl::GLProgram> GLRenderer::setProgram(
        Camera *camera,
        Scene *scene,
        Material *material,
        Object3D *object) {

    //    bool isScene = instanceof <Scene>(scene);

    //            if (!isScene) scene = _emptyScene;// scene could be a Mesh, Line, Points, ...
    //

    bool isMeshBasicMaterial = material->is<MeshBasicMaterial>();
    bool isMeshLambertMaterial = material->is<MeshLambertMaterial>();
    bool isMeshToonMaterial = material->is<MeshToonMaterial>();
    bool isMeshPhongMaterial = material->is<MeshPhongMaterial>();
    bool isMeshStandardMaterial = material->is<MeshStandardMaterial>();
    bool isShadowMaterial = material->is<ShadowMaterial>();
    bool isShaderMaterial = material->is<ShaderMaterial>();
    bool isEnvMap = material->is<MaterialWithEnvMap>() && material->as<MaterialWithEnvMap>()->envMap;

    textures.resetTextureUnits();

    auto fog = scene->fog;
    auto environment = isMeshStandardMaterial ? scene->environment : nullptr;
    int encoding = (_currentRenderTarget == nullptr) ? outputEncoding : _currentRenderTarget->texture->encoding;
    //                const envMap = cubemaps.get(material.envMap || environment);
    bool vertexAlphas = material->vertexColors &&
                        object->geometry() &&
                        object->geometry()->hasAttribute("color") &&
                        object->geometry()->getAttribute<float>("color")->itemSize() == 4;

    auto &materialProperties = properties.materialProperties.get(material->uuid);
    auto &lights = currentRenderState->getLights();

    if (_clippingEnabled) {

        if (_localClippingEnabled || camera != _currentCamera) {

            bool useCache = camera == _currentCamera && material->id == _currentMaterialId;

            // we might want to call this function with some ClippingGroup
            // object instead of the material, once it becomes feasible
            // (#8465, #8379)
            clipping.setState(material, camera, useCache);
        }
    }

    //

    bool needsProgramChange = false;
    bool isInstancedMesh = object->as<InstancedMesh>();

    if (material->version == materialProperties.version) {

        if (materialProperties.needsLights && (materialProperties.lightsStateVersion != lights.state.version)) {

            needsProgramChange = true;

        } else if (materialProperties.outputEncoding != encoding) {

            needsProgramChange = true;

        } else if (isInstancedMesh && !materialProperties.instancing) {

            needsProgramChange = true;

        } else if (!isInstancedMesh && materialProperties.instancing) {

            needsProgramChange = true;

        } else if (false /*materialProperties.envMap != envMap*/) {

            needsProgramChange = true;

        } else if (fog && material->fog && materialProperties.fog && !(fog.value() == materialProperties.fog.value())) {

            needsProgramChange = true;

        } else if (materialProperties.numClippingPlanes &&
                   (materialProperties.numClippingPlanes.value() != clipping.numPlanes ||
                    materialProperties.numIntersection != clipping.numIntersection)) {

            needsProgramChange = true;

        } else if (materialProperties.vertexAlphas != vertexAlphas) {

            needsProgramChange = true;
        }

    } else {

        needsProgramChange = true;
        materialProperties.version = material->version;
    }

    //

    auto program = materialProperties.currentProgram;

    if (needsProgramChange) {

        program = getProgram(material, scene, object);
    }

    bool refreshProgram = false;
    bool refreshMaterial = false;
    bool refreshLights = false;

    auto p_uniforms = program->getUniforms();
    auto &m_uniforms = materialProperties.uniforms;

    if (state.useProgram(program->program, textEnabled_)) {

        refreshProgram = true;
        refreshMaterial = true;
        refreshLights = true;
    }

    if (material->id != _currentMaterialId) {

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

                auto &uCamPos = p_uniforms->map["cameraPosition"];
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
            isShadowMaterial) {

            p_uniforms->setValue("viewMatrix", camera->matrixWorldInverse);
        }
    }

    if (refreshMaterial || materialProperties.receiveShadow != object->receiveShadow) {

        materialProperties.receiveShadow = object->receiveShadow;
        p_uniforms->setValue("receiveShadow", object->receiveShadow);
    }

    if (refreshMaterial) {

        p_uniforms->setValue("toneMappingExposure", toneMappingExposure);

        if (materialProperties.needsLights) {

            // the current material requires lighting info

            // note: all lighting uniforms are always set correctly
            // they simply reference the renderer's state for their
            // values
            //
            // use the current material's .needsUpdate flags to set
            // the GL state when required

            markUniformsLightsNeedsUpdate(*m_uniforms, refreshLights);
        }

        // refresh uniforms common to several materials

        if (fog && material->fog) {

            materials.refreshFogUniforms(*m_uniforms, *fog);
        }

        materials.refreshMaterialUniforms(*m_uniforms, material, _pixelRatio, _size.height);

        gl::GLUniforms::upload(materialProperties.uniformsList, *m_uniforms, &textures);
    }

    if (isShaderMaterial) {

        auto m = dynamic_cast<ShaderMaterial *>(material);
        if (m->uniformsNeedUpdate) {

            gl::GLUniforms::upload(materialProperties.uniformsList, *m_uniforms, &textures);
            m->uniformsNeedUpdate = false;
        }
    }

    if (false /*material.isSpriteMaterial*/) {

        // TODO
        //                        p_uniforms->setValue("center", object->center);
    }

    // common matrices

    p_uniforms->setValue("modelViewMatrix", object->modelViewMatrix);
    p_uniforms->setValue("normalMatrix", object->normalMatrix);
    p_uniforms->setValue("modelMatrix", *object->matrixWorld);

    return program;
}

void GLRenderer::markUniformsLightsNeedsUpdate(UniformMap &uniforms, bool value) {
    uniforms.at("ambientLightColor").needsUpdate = value;
    uniforms.at("lightProbe").needsUpdate = value;

    uniforms.at("directionalLights").needsUpdate = value;
    uniforms.at("directionalLightShadows").needsUpdate = value;
    uniforms.at("pointLights").needsUpdate = value;
    uniforms.at("pointLightShadows").needsUpdate = value;
    uniforms.at("spotLights").needsUpdate = value;
    uniforms.at("spotLightShadows").needsUpdate = value;
    uniforms.at("rectAreaLights").needsUpdate = value;
    uniforms.at("hemisphereLights").needsUpdate = value;
}

bool GLRenderer::materialNeedsLights(Material *material) {

    bool isMeshLambertMaterial = material->as<MeshLambertMaterial>();
    bool isMeshToonMaterial = material->as<MeshToonMaterial>();
    bool isMeshPhongMaterial = material->as<MeshPhongMaterial>();
    bool isMeshStandardMaterial = material->as<MeshStandardMaterial>();
    bool isShadowMaterial = material->as<ShadowMaterial>();
    bool isShaderMaterial = material->as<ShaderMaterial>();
    bool lights = false;

    if (material->as<MaterialWithLights>()) {
        lights = material->as<MaterialWithLights>()->lights;
    }

    return isMeshLambertMaterial || isMeshToonMaterial || isMeshPhongMaterial ||
           isMeshStandardMaterial || isShadowMaterial ||
           (isShaderMaterial && lights);
}

void GLRenderer::setRenderTarget(const std::shared_ptr<GLRenderTarget> &renderTarget, int activeCubeFace, int activeMipmapLevel) {

    _currentRenderTarget = renderTarget;
    _currentActiveCubeFace = activeCubeFace;
    _currentActiveMipmapLevel = activeMipmapLevel;

    if (renderTarget && !properties.renderTargetProperties.get(renderTarget->uuid).glFramebuffer) {

        textures.setupRenderTarget(renderTarget);
    }

    unsigned int framebuffer = 0;

    if (renderTarget) {

        const auto &texture = renderTarget->texture;

        framebuffer = *properties.renderTargetProperties.get(renderTarget->uuid).glFramebuffer;

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

            if (_currentDrawBuffers.size() != 1 || _currentDrawBuffers[0] != GL_COLOR_ATTACHMENT0) {

                _currentDrawBuffers[0] = GL_COLOR_ATTACHMENT0;
                _currentDrawBuffers.resize(1);

                needsUpdate = true;
            }

        } else {

            if (_currentDrawBuffers.size() != 1 || _currentDrawBuffers[0] != GL_BACK) {

                _currentDrawBuffers.clear();
                _currentDrawBuffers.emplace_back(GL_BACK);

                needsUpdate = true;
            }
        }

        if (needsUpdate) {

            glDrawBuffers((int) _currentDrawBuffers.size(), _currentDrawBuffers.data());
        }
    }

    state.viewport(_currentViewport);
    state.scissor(_currentScissor);
    state.setScissorTest(_currentScissorTest.value_or(false));
}

void GLRenderer::enableTextRendering() {
    if (!textEnabled_) {
        textEnabled_ = TextHandle::init();
        TextHandle::setViewport(static_cast<int>(_viewport.z), static_cast<int>(_viewport.w));
    }
}

TextHandle &GLRenderer::textHandle(const std::string &str) {
    textHandles_.emplace_back(std::make_unique<TextHandle>(str));
    return *textHandles_.back();
}

void GLRenderer::renderText() {

    if (textEnabled_ && !textHandles_.empty()) {

        TextHandle::beginDraw();
        auto it = textHandles_.begin();
        while (it != textHandles_.end()) {

            if ((*it)->invalidate_) {
                it = textHandles_.erase(it);
            } else {

                (*it)->render();
                ++it;
            }
        }
        TextHandle::endDraw();
        bindingStates.reset();
    }
}

GLRenderer::~GLRenderer() {
    if (textEnabled_) {
        TextHandle::terminate();
    }
}

GLRenderer::OnMaterialDispose::OnMaterialDispose(GLRenderer &scope) : scope_(scope) {}

void GLRenderer::OnMaterialDispose::onEvent(Event &event) {
    auto material = static_cast<Material *>(event.target);

    material->removeEventListener("dispose", this);

    scope_.deallocateMaterial(*material);
}
