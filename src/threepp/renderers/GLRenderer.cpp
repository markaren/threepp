
#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"

using namespace threepp;

GLRenderer::GLRenderer(Canvas &canvas, const GLRenderer::Parameters &parameters)
    : canvas_(canvas), _width(canvas.getWidth()), _height(canvas.getHeight()),
      _viewport(0, 0, _width, _height),
      _scissor(0, 0, _width, _height), state(canvas),
      background(state, parameters.premultipliedAlpha),
      bufferRenderer(info),
      indexedBufferRenderer(info),
      clipping(properties),
      bindingStates(attributes),
      geometries(attributes, info, bindingStates),
      textures(state, properties, info),
      objects(geometries, attributes, info),
      renderLists(properties),
      shadowMap(objects),
      programCache(bindingStates, clipping),
      onMaterialDispose(*this) {

    info.programs = &programCache.programs;
}

int GLRenderer::getTargetPixelRatio() const {
    return _pixelRatio;
}

void GLRenderer::getSize(Vector2 &target) const {
    target.set((float) _width, (float) _height);
}

void GLRenderer::setSize(int width, int height) {

    _width = width;
    _height = height;

    canvas_.setSize(width * _pixelRatio, height * _pixelRatio);
}

void GLRenderer::getDrawingBufferSize(Vector2 &target) const {

    target.set((float) (_width * _pixelRatio), (float) (_height * _pixelRatio)).floor();
}

void GLRenderer::setDrawingBufferSize(int width, int height, int pixelRatio) {

    _width = width;
    _height = height;

    _pixelRatio = pixelRatio;

    canvas_.setSize(width * pixelRatio, height * pixelRatio);

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

    state.viewport(_currentViewport.copy(_viewport).multiplyScalar((float) _pixelRatio).floor());
}

void GLRenderer::setViewport(int x, int y, int width, int height) {

    _viewport.set((float) x, (float) y, (float) width, (float) height);

    state.viewport(_currentViewport.copy(_viewport).multiplyScalar((float) _pixelRatio).floor());
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

void GLRenderer::deallocateMaterial(Material *material) {

    releaseMaterialProgramReferences(material);

    properties.materialProperties.remove(material->uuid);
}

void GLRenderer::releaseMaterialProgramReferences(Material *material) {

    auto &programs = properties.materialProperties.get(material->uuid).programs;

    if (!programs.empty()) {

        for (auto &[key, program] : programs) {

            programCache.releaseProgram(program);
        }
    }
}

void GLRenderer::renderBufferDirect(Camera *camera, Scene *scene, BufferGeometry *geometry, Material *material, Object3D *object, std::optional<GeometryGroup> group) {

    //    if (scene == nullptr) scene = &_emptyScene;// renderBufferDirect second parameter used to be fog (could be nullptr)

    bool isMesh = instanceof <Mesh>(object);

    const auto frontFaceCW = (isMesh && object->matrixWorld.determinant() < 0);

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

    if (isWireframeMaterial) {

        if (wireframeMaterial->wireframe) {

            index = geometries.getWireframeAttribute(geometry);
            rangeFactor = 2;
        }
    }

    bindingStates.setup(object, material, program, geometry, index);

    gl::Buffer attribute{};
    gl::BufferRenderer *renderer = &bufferRenderer;

    if (index != nullptr) {

        attribute = attributes.get(index);

        renderer = &indexedBufferRenderer;
        indexedBufferRenderer.setIndex(attribute);
    }

    //

    const auto dataCount = (index != nullptr) ? index->count() : position->count();

    const auto rangeStart = geometry->drawRange.start * rangeFactor;
    const auto rangeCount = geometry->drawRange.count * rangeFactor;

    const auto groupStart = group ? group->start * rangeFactor : 0;
    const auto groupCount = group ? group->count * rangeFactor : Infinity<int>;

    const auto drawStart = std::max(rangeStart, groupStart);
    const auto drawEnd = std::min(dataCount, std::min(rangeStart + rangeCount, groupStart + groupCount)) - 1;

    const auto drawCount = std::max(0, drawEnd - drawStart + 1);

    if (drawCount == 0) return;

    //

    if (isMesh) {

        if (isWireframeMaterial) {

            if (wireframeMaterial->wireframe) {

                state.setLineWidth(wireframeMaterial->wireframeLinewidth * (float) getTargetPixelRatio());
                renderer->setMode(GL_LINES);
            }

        } else {

            renderer->setMode(GL_TRIANGLES);
        }

    } else if (instanceof <Line>(object)) {

        float lineWidth = 1;
        if (isWireframeMaterial) {
            lineWidth = wireframeMaterial->wireframeLinewidth;
        }

        state.setLineWidth(lineWidth * getTargetPixelRatio());

        if (instanceof <LineSegments>(object)) {

            renderer->setMode(GL_LINES);

        } else if (instanceof <LineLoop>(object)) {

            renderer->setMode(GL_LINE_LOOP);

        } else {

            renderer->setMode(GL_LINE_STRIP);
        }

    } else if (instanceof <Points>(object)) {

        renderer->setMode(GL_POINTS);

    } else if (false /*object.isSprite*/) {

        //                renderer.setMode(GL_TRIANGLES);
    }

    if (instanceof <InstancedMesh>(object)) {

        auto instancedMesh = dynamic_cast<InstancedMesh *>(object);

        //TODO
        //        renderer->renderInstances(drawStart, drawCount, object->count);

    } else if (instanceof <InstancedBufferGeometry>(geometry)) {

        auto g = dynamic_cast<InstancedBufferGeometry *>(geometry);
        const auto instanceCount = std::min(g->instanceCount, g->_maxInstanceCount);

        //TODO
        //                renderer->renderInstances(drawStart, drawCount, instanceCount);

    } else {

        renderer->render(drawStart, drawCount);
    }
}

void GLRenderer::compile(Scene *scene, Camera *camera) {

    currentRenderState = renderStates.get(scene);
    currentRenderState->init();

    scene->traverseVisible([&](Object3D &object) {
        auto light = dynamic_cast<Light *>(&object);

        if (light && object.layers.test(camera->layers)) {

            currentRenderState->pushLight(light);

            if (object.castShadow) {

                currentRenderState->pushShadow(light);
            }
        }
    });

    currentRenderState->setupLights();

    scene->traverse([&](Object3D &object) {
        auto material = object.material();

        if (material) {

            if (false /*Array.isArray(material)*/) {

//                for (let i = 0; i < material.length; i++) {
//
//                    const material2 = material[i];
//
//                    getProgram(material2, scene, object);
//                }

            } else {

                getProgram(material, scene, &object);
            }
        }
    });
}

void GLRenderer::render(const std::shared_ptr<Scene> &scene, const std::shared_ptr<Camera> &camera) {

    // update scene graph

    if (scene->autoUpdate) scene->updateMatrixWorld();

    // update camera matrices and frustum

    if (camera->parent == nullptr) camera->updateMatrixWorld();

    //
    //    if ( scene.isScene === true ) scene.onBeforeRender( _this, scene, camera, _currentRenderTarget );

    currentRenderState = renderStates.get(scene.get(), (int) renderStateStack.size());
    currentRenderState->init();

    renderStateStack.emplace_back(currentRenderState);

    _projScreenMatrix.multiplyMatrices(camera->projectionMatrix, camera->matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_projScreenMatrix);

    _localClippingEnabled = this->localClippingEnabled;
    _clippingEnabled = clipping.init(this->clippingPlanes, _localClippingEnabled, camera.get());

    currentRenderList = renderLists.get(scene.get(), (int) renderListStack.size());
    currentRenderList->init();

    renderListStack.emplace_back(currentRenderList);

    projectObject(scene.get(), camera.get(), 0, sortObjects);

    currentRenderList->finish();

    if (sortObjects) {

        currentRenderList->sort();
    }

    //

    if (_clippingEnabled) clipping.beginShadows();

    auto &shadowsArray = currentRenderState->getShadowsArray();

    shadowMap.render(*this, shadowsArray, scene.get(), camera.get());

    currentRenderState->setupLights();
    currentRenderState->setupLightsView(camera.get());

    if (_clippingEnabled) clipping.endShadows();

    //

    if (this->info.autoReset) this->info.reset();

    //

    background.render(*this, scene.get());

    // render scene

    auto &opaqueObjects = currentRenderList->opaque;
    auto &transmissiveObjects = currentRenderList->transmissive;
    auto &transparentObjects = currentRenderList->transparent;
    //
    if (!opaqueObjects.empty()) renderObjects(opaqueObjects, scene.get(), camera.get());
    if (!transmissiveObjects.empty()) renderTransmissiveObjects(opaqueObjects, transmissiveObjects, scene.get(), camera.get());
    if (!transparentObjects.empty()) renderObjects(transparentObjects, scene.get(), camera.get());

    //

    if (_currentRenderTarget) {

        // Generate mipmap if we're using any kind of mipmap filtering

        //                textures.updateRenderTargetMipmap( _currentRenderTarget );

        // resolve multisample renderbuffers to a single-sample texture if necessary

        //        textures.updateMultisampleRenderTarget( _currentRenderTarget );
    }

    //

    //    if ( scene.isScene === true ) scene.onAfterRender( _this, scene, camera );

    // Ensure depth buffer writing is enabled so it can be cleared on next render

    state.depthBuffer.setTest(true);
    state.depthBuffer.setMask(true);
    state.colorBuffer.setMask(true);
    //
    state.setPolygonOffset(false);

    // finish);

    _currentMaterialId = -1;
    _currentCamera = nullptr;

    renderStateStack.pop_back();

    if (!renderStateStack.empty()) {

        currentRenderState = renderStateStack[renderStateStack.size() - 1];

    } else {

        currentRenderState = nullptr;
    }

    renderListStack.pop_back();

    if (!renderListStack.empty()) {

        currentRenderList = renderListStack[renderListStack.size() - 1];

    } else {

        currentRenderList = nullptr;
    }
}

void GLRenderer::projectObject(Object3D *object, Camera *camera, int groupOrder, bool sortObjects) {

    if (!object->visible) return;

    bool visible = object->layers.test(camera->layers);

    if (visible) {

        if (instanceof <Group>(object)) {

            groupOrder = object->renderOrder;

        } else if (instanceof <LOD>(object)) {

            auto lod = dynamic_cast<LOD *>(object);
            if (lod->autoUpdate) lod->update(camera);

        } else if (instanceof <Light>(object)) {

            auto light = dynamic_cast<Light *>(object);

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

        } else if (instanceof <Mesh>(object) || instanceof <Line>(object) || instanceof <Points>(object)) {

            if (!object->frustumCulled || _frustum.intersectsObject(*object)) {

                if (sortObjects) {

                    _vector3.setFromMatrixPosition(object->matrixWorld)
                            .applyMatrix4(_projScreenMatrix);
                }

                auto geometry = objects.update(object);
                auto material = object->material();

                if (false /*Array.isArray( material )*/) {

                    //                    const groups = geometry.groups;
                    //
                    //                    for ( let i = 0, l = groups.length; i < l; i ++ ) {
                    //
                    //                        const group = groups[ i ];
                    //                        const groupMaterial = material[ group.materialIndex ];
                    //
                    //                        if ( groupMaterial && groupMaterial.visible ) {
                    //
                    //                            currentRenderList.push( object, geometry, groupMaterial, groupOrder, _vector3.z, group );
                    //
                    //                        }
                    //
                    //                    }

                } else if (material->visible) {

                    currentRenderList->push(object, geometry, material, groupOrder, _vector3.z, std::nullopt);
                }
            }
        }
    }

    for (const auto &child : object->children) {

        projectObject(child, camera, groupOrder, sortObjects);
    }
}

void GLRenderer::renderTransmissiveObjects(std::vector<gl::RenderItem> &opaqueObjects, std::vector<gl::RenderItem> &transmissiveObjects, Scene *scene, Camera *camera) {
    //TODO
}


void GLRenderer::renderObjects(std::vector<gl::RenderItem> &renderList, Scene *scene, Camera *camera) {

    auto overrideMaterial = scene->overrideMaterial;

    for (const auto &renderItem : renderList) {

        auto object = renderItem.object;
        auto geometry = renderItem.geometry;
        auto material = overrideMaterial == nullptr ? renderItem.material : overrideMaterial.get();
        auto group = renderItem.group;

        if (false /*camera.isArrayCamera*/) {

            //            const cameras = camera.cameras;
            //
            //            for ( let j = 0, jl = cameras.length; j < jl; j ++ ) {
            //
            //                const camera2 = cameras[ j ];
            //
            //                if ( object.layers.test( camera2.layers ) ) {
            //
            //                    state.viewport( _currentViewport.copy( camera2.viewport ) );
            //
            //                    currentRenderState.setupLightsView( camera2 );
            //
            //                    renderObject( object, scene, camera2, geometry, material, group );
            //
            //                }
            //
            //            }

        } else {

            renderObject(object, scene, camera, geometry, material, group);
        }
    }
}

void GLRenderer::renderObject(Object3D *object, Scene *scene, Camera *camera, BufferGeometry *geometry, Material *material, std::optional<GeometryGroup> group) {

    //    object.onBeforeRender( _this, scene, camera, geometry, material, group );

    object->modelViewMatrix.multiplyMatrices(camera->matrixWorldInverse, object->matrixWorld);
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

    //    object.onAfterRender( _this, scene, camera, geometry, material, group );
}

std::shared_ptr<gl::GLProgram> GLRenderer::getProgram(Material *material, Scene *scene, Object3D *object) {

    //    bool isScene = instanceof <Scene>(scene);
    //
    //    if (!isScene) scene = &_emptyScene;// scene could be a Mesh, Line, Points, ...

    auto &materialProperties = properties.materialProperties.get(material->uuid);

    auto &lights = currentRenderState->getLights();
    auto &shadowsArray = currentRenderState->getShadowsArray();

    auto lightsStateVersion = lights.state.version;

    auto parameters = programCache.getParameters(*this, material, lights.state, (int) shadowsArray.size(), scene, object);
    auto programCacheKey = programCache.getProgramCacheKey(*this, parameters);

    auto &programs = materialProperties.programs;

    // always update environment and fog - changing these trigger an getProgram call, but it's possible that the program doesn't change

    materialProperties.environment = instanceof <MeshStandardMaterial>(material) ? scene->environment : std::nullopt;
    materialProperties.fog = scene->fog;
    //    materialProperties.envMap = cubemaps.get( material.envMap || materialProperties.environment );

    if (programs.empty()) {

        // new material

        material->addEventListener("dispose", &onMaterialDispose);
    }

    std::shared_ptr<gl::GLProgram> program = nullptr;

    if (programs.count(programCacheKey)) {

        program = programs[programCacheKey];

        if (materialProperties.currentProgram == program && materialProperties.lightsStateVersion == lightsStateVersion) {

            updateCommonMaterialProperties(material, parameters);

            return program;
        }

    } else {

        parameters.uniforms = programCache.getUniforms(material);

        // material.onBuild( parameters, this );

        // material.onBeforeCompile( parameters, this );

        program = programCache.acquireProgram(*this, parameters, programCacheKey);
        programs[programCacheKey] = program;

        materialProperties.uniforms = parameters.uniforms;
    }

    auto &uniforms = materialProperties.uniforms;

    if (! instanceof <ShaderMaterial>(material) && ! instanceof <ShaderMaterial>(material) || material->clipping) {

        uniforms["clippingPlanes"] = clipping.uniform;
    }

    updateCommonMaterialProperties(material, parameters);

    // store the light setup it was created for

    materialProperties.needsLights = materialNeedsLights(material);
    materialProperties.lightsStateVersion = lightsStateVersion;

    if (materialProperties.needsLights) {

        // wire up the material to this renderer's lighting state

        uniforms["ambientLightColor"].setValue(lights.state.ambient);
        uniforms["lightProbe"].setValue(lights.state.probe);
        uniforms["directionalLights"].setValue(lights.state.directional);
        uniforms["directionalLightShadows"].setValue(lights.state.directionalShadow);
        uniforms["spotLights"].setValue(lights.state.spot);
        uniforms["spotLightShadows"].setValue(lights.state.spotShadow);
        //        uniforms["rectAreaLights"].setValue(lights.state.rectArea);
        //        uniforms["ltc_1"].setValue(lights.state.rectAreaLTC1);
        //        uniforms["ltc_2"].setValue(lights.state.rectAreaLTC2);
        uniforms["pointLights"].setValue(lights.state.point);
        uniforms["pointLightShadows"].setValue(lights.state.pointShadow);
        //        uniforms["hemisphereLights"].setValue(lights.state.hemi);

        uniforms["directionalShadowMap"].setValue(lights.state.directionalShadowMap);
        uniforms["directionalShadowMatrix"].setValue(lights.state.directionalShadowMatrix);
        uniforms["spotShadowMap"].setValue(lights.state.spotShadowMap);
        uniforms["spotShadowMatrix"].setValue(lights.state.spotShadowMatrix);
        uniforms["pointShadowMap"].setValue(lights.state.pointShadowMap);
        uniforms["pointShadowMatrix"].setValue(lights.state.pointShadowMatrix);
    }

    auto progUniforms = program->getUniforms();
    auto uniformsList = gl::GLUniforms::seqWithValue(progUniforms->seq, uniforms);

    materialProperties.currentProgram = program;
    materialProperties.uniformsList = uniformsList;

    return program;
}

void GLRenderer::updateCommonMaterialProperties(Material *material, gl::ProgramParameters &parameters) {

    auto &materialProperties = properties.materialProperties.get(material->uuid);

    materialProperties.outputEncoding = parameters.outputEncoding;
    materialProperties.instancing = parameters.instancing;
    materialProperties.numClippingPlanes = parameters.numClippingPlanes;
    materialProperties.numIntersection = parameters.numClipIntersection;
    materialProperties.vertexAlphas = parameters.vertexAlphas;
}


std::shared_ptr<gl::GLProgram> GLRenderer::setProgram(Camera *camera, Scene *scene, Material *material, Object3D *object) {

    //    bool isScene = instanceof <Scene>(scene);

    //            if (!isScene) scene = _emptyScene;// scene could be a Mesh, Line, Points, ...
    //

    bool isMeshBasicMaterial = instanceof <MeshBasicMaterial>(material);
    bool isMeshLambertMaterial = instanceof <MeshLambertMaterial>(material);
    bool isMeshToonMaterial = instanceof <MeshToonMaterial>(material);
    bool isMeshPhongMaterial = instanceof <MeshPhongMaterial>(material);
    bool isMeshStandardMaterial = instanceof <MeshStandardMaterial>(material);
    bool isShadowMaterial = instanceof <ShadowMaterial>(material);
    bool isShaderMaterial = instanceof <ShaderMaterial>(material);
    bool isEnvMap = instanceof <MaterialWithEnvMap>(material);

    textures.resetTextureUnits();

    auto fog = scene->fog;
    auto environment = instanceof <MeshStandardMaterial>(material) ? scene->environment : std::nullopt;
    int encoding = (_currentRenderTarget == nullptr) ? outputEncoding : _currentRenderTarget->texture.encoding;
    //                const envMap = cubemaps.get(material.envMap || environment);
    bool vertexAlphas = material->vertexColors && object->geometry() && object->geometry()->hasAttribute("color") && object->geometry()->getAttribute<float>("color")->itemSize() == 4;

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
    bool isInstancedMesh = instanceof <InstancedMesh>(object);

    if (material->version == materialProperties.version) {

        if (materialProperties.needsLights && (materialProperties.lightsStateVersion != lights.state.version)) {

            needsProgramChange = true;

        } else if (materialProperties.outputEncoding != encoding) {

            needsProgramChange = true;

        } else if (isInstancedMesh && materialProperties.instancing) {

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

    if (needsProgramChange == true) {

        program = getProgram(material, scene, object);
    }

    bool refreshProgram = false;
    bool refreshMaterial = false;
    bool refreshLights = false;

    auto p_uniforms = program->getUniforms();
    auto &m_uniforms = materialProperties.uniforms;

    if (state.useProgram(*program->program)) {

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

                auto uCamPos = p_uniforms->map["cameraPosition"];
                _vector3.setFromMatrixPosition(camera->matrixWorld);
                uCamPos->setValue(_vector3);
            }
        }

        if (isMeshPhongMaterial ||
            isMeshToonMaterial ||
            isMeshLambertMaterial ||
            isMeshBasicMaterial ||
            isMeshStandardMaterial ||
            isShaderMaterial) {

            p_uniforms->setValue("isOrthographic", instanceof <OrthographicCamera>(camera));
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

            markUniformsLightsNeedsUpdate(m_uniforms, refreshLights);
        }

        // refresh uniforms common to several materials

        if (fog && material->fog) {

            materials.refreshFogUniforms(m_uniforms, *fog);
        }

        materials.refreshMaterialUniforms(m_uniforms, material, _pixelRatio, (float) _height /*, _transmissionRenderTarget*/);

        gl::GLUniforms::upload(materialProperties.uniformsList, m_uniforms, &textures);
    }

    if (isShaderMaterial) {

        auto m = dynamic_cast<ShaderMaterial *>(material);
        if (m->uniformsNeedUpdate) {

            gl::GLUniforms::upload(materialProperties.uniformsList, m_uniforms, &textures);
            m->uniformsNeedUpdate = false;
        }
    }

    //                if (material.isSpriteMaterial) {
    //
    //                    p_uniforms->setValue("center", object.center);
    //                }

    // common matrices

    p_uniforms->setValue("modelViewMatrix", object->modelViewMatrix);
    p_uniforms->setValue("normalMatrix", object->normalMatrix);
    p_uniforms->setValue("modelMatrix", object->matrixWorld);

    return program;
}

void GLRenderer::markUniformsLightsNeedsUpdate(std::unordered_map<std::string, Uniform> &uniforms, bool value) {
    uniforms["ambientLightColor"].needsUpdate = value;
    uniforms["lightProbe"].needsUpdate = value;

    uniforms["directionalLights"].needsUpdate = value;
    uniforms["directionalLightShadows"].needsUpdate = value;
    uniforms["pointLights"].needsUpdate = value;
    uniforms["pointLightShadows"].needsUpdate = value;
    uniforms["spotLights"].needsUpdate = value;
    uniforms["spotLightShadows"].needsUpdate = value;
    uniforms["rectAreaLights"].needsUpdate = value;
    uniforms["hemisphereLights"].needsUpdate = value;
}

bool GLRenderer::materialNeedsLights(Material *material) {

    bool isMeshLambertMaterial = instanceof <MeshLambertMaterial>(material);
    bool isMeshToonMaterial = instanceof <MeshToonMaterial>(material);
    bool isMeshPhongMaterial = instanceof <MeshPhongMaterial>(material);
    bool isMeshStandardMaterial = instanceof <MeshStandardMaterial>(material);
    bool isShadowMaterial = instanceof <ShadowMaterial>(material);
    bool isShaderMaterial = instanceof <ShaderMaterial>(material);
    bool lights = false;
    if (instanceof <MaterialWithLights>(material)) {
        lights = dynamic_cast<MaterialWithLights *>(material)->lights;
    }

    return isMeshLambertMaterial || isMeshToonMaterial || isMeshPhongMaterial ||
           isMeshStandardMaterial || isShadowMaterial ||
           (isShaderMaterial && lights);
}


GLRenderer::OnMaterialDispose::OnMaterialDispose(GLRenderer &scope) : scope_(scope) {}

void GLRenderer::OnMaterialDispose::onEvent(Event &event) {
    auto material = static_cast<Material *>(event.target);

    material->removeEventListener("dispose", this);

    scope_.deallocateMaterial(material);
}
