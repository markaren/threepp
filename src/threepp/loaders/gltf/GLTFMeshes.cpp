// GLTFParser: primitive geometry, meshes, GPU instancing and the node tree.
// See GLTFParser.hpp.

#include "GLTFParser.hpp"

namespace threepp::gltf {

    // Decode one primitive's BufferGeometry (attributes, morph deltas,
    // index, computed normals). Cached by (meshIdx, primIdx, hasSkin) and
    // shared by every Mesh that references the primitive — the immutable
    // vertex data is decoded exactly once.
    std::shared_ptr<BufferGeometry> GLTFParser::buildPrimitiveGeometry(
            int meshIdx, int primIdx, const json& prim, bool hasSkin) {
        const std::tuple<int, int, bool> key{meshIdx, primIdx, hasSkin};
        if (auto it = geometryCache.find(key); it != geometryCache.end())
            return it->second;

        auto geometry = BufferGeometry::create();
        const auto& attrs = prim["attributes"];

        auto addFloatAttr = [&](const char* gltfKey, const char* threeKey, int itemSize) {
            if (!attrs.contains(gltfKey)) return;
            int accIdx = attrs[gltfKey].get<int>();
            auto data = readFloats(accIdx);
            geometry->setAttribute(
                    threeKey,
                    FloatBufferAttribute::create(std::move(data), itemSize));
        };

        addFloatAttr("POSITION", "position", 3);
        addFloatAttr("NORMAL", "normal", 3);
        addFloatAttr("TEXCOORD_0", "uv", 2);
        addFloatAttr("TEXCOORD_1", "uv2", 2);
        // COLOR_0: use actual accessor component count (VEC3 or VEC4).
        // The glTF-recommended encodings are normalized uint8/uint16 —
        // keep those narrow (1/4 resp. 1/2 the float footprint) instead
        // of baking the normalization into widened floats. Both
        // renderers expand them on fetch/upload; the normalized flag
        // rides on the attribute.
        if (attrs.contains("COLOR_0")) {
            int accIdx = attrs["COLOR_0"].get<int>();
            auto [ptr, stride, count, ct, nc, accNormalized] = getAccessor(accIdx);
            if (preserveNarrowAttributes && ct == COMP_UNSIGNED_BYTE && accNormalized) {
                geometry->setAttribute("color",
                        Uint8BufferAttribute::create(readNarrow<uint8_t>(accIdx), nc, true));
            } else if (preserveNarrowAttributes && ct == COMP_UNSIGNED_SHORT && accNormalized) {
                geometry->setAttribute("color",
                        Uint16BufferAttribute::create(readNarrow<uint16_t>(accIdx), nc, true));
            } else {
                geometry->setAttribute("color",
                        FloatBufferAttribute::create(readFloats(accIdx), nc));
            }
        }
        addFloatAttr("TANGENT", "tangent", 4);

        if (hasSkin) {
            if (attrs.contains("JOINTS_0")) {
                auto data = readJointIndicesAsFloat(attrs["JOINTS_0"].get<int>());
                geometry->setAttribute("skinIndex",
                        FloatBufferAttribute::create(std::move(data), 4));
            }
            addFloatAttr("WEIGHTS_0", "skinWeight", 4);
        }

        // --- Morph targets (POSITION/NORMAL deltas) ---
        // glTF morph targets are always relative (deltas from the base attribute).
        if (prim.contains("targets")) {
            const auto& targets = prim["targets"];
            for (const auto& target : targets) {
                if (target.contains("POSITION")) {
                    auto data = readFloats(target["POSITION"].get<int>());
                    geometry->getOrCreateMorphAttribute("position")
                            ->emplace_back(FloatBufferAttribute::create(std::move(data), 3));
                }
                if (target.contains("NORMAL")) {
                    auto data = readFloats(target["NORMAL"].get<int>());
                    geometry->getOrCreateMorphAttribute("normal")
                            ->emplace_back(FloatBufferAttribute::create(std::move(data), 3));
                }
            }
            if (!targets.empty()) {
                geometry->morphTargetsRelative = true;
            }
        }

        // --- Indices ---
        if (prim.contains("indices")) {
            auto indices = readIndices(prim["indices"].get<int>());
            geometry->setIndex(std::move(indices));
        }

        // Compute vertex normals if absent
        if (!attrs.contains("NORMAL")) {
            geometry->computeVertexNormals();
        }

        geometryCache[key] = geometry;
        return geometry;
    }

    // Exporters can emit primitives whose accessor references are
    // simply invalid: OpenCASCADE's RWGltf_CafWriter writes
    // {"attributes":{"POSITION":-1},"indices":-1} for faces it failed to
    // triangulate. Decoding one aborts the whole document, so a 165 MB
    // assembly is lost over a single degenerate face. Validate every
    // index a primitive references up front and skip just that
    // primitive, keeping the rest of the mesh.
    bool GLTFParser::primitiveAccessorsValid(int meshIdx, const std::string& meshName,
                                 int primIdx, const json& prim) const {
        auto reject = [&](const std::string& why) {
            std::cerr << "GLTFLoader: skipping primitive " << primIdx
                      << " of mesh " << meshIdx;
            if (!meshName.empty()) std::cerr << " ('" << meshName << "')";
            std::cerr << " - " << why << std::endl;
            return false;
        };

        if (!prim.contains("attributes") || !prim["attributes"].is_object())
            return reject("no attributes");

        for (auto it = prim["attributes"].begin(); it != prim["attributes"].end(); ++it) {
            if (!accessorIndexValid(it.value()))
                return reject("attribute " + it.key() + " references invalid accessor " +
                              it.value().dump());
        }

        if (prim.contains("indices") && !accessorIndexValid(prim["indices"]))
            return reject("indices reference invalid accessor " + prim["indices"].dump());

        if (prim.contains("targets")) {
            for (const auto& target : prim["targets"]) {
                if (!target.is_object()) continue;
                for (auto it = target.begin(); it != target.end(); ++it) {
                    if (!accessorIndexValid(it.value()))
                        return reject("morph target " + it.key() +
                                      " references invalid accessor " + it.value().dump());
                }
            }
        }

        return true;
    }

    std::shared_ptr<Object3D> GLTFParser::loadMesh(int meshIdx, bool hasSkin) {
        const auto& meshDef = gltf["meshes"][meshIdx];
        const auto& primitives = meshDef["primitives"];

        const std::string meshName = meshDef.value("name", "");
        std::vector<std::shared_ptr<Mesh>> meshes;

        int primIdx = 0;
        for (const auto& prim : primitives) {
            // glTF "mode" (default 4/TRIANGLES if absent): POINTS(0),
            // LINES(1), LINE_LOOP(2), LINE_STRIP(3), TRIANGLES(4),
            // TRIANGLE_STRIP(5), TRIANGLE_FAN(6). The renderers here
            // only understand flat triangle lists (raster draw calls
            // and the Vulkan RT BLAS both assume indexCount/vertexCount
            // is a triangle count) — feeding them a point cloud or line
            // strip mis-decodes as garbage triangles and, for vertex
            // counts not a multiple of 3, previously overran the
            // position buffer in computeVertexNormals(). Skip
            // unsupported topologies rather than misrender them.
            const int primMode = prim.value("mode", 4);
            if (primMode != 4) {
                std::cerr << "GLTFLoader: skipping primitive " << primIdx
                          << " of mesh " << meshIdx << " - unsupported mode "
                          << primMode << " (only TRIANGLES is supported)" << std::endl;
                ++primIdx;
                continue;
            }

            if (!primitiveAccessorsValid(meshIdx, meshName, primIdx, prim)) {
                ++primIdx;
                continue;
            }

            // Shared, cached geometry (decoded once per mesh/prim/skin).
            auto geometry = buildPrimitiveGeometry(meshIdx, primIdx, prim, hasSkin);
            const auto& attrs = prim["attributes"];

            // --- Material (cached by matIdx) ---
            std::shared_ptr<Material> mat;
            if (prim.contains("material") && gltf.contains("materials")) {
                mat = loadMaterial(prim["material"].get<int>());
            } else {
                // Default: white MeshStandardMaterial
                mat = MeshStandardMaterial::create();
            }

            std::shared_ptr<Mesh> mesh;
            if (hasSkin && attrs.contains("JOINTS_0"))
                mesh = SkinnedMesh::create(geometry, mat);
            else
                mesh = Mesh::create(geometry, mat);

            // Morph influences are per-Mesh: the morph delta attributes
            // live on the shared geometry; the weights selecting them do
            // not. Count comes from the primitive's target list.
            const size_t numMorphTargets =
                    prim.contains("targets") ? prim["targets"].size() : 0;
            if (numMorphTargets > 0) {
                auto& influences = mesh->morphTargetInfluences();
                influences.assign(numMorphTargets, 0.0f);
                // Initial weights live on the mesh, not the primitive.
                if (meshDef.contains("weights")) {
                    const auto& weights = meshDef["weights"];
                    const size_t n = std::min(numMorphTargets, weights.size());
                    for (size_t i = 0; i < n; ++i) {
                        influences[i] = weights[i].get<float>();
                    }
                }
            }

            // Tag for KHR_materials_variants post-load resolution (per-Mesh)
            mesh->userData["__gltfMeshIdx"] = meshIdx;
            mesh->userData["__gltfPrimIdx"] = primIdx;

            // Collect per-primitive variant mappings exactly once per
            // (mesh, prim): loadMesh may run for several referencing nodes,
            // and the mappings are appended — the guard prevents duplicates.
            if (!variantNames.empty() &&
                prim.contains("extensions") &&
                prim["extensions"].contains("KHR_materials_variants") &&
                variantsCollected.insert({meshIdx, primIdx}).second) {
                const auto& vext = prim["extensions"]["KHR_materials_variants"];
                if (vext.contains("mappings")) {
                    for (const auto& mapping : vext["mappings"]) {
                        PrimVariantMapping pvm;
                        pvm.materialIdx = mapping["material"].get<int>();
                        pvm.variantIndices = mapping["variants"].get<std::vector<int>>();
                        primVariantData[meshIdx][primIdx].push_back(std::move(pvm));
                    }
                }
            }

            meshes.push_back(mesh);
            ++primIdx;
        }

        // Single non-skinned primitive → return the Mesh directly (named
        // after the mesh), skipping a redundant Group wrapper + clone (the
        // old path cloned only because it built the Group first). Skinned
        // meshes stay wrapped in a Group so buildMeshObjForNode can bind
        // each SkinnedMesh child to the skeleton.
        if (meshes.size() == 1 && !hasSkin) {
            meshes[0]->name = meshName;
            return meshes[0];
        }

        auto group = Group::create();
        group->name = meshName;
        for (auto& m : meshes) group->add(m);
        return group;
    }

    // EXT_mesh_gpu_instancing — node-level extension carrying per-instance
    // TRANSLATION (vec3) / ROTATION (vec4 quat xyzw) / SCALE (vec3) accessor
    // arrays. Any subset may be present; missing components default to
    // identity. All provided accessors must share the same count.
    //
    // We walk `meshObj` (single Mesh or Group of Meshes from loadMesh) and
    // replace each non-skinned Mesh with an InstancedMesh sharing its
    // geometry + material. Per-instance matrices are composed from the TRS
    // accessors and uploaded into instanceMatrix. The replacement preserves
    // mesh name + userData so variant resolution and friends keep working.
    std::shared_ptr<Object3D> GLTFParser::applyGpuInstancing(const json& extData,
                                                const std::shared_ptr<Object3D>& meshObj) {
        if (!extData.contains("attributes") || !meshObj) return meshObj;
        const auto& attrs = extData["attributes"];

        // Count comes from any provided accessor (spec: all must match).
        size_t count = 0;
        for (const char* key : {"TRANSLATION", "ROTATION", "SCALE"}) {
            if (attrs.contains(key)) {
                count = gltf["accessors"][attrs[key].get<int>()]["count"].get<size_t>();
                break;
            }
        }
        if (count == 0) return meshObj;

        const std::vector<float> t = attrs.contains("TRANSLATION")
                ? readFloats(attrs["TRANSLATION"].get<int>()) : std::vector<float>();
        const std::vector<float> r = attrs.contains("ROTATION")
                ? readFloats(attrs["ROTATION"].get<int>()) : std::vector<float>();
        const std::vector<float> s = attrs.contains("SCALE")
                ? readFloats(attrs["SCALE"].get<int>()) : std::vector<float>();

        std::vector<Matrix4> mats(count);
        for (size_t i = 0; i < count; ++i) {
            Vector3 tv(0, 0, 0), sv(1, 1, 1);
            Quaternion qv;
            if (!t.empty()) tv.set(t[i * 3], t[i * 3 + 1], t[i * 3 + 2]);
            if (!r.empty()) qv.set(r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
            if (!s.empty()) sv.set(s[i * 3], s[i * 3 + 1], s[i * 3 + 2]);
            mats[i].compose(tv, qv, sv);
        }

        auto convert = [&](Mesh& m) -> std::shared_ptr<InstancedMesh> {
            // Skinned meshes can't reasonably share a single pose buffer
            // across instances; leave them as the regular Mesh.
            if (dynamic_cast<SkinnedMesh*>(&m)) return nullptr;
            auto inst = InstancedMesh::create(m.geometry(), m.material(), count);
            inst->name = m.name;
            inst->userData = m.userData;
            for (size_t i = 0; i < count; ++i) inst->setMatrixAt(i, mats[i]);
            inst->instanceMatrix()->needsUpdate();
            inst->computeBoundingSphere();
            return inst;
        };

        // Single-Mesh case (loadMesh unwrapped a single-primitive mesh).
        // Mesh and Group are siblings under Object3D, so as<Mesh>() is
        // null on a Group.
        if (auto* m = meshObj->as<Mesh>()) {
            if (auto inst = convert(*m)) return inst;
            return meshObj;
        }

        // Group-of-Meshes case: rebuild the group with InstancedMesh children.
        auto group = Group::create();
        group->name = meshObj->name;
        group->userData = meshObj->userData;
        for (auto* child : meshObj->children) {
            if (auto* cm = child->as<Mesh>()) {
                if (auto inst = convert(*cm)) {
                    group->add(inst);
                    continue;
                }
            }
            // Non-Mesh / SkinnedMesh: keep as-is, share the existing pointer.
            // children stores raw pointers; promote via shared_from_this if
            // available, otherwise we can't reattach safely — fall back to a
            // clone so we don't leave dangling refs.
            group->add(child->clone());
        }
        return group;
    }

    // Build the Mesh / multi-prim Group for a node (with skin + instancing
    // applied). Caller decides whether to attach it as a child of the node's
    // wrapper (joint case) or replace the wrapper outright (collapse case).
    std::shared_ptr<Object3D> GLTFParser::buildMeshObjForNode(int nodeIdx) {
        const auto& nodeDef = gltf["nodes"][nodeIdx];
        int meshIdx = nodeDef["mesh"].get<int>();
        int skinIdx = nodeDef.value("skin", -1);
        bool hasSkin = skinIdx >= 0 && gltf.contains("skins");

        auto meshObj = loadMesh(meshIdx, hasSkin);

        if (hasSkin) {
            auto skel = loadSkin(skinIdx);
            // meshObj is always a Group when hasSkin (no unwrap)
            for (auto child : meshObj->children) {
                if (auto sm = dynamic_cast<SkinnedMesh*>(child))
                    sm->bind(skel, Matrix4());
            }
        }

        // EXT_mesh_gpu_instancing — replace Mesh children with
        // InstancedMesh driven by the extension's per-instance TRS
        // accessor arrays. Skipped for skinned meshes (spec advises
        // against combining with KHR_skin).
        if (!hasSkin && nodeDef.contains("extensions") &&
            nodeDef["extensions"].contains("EXT_mesh_gpu_instancing")) {
            meshObj = applyGpuInstancing(
                    nodeDef["extensions"]["EXT_mesh_gpu_instancing"], meshObj);
        }

        return meshObj;
    }

    // For non-joint nodes with a mesh, replace the pre-created Group
    // wrapper with the actual meshObj — transferring transform + name.
    // This eliminates a redundant Object3D layer per mesh node
    // (Group("X") → Mesh("X") collapses to a single Mesh("X")).
    //
    // Joint nodes (Bones) keep their wrapper so skin binding / skeleton
    // wiring remains intact.
    void GLTFParser::tryCollapseMeshWrapper(int nodeIdx) {
        if (collapsedMeshNodes.count(nodeIdx)) return;
        const auto& nodeDef = gltf["nodes"][nodeIdx];
        if (!nodeDef.contains("mesh") || !gltf.contains("meshes")) return;
        if (jointNodeSet.count(nodeIdx)) return;

        auto meshObj = buildMeshObjForNode(nodeIdx);

        auto& wrapper = nodeObjects[nodeIdx];
        meshObj->position.copy(wrapper->position);
        meshObj->quaternion.copy(wrapper->quaternion);
        meshObj->scale.copy(wrapper->scale);
        // Adopt wrapper's name (set in preCreateNodes from the node's
        // glTF name or the synthetic "node_N" fallback). Animation
        // tracks reference nodes by this name — see loadAnimations.
        meshObj->name = wrapper->name;

        // Multi-primitive meshes (Group containing several Meshes):
        // propagate the node's name to inner primitives so traversal
        // by name still finds them.
        if (meshObj->children.size() > 1) {
            int idx = 0;
            for (auto* child : meshObj->children) {
                child->name = wrapper->name + "_" + std::to_string(idx++);
            }
        }

        nodeObjects[nodeIdx] = meshObj;
        collapsedMeshNodes.insert(nodeIdx);
    }

    // Build node hierarchy using pre-created node objects
    void GLTFParser::buildNode(int nodeIdx) {
        if (!builtNodes.insert(nodeIdx).second) return; // already built
        const auto& nodeDef = gltf["nodes"][nodeIdx];
        auto& obj = nodeObjects[nodeIdx];

        // Mesh attachment is only needed here for joints (Bones) that
        // also carry a mesh — non-joint mesh nodes were already
        // collapsed (their meshObj IS nodeObjects[nodeIdx]).
        if (!collapsedMeshNodes.count(nodeIdx) &&
            nodeDef.contains("mesh") && gltf.contains("meshes")) {

            auto meshObj = buildMeshObjForNode(nodeIdx);

            // DCC tools (Blender, Maya, ...) put the user-facing object name on
            // the glTF node; the mesh name is the mesh-data name and is often
            // generic ("Object_0"). When the node has an explicit name, prefer
            // it on the mesh container so traversal-by-name finds Blender names.
            if (nodeDef.contains("name")) {
                const auto nodeName = nodeDef["name"].get<std::string>();
                meshObj->name = nodeName;
                // For multi-primitive meshes (Group containing several Meshes),
                // also propagate the name to inner primitives so they're findable.
                if (!meshObj->children.empty()) {
                    int idx = 0;
                    for (auto* child : meshObj->children) {
                        child->name = nodeName + "_" + std::to_string(idx++);
                    }
                }
            }

            obj->add(meshObj);
        }

        // KHR_lights_punctual: extract lights from glTF nodes
        if (nodeDef.contains("extensions") &&
            nodeDef["extensions"].contains("KHR_lights_punctual")) {
            int lightIdx = nodeDef["extensions"]["KHR_lights_punctual"]["light"].get<int>();
            if (gltf.contains("extensions") &&
                gltf["extensions"].contains("KHR_lights_punctual") &&
                gltf["extensions"]["KHR_lights_punctual"].contains("lights")) {
                const auto& lightDef = gltf["extensions"]["KHR_lights_punctual"]["lights"][lightIdx];
                std::string ltype = lightDef.value("type", "point");
                float intensity = lightDef.value("intensity", 1.0f);
                Color color(1.f, 1.f, 1.f);
                if (lightDef.contains("color")) {
                    auto c = lightDef["color"].get<std::vector<float>>();
                    if (c.size() >= 3) color.setRGB(c[0], c[1], c[2]);
                }
                float range = lightDef.value("range", 0.0f);

                std::shared_ptr<Light> light;
                if (ltype == "directional") {
                    light = DirectionalLight::create(color, intensity);
                } else if (ltype == "spot") {
                    // Cone angles live in the nested "spot" object (KHR_lights_punctual §spot)
                    const json spotDef = lightDef.value("spot", json::object());
                    float innerCone = spotDef.value("innerConeAngle", 0.0f);
                    float outerCone = spotDef.value("outerConeAngle", math::PI / 4.f);
                    float penumbra = (outerCone > 0.f) ? (1.f - innerCone / outerCone) : 0.f;
                    light = SpotLight::create(color, intensity, range, outerCone, penumbra);
                } else {
                    // "point" or fallback
                    light = PointLight::create(color, intensity, range);
                }
                if (light) {
                    light->name = lightDef.value("name", "light_" + std::to_string(lightIdx));
                    light->visible = false;  // hidden by default; user opts in
                    obj->add(light);
                    std::cerr << "[GLTFLoader] Light: " << light->name
                              << " type=" << ltype << " intensity=" << intensity
                              << " range=" << range << std::endl;
                }
            }
        }

        if (nodeDef.contains("children")) {
            for (int ci : nodeDef["children"].get<std::vector<int>>())
                obj->add(nodeObjects[ci]);
        }
    }

    std::shared_ptr<Group> GLTFParser::loadScene(int sceneIdx) {
        const auto& sceneDef = gltf["scenes"][sceneIdx];
        auto root = Group::create();
        root->name = sceneDef.value("name", "Scene");

        if (sceneDef.contains("nodes")) {
            // Only nodes reachable from this scene's roots are built, so
            // meshes of unreachable nodes are never decoded. Ascending
            // order matches the original full 0..n iteration for the
            // reachable subset.
            const std::vector<int> reachable = collectReachable(sceneDef);

            // Pass 1: collapse Group-wrapper-around-Mesh layers for
            // every non-joint mesh node. Must run before buildNode's
            // child-attach so parent nodes pick up the collapsed
            // (Mesh) child, not the discarded Group wrapper.
            for (int i : reachable) tryCollapseMeshWrapper(i);

            // Pass 2: attach lights / joint-meshes / children.
            for (int i : reachable) buildNode(i);

            for (int nodeIdx : sceneDef["nodes"].get<std::vector<int>>())
                root->add(nodeObjects[nodeIdx]);
        }

        // Sketchfab/Blender exports often wrap a Mesh in a chain of named
        // groups (e.g. "Cone_2" → "Object_8" → Mesh), where only the topmost
        // wrapper carries the user-facing name. Walk up from each Mesh through
        // single-child wrapper groups and adopt the topmost wrapper's name so
        // traverseType<Mesh> / getObjectByName see Blender's names.
        //
        // Stop at the scene root: its name is the scene's name ("Scene" fallback)
        // and shouldn't propagate onto a single-child mesh.
        Object3D* rootPtr = root.get();
        root->traverseType<Mesh>([rootPtr](Mesh& m) {
            Object3D* candidate = &m;
            Object3D* cur = &m;
            while (cur->parent && cur->parent != rootPtr &&
                   cur->parent->children.size() == 1 &&
                   !dynamic_cast<Mesh*>(cur->parent)) {
                candidate = cur->parent;
                cur = cur->parent;
            }
            if (candidate != &m && !candidate->name.empty()) {
                m.name = candidate->name;
            }
        });

        return root;
    }

}// namespace threepp::gltf
