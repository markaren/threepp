// https://github.com/mrdoob/three.js/blob/r129/src/loaders/ObjectLoader.js
// https://github.com/mrdoob/three.js/blob/r129/src/loaders/MaterialLoader.js
// https://github.com/mrdoob/three.js/blob/r129/src/loaders/BufferGeometryLoader.js

#include "threepp/loaders/ObjectLoader.hpp"

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/BooleanKeyframeTrack.hpp"
#include "threepp/animation/tracks/ColorKeyframeTrack.hpp"
#include "threepp/animation/tracks/NumberKeyframeTrack.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/StringKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/geometries.hpp"
#include "threepp/geometries/LatheGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/PolyhedronGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/SogLoader.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatLod.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/materials/materials.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include "ObjectJsonConstants.hpp"
#include "ObjectJsonParse.hpp"
#include "threepp/utils/Base64.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <unordered_map>

using json = nlohmann::json;
using namespace threepp;
using namespace threepp::objectjson;

namespace {

    // -------------------------------------------------------------- objects

    struct ParseContext {

        const GeometryMap& geometries;
        const MaterialMap& materials;
        const TextureMap& textures;
        const AnimationMap& animations;
        Warnings& warnings;
        // Base directory for the paths of referenced assets, same as for
        // referenced images.
        const std::filesystem::path& resourcePath;
        // The archive this document came out of, and where it is. A linked
        // asset may be a member of it, and re-importing one has to be able to
        // say which archive it came from — see resolveLinkedAsset.
        const ZipReader* archive{nullptr};
        const std::filesystem::path& archivePath;
        // ObjectLoader::setSplatCloudResolver, consulted before a splat cloud
        // is loaded from its file. Null or empty means always load.
        const ObjectLoader::SplatCloudResolver* splatResolver{nullptr};
    };

    std::shared_ptr<BufferGeometry> lookupGeometry(const json& j, const ParseContext& ctx) {

        if (!j.contains("geometry") || !j["geometry"].is_string()) return nullptr;

        const auto it = ctx.geometries.find(j["geometry"].get<std::string>());
        if (it == ctx.geometries.end()) {
            ctx.warnings.add("undefined geometry '" + j["geometry"].get<std::string>() + "'");
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<Material>> lookupMaterials(const json& j, const ParseContext& ctx) {

        std::vector<std::shared_ptr<Material>> out;
        if (!j.contains("material")) return out;

        const auto resolve = [&](const std::string& uuid) -> std::shared_ptr<Material> {
            const auto it = ctx.materials.find(uuid);
            if (it == ctx.materials.end()) {
                ctx.warnings.add("undefined material '" + uuid + "'");
                return nullptr;
            }
            return it->second;
        };

        if (j["material"].is_array()) {
            for (const auto& entry : j["material"]) {
                if (auto m = resolve(entry.get<std::string>())) out.push_back(m);
            }
        } else if (j["material"].is_string()) {
            if (auto m = resolve(j["material"].get<std::string>())) out.push_back(m);
        }

        return out;
    }

    void applyInstanceAttribute(FloatBufferAttribute* target, const json& j) {

        if (!target || !j.contains("array")) return;

        const auto source = j["array"].get<std::vector<float>>();
        auto& dst = target->array();
        const auto n = std::min(source.size(), dst.size());
        std::copy_n(source.begin(), n, dst.begin());
        target->needsUpdate();
    }

    void applyShadow(LightWithShadow& light, const json& j) {

        if (!light.shadow) return;

        auto& shadow = *light.shadow;
        shadow.bias = value(j, "bias", shadow.bias);
        shadow.normalBias = value(j, "normalBias", shadow.normalBias);
        shadow.radius = value(j, "radius", shadow.radius);
        if (j.contains("mapSize")) shadow.mapSize = vector2From(j["mapSize"], shadow.mapSize);

        if (!j.contains("camera") || !shadow.camera) return;

        const auto& cam = j["camera"];
        // Keep the existing camera instance (its concrete type is dictated by
        // the light) but adopt the serialized identity and parameters.
        if (cam.contains("uuid")) shadow.camera->uuid = cam["uuid"].get<std::string>();
        shadow.camera->name = value<std::string>(cam, "name", "");
        shadow.camera->zoom = value(cam, "zoom", shadow.camera->zoom);
        shadow.camera->nearPlane = value(cam, "near", shadow.camera->nearPlane);
        shadow.camera->farPlane = value(cam, "far", shadow.camera->farPlane);

        if (auto* persp = dynamic_cast<PerspectiveCamera*>(shadow.camera.get())) {
            persp->fov = value(cam, "fov", persp->fov);
            persp->aspect = value(cam, "aspect", persp->aspect);
            persp->focus = value(cam, "focus", persp->focus);
        } else if (auto* ortho = dynamic_cast<OrthographicCamera*>(shadow.camera.get())) {
            ortho->left = value(cam, "left", ortho->left);
            ortho->right = value(cam, "right", ortho->right);
            ortho->top = value(cam, "top", ortho->top);
            ortho->bottom = value(cam, "bottom", ortho->bottom);
        }

        shadow.camera->updateProjectionMatrix();
    }

    std::shared_ptr<Object3D> createSplatCloud(const json& j, const ParseContext& ctx);

    std::shared_ptr<Object3D> createObject(const json& j, const ParseContext& ctx) {

        const auto type = value<std::string>(j, "type", "Object3D");

        auto geometry = lookupGeometry(j, ctx);
        auto materials = lookupMaterials(j, ctx);
        auto material = materials.empty() ? nullptr : materials.front();

        if (type == "Scene") {

            auto scene = Scene::create();

            if (j.contains("background")) {
                if (j["background"].is_number()) {
                    scene->background = Background(colorFrom(j["background"]));
                } else if (j["background"].is_string()) {
                    const auto it = ctx.textures.find(j["background"].get<std::string>());
                    if (it != ctx.textures.end()) scene->background = Background(it->second);
                }
            }

            if (j.contains("environment") && j["environment"].is_string()) {
                const auto it = ctx.textures.find(j["environment"].get<std::string>());
                if (it != ctx.textures.end()) scene->environment = it->second;
            }

            scene->backgroundBlurriness = value(j, "backgroundBlurriness", 0.f);
            scene->backgroundIntensity = value(j, "backgroundIntensity", 1.f);
            scene->environmentIntensity = value(j, "environmentIntensity", 1.f);
            const auto readEuler = [&](const char* key, Euler& e) {
                if (!j.contains(key) || !j[key].is_array() || j[key].size() < 3) return;
                const auto& a = j[key];
                static const char* orderNames[] = {"XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX"};
                auto order = Euler::default_order;
                if (a.size() > 3 && a[3].is_string()) {
                    for (int i = 0; i < 6; ++i) {
                        if (a[3].get<std::string>() == orderNames[i]) order = static_cast<Euler::RotationOrders>(i);
                    }
                }
                e.set(a[0].get<float>(), a[1].get<float>(), a[2].get<float>(), order);
            };
            readEuler("backgroundRotation", scene->backgroundRotation);
            readEuler("environmentRotation", scene->environmentRotation);

            if (j.contains("fog") && j["fog"].is_object()) {
                const auto& fog = j["fog"];
                const auto fogType = value<std::string>(fog, "type", "Fog");
                if (fogType == "FogExp2") {
                    scene->fog = FogExp2(colorFrom(fog["color"]), value(fog, "density", 0.00025f));
                } else {
                    scene->fog = Fog(colorFrom(fog["color"]), value(fog, "near", 1.f), value(fog, "far", 1000.f));
                }
            }

            return scene;
        }

        if (type == "PerspectiveCamera") {
            auto camera = PerspectiveCamera::create(
                    value(j, "fov", 50.f), value(j, "aspect", 1.f),
                    value(j, "near", 0.1f), value(j, "far", 2000.f));
            camera->zoom = value(j, "zoom", 1.f);
            camera->focus = value(j, "focus", 10.f);
            camera->filmGauge = value(j, "filmGauge", 35.f);
            camera->filmOffset = value(j, "filmOffset", 0.f);
            camera->updateProjectionMatrix();
            return camera;
        }

        if (type == "OrthographicCamera") {
            auto camera = OrthographicCamera::create(
                    value(j, "left", -1.f), value(j, "right", 1.f),
                    value(j, "top", 1.f), value(j, "bottom", -1.f),
                    value(j, "near", 0.1f), value(j, "far", 2000.f));
            camera->zoom = value(j, "zoom", 1.f);
            camera->updateProjectionMatrix();
            return camera;
        }

        if (type == "AmbientLight") {
            return AmbientLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f));
        }
        if (type == "DirectionalLight") {
            auto light = DirectionalLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f));
            if (j.contains("shadow")) applyShadow(*light, j["shadow"]);
            return light;
        }
        if (type == "PointLight") {
            auto light = PointLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f),
                                            value(j, "distance", 0.f), value(j, "decay", 1.f));
            if (j.contains("shadow")) applyShadow(*light, j["shadow"]);
            return light;
        }
        if (type == "SpotLight") {
            auto light = SpotLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f),
                                           value(j, "distance", 0.f), value(j, "angle", math::PI / 3),
                                           value(j, "penumbra", 0.f), value(j, "decay", 1.f));
            if (j.contains("shadow")) applyShadow(*light, j["shadow"]);
            return light;
        }
        if (type == "HemisphereLight") {
            return HemisphereLight::create(colorFrom(j.value("color", 0xffffffu)),
                                           colorFrom(j.value("groundColor", 0xffffffu)),
                                           value(j, "intensity", 1.f));
        }
        if (type == "RectAreaLight") {
            return RectAreaLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f),
                                         value(j, "width", 1.f), value(j, "height", 1.f));
        }

        if (type == "SkinnedMesh") {
            auto mesh = SkinnedMesh::create(geometry, material);
            if (materials.size() > 1) mesh->setMaterials(materials);
            mesh->bindMode = value<std::string>(j, "bindMode", "attached") == "detached"
                                     ? SkinnedMesh::BindMode::Detached
                                     : SkinnedMesh::BindMode::Attached;
            if (j.contains("bindMatrix")) {
                mesh->bindMatrix = matrixFrom(j["bindMatrix"]);
                mesh->bindMatrixInverse.copy(mesh->bindMatrix).invert();
            }
            return mesh;
        }

        if (type == "InstancedMesh") {
            auto mesh = InstancedMesh::create(geometry, material, value(j, "count", size_t(0)));
            if (materials.size() > 1) mesh->setMaterials(materials);
            if (j.contains("instanceMatrix")) applyInstanceAttribute(mesh->instanceMatrix(), j["instanceMatrix"]);
            if (j.contains("instanceColor") && mesh->count() > 0) {
                // Allocate the per-instance colour buffer through the public API
                // before filling it in. A count-0 mesh has no buffer to allocate
                // (and setColorAt would rightly throw), so skip it.
                Color c;
                mesh->setColorAt(0, c);
                applyInstanceAttribute(mesh->instanceColor(), j["instanceColor"]);
            }
            return mesh;
        }

        if (type == "Mesh") {
            auto mesh = Mesh::create(geometry, material);
            if (materials.size() > 1) mesh->setMaterials(materials);
            return mesh;
        }

        if (type == "LOD") return LOD::create();
        if (type == "LineSegments") return LineSegments::create(geometry, material);
        if (type == "LineLoop") return LineLoop::create(geometry, material);
        if (type == "Line") return Line::create(geometry, material);
        if (type == "Points") {
            auto points = Points::create(geometry ? geometry : BufferGeometry::create(),
                                         material ? material : PointsMaterial::create());
            if (materials.size() > 1) points->setMaterials(materials);
            return points;
        }
        if (type == "Sprite") {
            auto spriteMaterial = std::dynamic_pointer_cast<SpriteMaterial>(material);
            auto sprite = Sprite::create(spriteMaterial ? spriteMaterial : SpriteMaterial::create());
            if (j.contains("center")) sprite->center = vector2From(j["center"], {0.5f, 0.5f});
            return sprite;
        }
        if (type == "Group") return Group::create();
        if (type == "Bone") return Bone::create();
        if (type == "SplatCloud") return createSplatCloud(j, ctx);
        // A robot writes its type as the plain Object3D it also is, so a reader
        // without the articulation extension still gets the hierarchy — frozen,
        // exactly as before this existed. With it, the object comes back live.
        if (type == "Object3D") {
            if (j.contains("threeppRobot")) return Robot::create();
            return Object3D::create();
        }

        ctx.warnings.add("unsupported object type '" + type + "' - skipped");
        return nullptr;
    }

    // ------------------------------------------------------- articulation

    // Rebuild a Robot's joint table over the hierarchy that has just been
    // parsed. See ObjectExporter::writeArticulation for what is in the block and
    // why it is there; the short version is that this is what stops a document
    // round trip from having to re-import the URDF, which is what used to throw
    // away everything authored into a robot's subtree.
    //
    // The nodes are the document's, referenced by uuid. Robot's tables hold
    // shared_ptrs and its origPose_/articulatedJoints_ hold RAW pointers into
    // the same nodes, so the table has to be a co-owner: a URDF-built robot owns
    // its links and joints outright, and that ownership is what keeps those raw
    // pointers valid when a link is deleted out of the hierarchy. Sharing the
    // node's existing control block (shared_from_this) keeps that property
    // without duplicating the node.
    void restoreArticulation(Robot& robot, const json& j, Warnings& warnings) {

        std::unordered_map<std::string, Object3D*> byUuid;
        robot.traverse([&](Object3D& node) { byUuid.emplace(node.uuid, &node); });

        const auto find = [&](const json& entry, const char* key) -> Object3D* {
            if (!entry.contains(key) || !entry[key].is_string()) return nullptr;
            const auto it = byUuid.find(entry[key].get<std::string>());
            return it == byUuid.end() ? nullptr : it->second;
        };

        const auto refer = [](Object3D* node) {
            // Everything the loader builds is heap-shared; a node that is not
            // (one on the stack) can still be referred to, just not co-owned.
            if (node->weak_from_this().expired()) {
                return std::shared_ptr<Object3D>(node, [](Object3D*) {});
            }
            return node->shared_from_this();
        };

        std::size_t missing = 0;

        if (j.contains("links") && j["links"].is_array()) {
            for (const auto& uuid : j["links"]) {
                if (!uuid.is_string()) continue;
                const auto it = byUuid.find(uuid.get<std::string>());
                if (it == byUuid.end()) {
                    ++missing;
                    continue;
                }
                robot.addLink(refer(it->second));
            }
        }

        // Values are applied only after the tables are complete: setJointValue
        // indexes by DOF, and the DOF count is not known until the last joint
        // is in.
        std::vector<float> values;

        if (j.contains("joints") && j["joints"].is_array()) {
            for (const auto& entry : j["joints"]) {

                auto* node = find(entry, "node");
                if (!node) {
                    ++missing;
                    continue;
                }

                Robot::JointInfo info;
                info.name = value<std::string>(entry, "name", "");
                const auto type = value<std::string>(entry, "type", "fixed");
                info.type = type == "revolute"    ? Robot::JointType::Revolute
                            : type == "prismatic" ? Robot::JointType::Prismatic
                                                  : Robot::JointType::Fixed;
                info.parent = value<std::string>(entry, "parent", "");
                info.child = value<std::string>(entry, "child", "");

                if (entry.contains("axis") && entry["axis"].size() >= 3) {
                    const auto& a = entry["axis"];
                    info.axis.set(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
                }
                if (entry.contains("limit") && entry["limit"].size() >= 2) {
                    const auto& l = entry["limit"];
                    info.range = Robot::JointRange{l[0].get<float>(), l[1].get<float>()};
                }

                // The node is sitting in its DRIVEN pose, which is the document's
                // to keep; the rest pose is the one the table needs.
                Vector3 restPosition{node->position};
                Quaternion restRotation{node->quaternion};
                if (entry.contains("rest") && entry["rest"].size() >= 7) {
                    const auto& r = entry["rest"];
                    restPosition.set(r[0].get<float>(), r[1].get<float>(), r[2].get<float>());
                    restRotation.set(r[3].get<float>(), r[4].get<float>(),
                                     r[5].get<float>(), r[6].get<float>());
                }

                robot.addJoint(refer(node), info, restPosition, restRotation);

                if (info.type != Robot::JointType::Fixed) {
                    values.push_back(value(entry, "value", 0.f));
                }
            }
        }

        robot.finalizeInPlace();

        if (const auto ee = value<std::string>(j, "endEffector", ""); !ee.empty()) {
            robot.setEndEffector(ee);
        }

        // Re-driving from the stored values does not move anything — the nodes
        // are already in this pose — but it is what puts the same numbers in the
        // joint table, so an inspector reads the angles the robot is standing in
        // rather than a row of zeros.
        for (std::size_t i = 0; i < values.size() && i < robot.numDOF(); ++i) {
            robot.setJointValue(i, values[i]);
        }

        if (missing > 0) {
            warnings.add("robot '" + robot.name + "': the articulation table references " +
                         std::to_string(missing) +
                         " node(s) missing from the hierarchy - those joints came back frozen");
        }
    }

    // ------------------------------------------------- linked asset subtrees

    // The xacro arguments the editor recorded on the placeholder when the robot was
    // imported. Re-importing without them would rebuild a DIFFERENT robot — a UR5e
    // saved as a UR5e would come back as whatever the file defaults to. Nothing about
    // RobotConfig is known at this level, and nothing needs to be: the entries are
    // plain strings, read by the one reader the loaders layer owns.
    std::map<std::string, std::string> readXacroArgs(const Object3D& object) {

        std::map<std::string, std::string> args;
        for (const auto& [name, value] : xacro::readArgsUserData(object)) args[name] = value;
        return args;
    }

    // `why` collects the loader's own account of a failure, which for a xacro is
    // the difference between a usable report and "it did not load".
    std::shared_ptr<Object3D> importAsset(const std::filesystem::path& path,
                                          const std::map<std::string, std::string>& xacroArgs,
                                          std::string& why) {

        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (extension == ".urdf" || extension == ".xacro") {
            URDFLoader loader;
            if (!xacroArgs.empty()) loader.setArgs(xacroArgs);
            auto robot = loader.load(path);
            if (!robot) why = loader.lastError();
            return robot;
        }

        ModelLoader loader;
        return loader.load(path);
    }

    // Pre-order, root excluded — the exact walk ObjectExporter numbered the
    // override table against.
    std::vector<Object3D*> flattenDescendants(Object3D& root) {

        std::vector<Object3D*> flat;

        std::function<void(Object3D&)> collect = [&](Object3D& node) {
            for (auto* child : node.children) {
                if (!child) continue;
                flat.push_back(child);
                collect(*child);
            }
        };
        collect(root);

        return flat;
    }

    // Replays the edits the document recorded on top of a freshly imported
    // subtree. Each entry carries the name the node had when the document was
    // written; a mismatch means the asset file has changed underneath and the
    // index no longer identifies the same node, so the edit is dropped rather
    // than applied to whatever now sits at that position.
    void applyAssetOverrides(Object3D& root, const json& nodes, Warnings& warnings) {

        if (!nodes.is_array()) return;

        const auto flat = flattenDescendants(root);

        std::size_t stale = 0;

        for (const auto& entry : nodes) {

            if (!entry.contains("i") || !entry["i"].is_number_integer()) continue;

            const auto index = entry["i"].get<std::int64_t>();
            if (index < 0 || static_cast<std::size_t>(index) >= flat.size()) {
                ++stale;
                continue;
            }

            auto& node = *flat[static_cast<std::size_t>(index)];
            if (node.name != value<std::string>(entry, "name", "")) {
                ++stale;
                continue;
            }

            if (entry.contains("matrix")) {
                node.matrix->copy(matrixFrom(entry["matrix"]));
                node.matrix->decompose(node.position, node.quaternion, node.scale);
                node.rotation.setFromQuaternion(node.quaternion);
            }

            node.visible = value(entry, "visible", true);
            node.castShadow = value(entry, "castShadow", false);
            node.receiveShadow = value(entry, "receiveShadow", false);
            node.frustumCulled = value(entry, "frustumCulled", true);
            node.renderOrder = value(entry, "renderOrder", 0);

            if (entry.contains("layers") && entry["layers"].is_number()) {
                applyLayers(node, static_cast<unsigned int>(entry["layers"].get<std::int64_t>()));
            }
            if (entry.contains("userData")) applyUserData(node, entry["userData"], warnings);
        }

        // One message, not one per node: a changed asset invalidates the whole
        // table at once and a few thousand identical lines help nobody.
        if (stale > 0) {
            warnings.add("linked asset '" + root.name + "' has changed since the document was saved - " +
                         std::to_string(stale) + " of " + std::to_string(nodes.size()) +
                         " saved edits could not be matched to a node");
        }
    }

    // One archive member, written to a temp file for the length of an import.
    //
    // Every importer in the tree takes a path, and giving them all a second
    // entry point that takes bytes would be a far larger change than writing
    // the bytes down — which is also why only self-contained formats may travel
    // in an archive: a file whose siblings are missing would import to
    // something quietly wrong. The extension is kept because that is what
    // importAsset dispatches on, and the uuid is what keeps two loads (or two
    // processes) out of each other's way.
    class TempAsset {

    public:
        explicit TempAsset(const std::string& entry, const std::vector<unsigned char>& bytes) {

            const std::filesystem::path name{entry};
            path_ = std::filesystem::temp_directory_path() /
                    ("threepp-asset-" + math::generateUUID() + name.extension().string());

            std::ofstream out(path_, std::ios::binary | std::ios::trunc);
            if (!out) return;
            if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()),
                                          static_cast<std::streamsize>(bytes.size()));
            ok_ = out.good();
        }

        TempAsset(const TempAsset&) = delete;
        TempAsset& operator=(const TempAsset&) = delete;

        ~TempAsset() {

            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }

        [[nodiscard]] bool ok() const { return ok_; }

        [[nodiscard]] const std::filesystem::path& path() const { return path_; }

    private:
        std::filesystem::path path_;
        bool ok_{false};
    };

    // Returns the re-imported subtree, or nullptr when the asset could not be
    // loaded — in which case the caller keeps the empty placeholder, so the
    // rest of the scene still opens.
    std::shared_ptr<Object3D> resolveLinkedAsset(const json& ref, const Object3D& placeholder,
                                                 const ParseContext& ctx) {

        const auto stored = value<std::string>(ref, "path", "");
        if (stored.empty()) {
            ctx.warnings.add("linked asset '" + placeholder.name + "' has no path");
            return nullptr;
        }

        // An asset that lives inside an archive: a member of the one this
        // document came out of, or one named by a "<archive>|<entry>" mark that
        // an earlier load left behind — a document saved loose can still point
        // into an archive, and that path means what it says wherever it is read.
        std::string markArchive, markEntry;
        std::optional<ZipReader> other;
        const ZipReader* holder = nullptr;

        if (splitArchiveAsset(stored, markArchive, markEntry)) {

            try {

                other.emplace(markArchive);
                holder = &*other;

            } catch (const std::exception& e) {
                ctx.warnings.add("could not open archive '" + markArchive + "' for linked asset '" +
                                 placeholder.name + "': " + e.what());
                return nullptr;
            }

        } else if (ctx.archive && ctx.archive->has(stored)) {

            markArchive = ctx.archivePath.generic_string();
            markEntry = stored;
            holder = ctx.archive;
        }

        std::optional<TempAsset> extracted;
        if (holder) {

            if (!holder->has(markEntry)) {
                ctx.warnings.add("archive '" + markArchive + "' has no entry '" + markEntry + "'");
                return nullptr;
            }

            extracted.emplace(markEntry, holder->read(markEntry));
            if (!extracted->ok()) {
                ctx.warnings.add("could not extract '" + markEntry + "' from '" + markArchive + "'");
                return nullptr;
            }
        }

        std::filesystem::path path{stored};
        if (extracted) {

            path = extracted->path();

        } else {

            if (path.is_relative() && !ctx.resourcePath.empty()) path = ctx.resourcePath / path;

            // Normalised so that re-saving this document writes the same relative
            // path it was loaded from, rather than one with a '..' baked in.
            std::error_code ec;
            if (auto canonical = std::filesystem::weakly_canonical(path, ec); !ec && !canonical.empty()) {
                path = canonical;
            }
        }

        std::shared_ptr<Object3D> imported;
        std::string error;
        try {
            imported = importAsset(path, readXacroArgs(placeholder), error);
        } catch (const std::exception& e) {
            error = e.what();
        }

        if (!imported) {
            ctx.warnings.add("could not re-import linked asset '" + path.string() + "'" +
                             (error.empty() ? "" : ": " + error));
            return nullptr;
        }

        // Identity and placement belong to the document; everything below the
        // root belongs to the asset file.
        imported->uuid = placeholder.uuid;
        imported->name = placeholder.name;
        imported->matrix->copy(*placeholder.matrix);
        imported->matrix->decompose(imported->position, imported->quaternion, imported->scale);
        imported->rotation.setFromQuaternion(imported->quaternion);
        imported->matrixAutoUpdate = placeholder.matrixAutoUpdate;
        imported->castShadow = placeholder.castShadow;
        imported->receiveShadow = placeholder.receiveShadow;
        imported->visible = placeholder.visible;
        imported->frustumCulled = placeholder.frustumCulled;
        imported->renderOrder = placeholder.renderOrder;
        applyLayers(*imported, placeholder.layers.mask());
        imported->userData = placeholder.userData;

        // Where the asset actually is now, which is not necessarily where the
        // machine that saved the document had it — and for one that came out of
        // an archive, the archive and the entry rather than the temp file this
        // import read, which is gone by the time anything asks. That mark is
        // what lets a re-save copy the bytes straight across.
        setAssetSource(*imported, extracted ? markArchive + archiveAssetMark + markEntry : path.string());

        if (ref.contains("nodes")) applyAssetOverrides(*imported, ref["nodes"], ctx.warnings);

        return imported;
    }

    // ------------------------------------------------------- splat clouds

    // Where a `threeppSplat` path resolves to on this machine: an archive
    // member extracted to a temp file (kept alive by `extracted`), or a path
    // relative to the document made absolute. `mark` is what userData's source
    // entry should say afterwards — the archive-and-entry mark for a member,
    // the absolute path otherwise — so a re-save copies the same bytes.
    bool resolveSplatPath(const std::string& stored, const ParseContext& ctx,
                          std::optional<TempAsset>& extracted, std::filesystem::path& path,
                          std::string& mark, std::string& why) {

        std::string markArchive, markEntry;
        std::optional<ZipReader> other;
        const ZipReader* holder = nullptr;

        if (splitArchiveAsset(stored, markArchive, markEntry)) {

            try {
                other.emplace(markArchive);
                holder = &*other;
            } catch (const std::exception& e) {
                why = "could not open archive '" + markArchive + "': " + e.what();
                return false;
            }

        } else if (ctx.archive && ctx.archive->has(stored)) {

            markArchive = ctx.archivePath.generic_string();
            markEntry = stored;
            holder = ctx.archive;
        }

        if (holder) {

            if (!holder->has(markEntry)) {
                why = "archive '" + markArchive + "' has no entry '" + markEntry + "'";
                return false;
            }
            extracted.emplace(markEntry, holder->read(markEntry));
            if (!extracted->ok()) {
                why = "could not extract '" + markEntry + "' from '" + markArchive + "'";
                return false;
            }
            path = extracted->path();
            mark = markArchive + archiveAssetMark + markEntry;
            return true;
        }

        path = std::filesystem::u8path(stored);
        if (path.is_relative() && !ctx.resourcePath.empty()) path = ctx.resourcePath / path;
        std::error_code ec;
        if (auto canonical = std::filesystem::weakly_canonical(path, ec); !ec && !canonical.empty()) {
            path = canonical;
        }
        const auto u8 = path.u8string();
        mark = std::string(u8.begin(), u8.end());
        return true;
    }

    // A splat cloud, from the `threeppSplat` block ObjectExporter wrote: the
    // live object the resolver hands back (the play snapshot's case), else
    // the file loaded by its content — a SOG asset, a 3DGS .ply or a colour
    // point cloud — with the importer's ops replayed. Anything that fails
    // leaves a plain Object3D placeholder, placed and named, with a warning.
    std::shared_ptr<Object3D> createSplatCloud(const json& j, const ParseContext& ctx) {

        const std::string name = value<std::string>(j, "name", value<std::string>(j, "uuid", "?"));
        const auto placeholder = [&](const std::string& why) -> std::shared_ptr<Object3D> {
            ctx.warnings.add("splat cloud '" + name + "' could not be restored: " + why +
                             " - an empty node stands in for it");
            return Object3D::create();
        };

        if (!j.contains("threeppSplat") || !j["threeppSplat"].is_object()) {
            return placeholder("no threeppSplat block");
        }
        const json& ref = j["threeppSplat"];
        const float pointMix = value(ref, "pointMix", 0.f);
        const float pointSize = value(ref, "pointSize", 2.f);

        if (ctx.splatResolver && *ctx.splatResolver) {
            if (auto live = (*ctx.splatResolver)(value<std::string>(j, "uuid", ""))) {
                live->setPointMix(pointMix);
                live->setPointSize(pointSize);
                return live;
            }
        }

        const auto stored = value<std::string>(ref, "path", "");
        if (stored.empty()) {
            return placeholder("the document carries no file for it (a snapshot without its live object?)");
        }

        std::optional<TempAsset> extracted;
        std::filesystem::path path;
        std::string mark, why;
        if (!resolveSplatPath(stored, ctx, extracted, path, mark, why)) return placeholder(why);

        const bool cull = value(ref, "cull", false);
        const int lod = value(ref, "lod", -1);

        SplatData data;
        splats::LodTable table;
        try {

            if (SogLoader::isSog(path)) {
                if (lod >= 0) {
                    data = SogLoader::load(path, {lod});
                } else {
                    auto loaded = splats::loadSogWithLod(path);
                    data = std::move(loaded.data);
                    table = std::move(loaded.table);
                }
            } else if (SplatLoader::isSplatPly(path)) {
                data = SplatLoader::loadPly(path);
            } else if (SplatLoader::isPointCloudPly(path)) {
                data = SplatLoader::loadPointCloudPly(path);
            } else {
                std::error_code ec;
                return placeholder(std::filesystem::exists(path, ec)
                                           ? "'" + path.string() + "' is not a splat scan or a point cloud"
                                           : "'" + path.string() + "' does not exist");
            }

        } catch (const std::exception& e) {
            return placeholder(e.what());
        }

        // The importer's cull, replayed on the same data: deterministic, so
        // the same splats go. Never under a LOD table, whose offsets it would
        // invalidate (the editor's import makes the same exception).
        if (cull && table.empty()) data.removeOutliers();

        auto cloud = SplatCloud::create(std::move(data));
        if (!table.empty()) cloud->setLodTable(std::move(table));
        cloud->setPointMix(pointMix);
        cloud->setPointSize(pointSize);
        // Where the file is NOW, for the inspector and the next save. parseObject
        // overwrites userData from the document right after this; restampSplatSource
        // puts it back.
        cloud->userData[splatSourceKey] = mark;
        return cloud;
    }

    // After the document's userData landed on the cloud: the source entry it
    // carries is the saving machine's absolute path, and the one createSplatCloud
    // resolved is this machine's. Same rule as assetSource on a linked model.
    void restampSplatSource(Object3D& object, const json& j, const ParseContext& ctx) {

        auto* cloud = dynamic_cast<SplatCloud*>(&object);
        if (!cloud || !j.contains("threeppSplat")) return;
        const auto stored = value<std::string>(j["threeppSplat"], "path", "");
        if (stored.empty()) return;

        std::optional<TempAsset> extracted;// not extracted here: has() decides the mark
        std::string markArchive, markEntry, mark;
        if (splitArchiveAsset(stored, markArchive, markEntry)) {
            mark = stored;
        } else if (ctx.archive && ctx.archive->has(stored)) {
            mark = ctx.archivePath.generic_string() + archiveAssetMark + stored;
        } else {
            std::filesystem::path path = std::filesystem::u8path(stored);
            if (path.is_relative() && !ctx.resourcePath.empty()) path = ctx.resourcePath / path;
            std::error_code ec;
            if (auto canonical = std::filesystem::weakly_canonical(path, ec); !ec && !canonical.empty()) {
                path = canonical;
            }
            const auto u8 = path.u8string();
            mark = std::string(u8.begin(), u8.end());
        }
        object.userData[splatSourceKey] = mark;
    }

    std::shared_ptr<Object3D> parseObject(const json& j, const ParseContext& ctx) {

        auto object = createObject(j, ctx);
        if (!object) return nullptr;

        if (j.contains("uuid")) object->uuid = j["uuid"].get<std::string>();
        object->name = value<std::string>(j, "name", "");

        if (j.contains("matrix")) {
            const auto matrix = matrixFrom(j["matrix"]);
            object->matrix->copy(matrix);
            object->matrix->decompose(object->position, object->quaternion, object->scale);
            object->rotation.setFromQuaternion(object->quaternion);
            object->matrixAutoUpdate = value(j, "matrixAutoUpdate", true);
        }

        object->castShadow = value(j, "castShadow", false);
        object->receiveShadow = value(j, "receiveShadow", false);
        object->visible = value(j, "visible", true);
        object->frustumCulled = value(j, "frustumCulled", true);
        object->renderOrder = value(j, "renderOrder", 0);

        if (j.contains("layers") && j["layers"].is_number()) {
            applyLayers(*object, static_cast<unsigned int>(j["layers"].get<std::int64_t>()));
        }

        if (j.contains("userData")) applyUserData(*object, j["userData"], ctx.warnings);
        restampSplatSource(*object, j, ctx);

        if (j.contains("animations")) {
            for (const auto& entry : j["animations"]) {
                if (!entry.is_string()) continue;
                const auto it = ctx.animations.find(entry.get<std::string>());
                if (it != ctx.animations.end()) object->animations.push_back(it->second);
            }
        }

        // A linked subtree has no children in the document — it has a file to
        // read them from. On failure the placeholder stands: correctly placed
        // and named, just empty, and the warning says why.
        if (j.contains("threeppAsset")) {
            if (auto linked = resolveLinkedAsset(j["threeppAsset"], *object, ctx)) return linked;
            return object;
        }

        std::vector<std::shared_ptr<Object3D>> children;
        if (j.contains("children")) {
            for (const auto& entry : j["children"]) {
                if (auto child = parseObject(entry, ctx)) children.push_back(child);
            }
        }

        auto* lod = object->as<LOD>();

        if (lod && j.contains("levels")) {

            lod->autoUpdate = value(j, "autoUpdate", true);

            std::vector<bool> consumed(children.size(), false);

            for (const auto& level : j["levels"]) {
                const auto uuid = value<std::string>(level, "object", "");
                for (size_t i = 0; i < children.size(); ++i) {
                    if (consumed[i] || children[i]->uuid != uuid) continue;
                    lod->addLevel(children[i], value(level, "distance", 0.f));
                    consumed[i] = true;
                    break;
                }
            }

            for (size_t i = 0; i < children.size(); ++i) {
                if (!consumed[i]) object->add(children[i]);
            }

        } else {
            for (const auto& child : children) object->add(child);
        }

        // After the children: the joint table names the nodes it drives, and
        // they have to be in the tree before they can be found.
        if (j.contains("threeppRobot")) {
            if (auto* robot = object->as<Robot>()) {
                restoreArticulation(*robot, j["threeppRobot"], ctx.warnings);
            }
        }

        return object;
    }

    SkeletonMap parseSkeletons(const json& j, Object3D& root, Warnings& warnings) {

        SkeletonMap skeletons;
        if (!j.is_array()) return skeletons;

        std::unordered_map<std::string, Bone*> bones;
        root.traverse([&](Object3D& o) {
            if (auto* bone = o.as<Bone>()) bones[bone->uuid] = bone;
        });

        for (const auto& entry : j) {

            std::vector<std::shared_ptr<Bone>> boneList;
            std::vector<Matrix4> boneInverses;

            if (entry.contains("bones")) {
                for (const auto& uuid : entry["bones"]) {
                    const auto it = bones.find(uuid.get<std::string>());
                    if (it == bones.end()) {
                        warnings.add("undefined bone '" + uuid.get<std::string>() + "'");
                        continue;
                    }
                    // Non-owning: the bone is already owned by the object tree.
                    boneList.emplace_back(it->second, [](Bone*) {});
                }
            }

            if (entry.contains("boneInverses")) {
                for (const auto& m : entry["boneInverses"]) boneInverses.push_back(matrixFrom(m));
            }

            auto skeleton = Skeleton::create(boneList, boneInverses);
            if (entry.contains("uuid")) skeleton->setUuid(entry["uuid"].get<std::string>());

            skeletons[skeleton->uuid()] = skeleton;
        }

        return skeletons;
    }

    void bindSkeletons(Object3D& root, const SkeletonMap& skeletons, const json& objectJson, Warnings& warnings) {

        if (skeletons.empty()) return;

        // The skeleton uuid lives on the SkinnedMesh JSON entry, so walk the
        // document in parallel with the tree.
        std::unordered_map<std::string, std::string> meshToSkeleton;

        const std::function<void(const json&)> collect = [&](const json& j) {
            if (j.contains("skeleton") && j.contains("uuid")) {
                meshToSkeleton[j["uuid"].get<std::string>()] = j["skeleton"].get<std::string>();
            }
            if (j.contains("children")) {
                for (const auto& child : j["children"]) collect(child);
            }
        };
        collect(objectJson);

        root.traverse([&](Object3D& o) {
            auto* mesh = o.as<SkinnedMesh>();
            if (!mesh) return;

            const auto it = meshToSkeleton.find(mesh->uuid);
            if (it == meshToSkeleton.end()) return;

            const auto skeleton = skeletons.find(it->second);
            if (skeleton == skeletons.end()) {
                warnings.add("undefined skeleton '" + it->second + "'");
                return;
            }

            mesh->bind(skeleton->second, mesh->bindMatrix);
        });
    }

}// namespace

void ObjectLoader::setResourcePath(const std::filesystem::path& path) {

    resourcePath_ = path;
}

void ObjectLoader::setSplatCloudResolver(SplatCloudResolver resolver) {

    splatResolver_ = std::move(resolver);
}

const std::vector<std::string>& ObjectLoader::warnings() const {

    return warnings_;
}

std::shared_ptr<Object3D> ObjectLoader::parse(const std::string& jsonText) {

    warnings_.clear();

    Warnings warnings;

    const auto j = json::parse(jsonText, nullptr, false);

    if (j.is_discarded() || !j.is_object()) {
        warnings.add("malformed JSON");
        warnings_ = std::move(warnings.messages);
        return nullptr;
    }

    if (!j.contains("object")) {
        warnings.add("document has no 'object' entry");
        warnings_ = std::move(warnings.messages);
        return nullptr;
    }

    const DocumentSource source{resourcePath_, archive_.get()};
    BufferCache buffers{source};

    const auto animations = parseAnimations(j.contains("animations") ? j["animations"] : json(), warnings);
    const auto geometries = parseGeometries(j.contains("geometries") ? j["geometries"] : json(), buffers, warnings);
    const auto images = parseImages(j.contains("images") ? j["images"] : json(), source, warnings);
    const auto textures = parseTextures(j.contains("textures") ? j["textures"] : json(), images);
    const auto materials = parseMaterials(j.contains("materials") ? j["materials"] : json(), textures, warnings);

    const ParseContext ctx{geometries, materials, textures, animations, warnings,
                           resourcePath_, archive_.get(), archivePath_, &splatResolver_};

    auto object = parseObject(j["object"], ctx);

    if (object) {
        const auto skeletons = parseSkeletons(j.contains("skeletons") ? j["skeletons"] : json(), *object, warnings);
        bindSkeletons(*object, skeletons, j["object"], warnings);
    }

    warnings_ = std::move(warnings.messages);

    return object;
}

std::shared_ptr<Object3D> ObjectLoader::load(const std::filesystem::path& path) {

    std::string text;

    // Sniffed, not decided by the extension: a .tpz is a zip whatever it is
    // called, and a JSON document is not one however it is named.
    if (ZipReader::looksLikeZip(path)) {

        try {

            archive_ = std::make_shared<ZipReader>(path);

        } catch (const std::exception& e) {

            std::cerr << "[ObjectLoader] " << e.what() << std::endl;
            return nullptr;
        }

        if (!archive_->has(objectjson::archiveDocument)) {

            std::cerr << "[ObjectLoader] " << path.string() << " is an archive with no "
                      << objectjson::archiveDocument << std::endl;
            archive_.reset();
            return nullptr;
        }

        const auto bytes = archive_->read(objectjson::archiveDocument);
        text.assign(bytes.begin(), bytes.end());

        // Absolute, because it goes into the assetSource mark of anything
        // re-imported out of it and that mark outlives this call.
        std::error_code ec;
        auto absolute = std::filesystem::weakly_canonical(path, ec);
        archivePath_ = ec || absolute.empty() ? path : absolute;

    } else {

        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "[ObjectLoader] unable to open " << path.string() << std::endl;
            return nullptr;
        }

        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    // Default the base directory to this document's, without clobbering an
    // explicit setResourcePath() or leaking into the next load(). An archive
    // resolves its own urls, but the document may still point outside itself —
    // at a linked asset, at an image it was saved before the archive existed —
    // and those are relative to the archive's directory.
    const auto configured = resourcePath_;
    if (resourcePath_.empty()) resourcePath_ = path.parent_path();

    auto object = parse(text);

    resourcePath_ = configured;
    archive_.reset();
    archivePath_.clear();

    return object;
}
