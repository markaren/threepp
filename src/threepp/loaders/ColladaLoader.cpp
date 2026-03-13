
#include "threepp/loaders/ColladaLoader.hpp"

#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "pugixml.hpp"

#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace threepp;

namespace {

    std::vector<float> parseFloatArray(const std::string& text) {
        std::vector<float> result;
        std::istringstream ss(text);
        float val;
        while (ss >> val) result.push_back(val);
        return result;
    }

    std::vector<int> parseIntArray(const std::string& text) {
        std::vector<int> result;
        std::istringstream ss(text);
        int val;
        while (ss >> val) result.push_back(val);
        return result;
    }

    std::string stripHash(const std::string& s) {
        if (!s.empty() && s[0] == '#') return s.substr(1);
        return s;
    }

    struct Source {
        std::vector<float> data;
        int stride{3};
    };

    struct Input {
        std::string semantic;
        std::string source;// id without '#'
        int offset{0};
        int set{0};
    };

    struct Primitive {
        std::string type;// "triangles" or "polylist"
        std::string material;
        int count{0};
        std::vector<Input> inputs;
        std::vector<int> p;
        std::vector<int> vcount;// for polylist
        int maxOffset{0};
    };

    struct GeometryData {
        std::string name;
        std::unordered_map<std::string, Source> sources;
        // <vertices id="..."> maps to a source id (POSITION semantic)
        std::unordered_map<std::string, std::string> verticesMap;
        std::vector<Primitive> primitives;
    };

    struct EffectData {
        Color diffuse{1.f, 1.f, 1.f};
        Color ambient{0.f, 0.f, 0.f};
        Color specular{1.f, 1.f, 1.f};
        float shininess{30.f};
        float transparency{1.f};
        std::string diffuseTextureId;// image id
        TextureWrapping wrapS{TextureWrapping::Repeat};
        TextureWrapping wrapT{TextureWrapping::Repeat};
    };

    TextureWrapping parseWrapMode(const std::string& value) {
        if (value == "MIRROR") return TextureWrapping::MirroredRepeat;
        if (value == "CLAMP" || value == "BORDER" || value == "NONE") return TextureWrapping::ClampToEdge;
        return TextureWrapping::Repeat;// WRAP or unspecified
    }

    Color parseColorNode(const pugi::xml_node& node) {
        const std::string text = node.child_value();
        auto parts = utils::split(text, ' ');
        if (parts.size() >= 3) {
            return Color(utils::parseFloat(parts[0]),
                         utils::parseFloat(parts[1]),
                         utils::parseFloat(parts[2]));
        }
        return Color(1.f, 1.f, 1.f);
    }

    void parseSources(const pugi::xml_node& meshNode, GeometryData& geo) {
        for (const auto& source : meshNode.children("source")) {
            const std::string id = source.attribute("id").value();
            const auto floatArray = source.child("float_array");
            if (!floatArray) continue;

            const auto technique = source.child("technique_common");
            const auto accessor = technique ? technique.child("accessor") : pugi::xml_node{};
            int stride = 3;
            if (accessor) {
                stride = accessor.attribute("stride").as_int(3);
            }

            Source src;
            src.data = parseFloatArray(floatArray.child_value());
            src.stride = stride;
            geo.sources[id] = std::move(src);
        }
    }

    void parseVertices(const pugi::xml_node& meshNode, GeometryData& geo) {
        for (const auto& verts : meshNode.children("vertices")) {
            const std::string id = verts.attribute("id").value();
            for (const auto& input : verts.children("input")) {
                const std::string semantic = input.attribute("semantic").value();
                if (semantic == "POSITION") {
                    geo.verticesMap[id] = stripHash(input.attribute("source").value());
                }
            }
        }
    }

    std::vector<Input> parseInputs(const pugi::xml_node& prim) {
        std::vector<Input> inputs;
        for (const auto& inp : prim.children("input")) {
            Input i;
            i.semantic = inp.attribute("semantic").value();
            i.source = stripHash(inp.attribute("source").value());
            i.offset = inp.attribute("offset").as_int(0);
            i.set = inp.attribute("set").as_int(0);
            inputs.push_back(i);
        }
        return inputs;
    }

    int maxOffset(const std::vector<Input>& inputs) {
        int m = 0;
        for (const auto& i : inputs) m = std::max(m, i.offset);
        return m;
    }

    void parsePrimitives(const pugi::xml_node& meshNode, GeometryData& geo) {
        // triangles
        for (const auto& tri : meshNode.children("triangles")) {
            Primitive p;
            p.type = "triangles";
            p.material = tri.attribute("material").value();
            p.count = tri.attribute("count").as_int(0);
            p.inputs = parseInputs(tri);
            p.maxOffset = maxOffset(p.inputs);
            p.p = parseIntArray(tri.child("p").child_value());
            geo.primitives.push_back(std::move(p));
        }

        // polylist
        for (const auto& poly : meshNode.children("polylist")) {
            Primitive p;
            p.type = "polylist";
            p.material = poly.attribute("material").value();
            p.count = poly.attribute("count").as_int(0);
            p.inputs = parseInputs(poly);
            p.maxOffset = maxOffset(p.inputs);
            p.vcount = parseIntArray(poly.child("vcount").child_value());
            p.p = parseIntArray(poly.child("p").child_value());
            geo.primitives.push_back(std::move(p));
        }

        // lines
        // (not handled here — only triangles and polylist)
    }

    GeometryData parseGeometry(const pugi::xml_node& geomNode) {
        GeometryData geo;
        geo.name = geomNode.attribute("name").value();

        const auto meshNode = geomNode.child("mesh");
        if (!meshNode) return geo;

        parseSources(meshNode, geo);
        parseVertices(meshNode, geo);
        parsePrimitives(meshNode, geo);

        return geo;
    }

    // Resolve the actual source id for a given input, accounting for <vertices> indirection
    const Source* resolveSource(const Input& input, const GeometryData& geo) {
        std::string sourceId = input.source;

        // If this is a VERTEX input, look up via vertices map
        if (input.semantic == "VERTEX") {
            auto it = geo.verticesMap.find(sourceId);
            if (it != geo.verticesMap.end()) sourceId = it->second;
        }

        auto it = geo.sources.find(sourceId);
        if (it == geo.sources.end()) return nullptr;
        return &it->second;
    }

    std::shared_ptr<BufferGeometry> buildGeometry(const GeometryData& geo, const std::string& materialSymbol) {
        // Find the primitive matching the material symbol (or first one)
        const Primitive* prim = nullptr;
        for (const auto& p : geo.primitives) {
            if (p.material == materialSymbol || prim == nullptr) {
                prim = &p;
                if (p.material == materialSymbol) break;
            }
        }
        if (!prim) return nullptr;

        // Find inputs
        const Input* posInput = nullptr;
        const Input* normalInput = nullptr;
        const Input* uvInput = nullptr;
        for (const auto& inp : prim->inputs) {
            if (inp.semantic == "VERTEX") posInput = &inp;
            else if (inp.semantic == "NORMAL") normalInput = &inp;
            else if (inp.semantic == "TEXCOORD" && inp.set == 0) uvInput = &inp;
        }

        const Source* posSource = posInput ? resolveSource(*posInput, geo) : nullptr;
        const Source* normalSource = normalInput ? resolveSource(*normalInput, geo) : nullptr;
        const Source* uvSource = uvInput ? resolveSource(*uvInput, geo) : nullptr;

        if (!posSource) return nullptr;

        int stride = prim->maxOffset + 1;

        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> uvs;

        auto addVertex = [&](int idx) {
            if (posInput) {
                int base = prim->p[idx * stride + posInput->offset] * posSource->stride;
                positions.push_back(posSource->data[base]);
                positions.push_back(posSource->data[base + 1]);
                positions.push_back(posSource->data[base + 2]);
            }
            if (normalSource && normalInput) {
                int base = prim->p[idx * stride + normalInput->offset] * normalSource->stride;
                normals.push_back(normalSource->data[base]);
                normals.push_back(normalSource->data[base + 1]);
                normals.push_back(normalSource->data[base + 2]);
            }
            if (uvSource && uvInput) {
                int base = prim->p[idx * stride + uvInput->offset] * uvSource->stride;
                uvs.push_back(uvSource->data[base]);
                uvs.push_back(uvSource->data[base + 1]);
            }
        };

        if (prim->type == "triangles") {
            int vertexCount = prim->count * 3;
            for (int i = 0; i < vertexCount; i++) {
                addVertex(i);
            }
        } else if (prim->type == "polylist") {
            int vi = 0;// vertex index into p
            for (int polyIdx = 0; polyIdx < prim->count; polyIdx++) {
                int vc = prim->vcount.empty() ? 3 : prim->vcount[polyIdx];
                // Fan triangulation from vertex 0
                for (int t = 1; t < vc - 1; t++) {
                    addVertex(vi);
                    addVertex(vi + t);
                    addVertex(vi + t + 1);
                }
                vi += vc;
            }
        }

        auto geometry = std::make_shared<BufferGeometry>();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        if (!normals.empty()) {
            geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        }
        if (!uvs.empty()) {
            geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        }

        return geometry;
    }

    EffectData parseEffect(const pugi::xml_node& effectNode) {
        EffectData effect;

        // Look for phong/lambert/blinn shading model
        auto profile = effectNode.child("profile_COMMON");
        if (!profile) return effect;

        // Build newparam maps: sampler sid -> surface sid, surface sid -> image id
        std::unordered_map<std::string, std::string> samplerToSurface;
        std::unordered_map<std::string, std::string> surfaceToImage;
        // sampler sid -> {surface sid, wrap_s, wrap_t}
        struct SamplerInfo { std::string surface; std::string wrapS, wrapT; };
        std::unordered_map<std::string, SamplerInfo> samplers;

        for (const auto& param : profile.children("newparam")) {
            const std::string sid = param.attribute("sid").value();
            if (const auto sampler = param.child("sampler2D")) {
                SamplerInfo info;
                info.surface = sampler.child_value("source");
                info.wrapS = sampler.child_value("wrap_s");
                info.wrapT = sampler.child_value("wrap_t");
                samplers[sid] = std::move(info);
                samplerToSurface[sid] = samplers[sid].surface;
            } else if (const auto surface = param.child("surface")) {
                surfaceToImage[sid] = surface.child_value("init_from");
            }
        }

        // Helper to resolve a texture attribute value to an image id, and capture wrap modes
        auto resolveTexture = [&](const std::string& texAttr, EffectData& eff) -> std::string {
            auto sampIt = samplers.find(texAttr);
            if (sampIt != samplers.end()) {
                if (!sampIt->second.wrapS.empty()) eff.wrapS = parseWrapMode(sampIt->second.wrapS);
                if (!sampIt->second.wrapT.empty()) eff.wrapT = parseWrapMode(sampIt->second.wrapT);
                auto imgIt = surfaceToImage.find(sampIt->second.surface);
                if (imgIt != surfaceToImage.end()) return imgIt->second;
                return "";
            }
            return texAttr;// might already be an image id
        };

        // Check for technique
        auto technique = profile.child("technique");
        if (!technique) return effect;

        pugi::xml_node shading;
        for (const char* model : {"phong", "lambert", "blinn", "constant"}) {
            shading = technique.child(model);
            if (shading) break;
        }
        if (!shading) return effect;

        // diffuse
        auto diffuseNode = shading.child("diffuse");
        if (diffuseNode) {
            auto colorNode = diffuseNode.child("color");
            if (colorNode) effect.diffuse = parseColorNode(colorNode);
            auto texNode = diffuseNode.child("texture");
            if (texNode) effect.diffuseTextureId = resolveTexture(texNode.attribute("texture").value(), effect);
        }

        // ambient
        auto ambientNode = shading.child("ambient");
        if (ambientNode) {
            auto colorNode = ambientNode.child("color");
            if (colorNode) effect.ambient = parseColorNode(colorNode);
        }

        // specular
        auto specularNode = shading.child("specular");
        if (specularNode) {
            auto colorNode = specularNode.child("color");
            if (colorNode) effect.specular = parseColorNode(colorNode);
        }

        // shininess
        auto shininessNode = shading.child("shininess");
        if (shininessNode) {
            auto floatNode = shininessNode.child("float");
            if (floatNode) effect.shininess = utils::parseFloat(floatNode.child_value());
        }

        // transparency
        auto transparencyNode = shading.child("transparency");
        if (transparencyNode) {
            auto floatNode = transparencyNode.child("float");
            if (floatNode) effect.transparency = utils::parseFloat(floatNode.child_value());
        }

        return effect;
    }

    std::shared_ptr<MeshPhongMaterial> buildMaterial(const EffectData& effect) {
        auto mat = MeshPhongMaterial::create();
        mat->color.copy(effect.diffuse);
        mat->specular.copy(effect.specular);
        mat->shininess = effect.shininess;
        if (effect.transparency < 1.f) {
            mat->transparent = true;
            mat->opacity = effect.transparency;
        }
        return mat;
    }

    // Apply all transform children in order and return the combined Matrix4
    Matrix4 buildNodeTransform(const pugi::xml_node& node) {
        Matrix4 result;

        for (const auto& child : node.children()) {
            const std::string name = child.name();

            if (name == "matrix") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 16) {
                    Matrix4 m;
                    m.set(vals[0], vals[1], vals[2], vals[3],
                          vals[4], vals[5], vals[6], vals[7],
                          vals[8], vals[9], vals[10], vals[11],
                          vals[12], vals[13], vals[14], vals[15]);
                    result.multiply(m);
                }
            } else if (name == "translate") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 3) {
                    Matrix4 m;
                    m.makeTranslation(vals[0], vals[1], vals[2]);
                    result.multiply(m);
                }
            } else if (name == "rotate") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 4) {
                    Vector3 axis(vals[0], vals[1], vals[2]);
                    float angle = vals[3] * math::DEG2RAD;
                    Matrix4 m;
                    Quaternion q;
                    q.setFromAxisAngle(axis, angle);
                    m.makeRotationFromQuaternion(q);
                    result.multiply(m);
                }
            } else if (name == "scale") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 3) {
                    Matrix4 m;
                    m.makeScale(vals[0], vals[1], vals[2]);
                    result.multiply(m);
                }
            }
        }

        return result;
    }

}// namespace

struct ColladaLoader::Impl {

    std::shared_ptr<Group> load(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "[ColladaLoader] No such file: '" << std::filesystem::absolute(path).string() << "'!" << std::endl;
            return nullptr;
        }

        pugi::xml_document doc;
        const auto result = doc.load_file(path.string().c_str());
        if (!result) {
            std::cerr << "[ColladaLoader] Failed to parse XML: " << result.description() << std::endl;
            return nullptr;
        }

        const auto root = doc.child("COLLADA");
        if (!root) {
            std::cerr << "[ColladaLoader] Not a valid COLLADA file." << std::endl;
            return nullptr;
        }

        // Parse geometries
        std::unordered_map<std::string, GeometryData> geometries;
        for (const auto& geom : root.child("library_geometries").children("geometry")) {
            const std::string id = geom.attribute("id").value();
            geometries[id] = parseGeometry(geom);
        }

        // Parse controllers: map controller id -> geometry id (via <skin source="...">)
        std::unordered_map<std::string, std::string> controllerToGeometry;
        for (const auto& ctrl : root.child("library_controllers").children("controller")) {
            const std::string id = ctrl.attribute("id").value();
            const auto skin = ctrl.child("skin");
            if (skin) {
                controllerToGeometry[id] = stripHash(skin.attribute("source").value());
            }
        }

        // Parse effects
        std::unordered_map<std::string, EffectData> effects;
        for (const auto& eff : root.child("library_effects").children("effect")) {
            const std::string id = eff.attribute("id").value();
            effects[id] = parseEffect(eff);
        }

        // Parse materials: material id -> effect id
        std::unordered_map<std::string, std::string> materialToEffect;
        for (const auto& mat : root.child("library_materials").children("material")) {
            const std::string id = mat.attribute("id").value();
            const auto instEffect = mat.child("instance_effect");
            if (instEffect) {
                materialToEffect[id] = stripHash(instEffect.attribute("url").value());
            }
        }

        // Parse images: image id -> filename
        const auto baseDir = path.parent_path();
        std::unordered_map<std::string, std::filesystem::path> images;
        for (const auto& img : root.child("library_images").children("image")) {
            const std::string id = img.attribute("id").value();
            const std::string filename = img.child_value("init_from");
            if (!filename.empty()) {
                images[id] = baseDir / filename;
            }
        }

        TextureLoader texLoader;

        // Helper: resolve effect + texture -> Material
        auto loadMaterial = [&](const std::string& matId) -> std::shared_ptr<Material> {
            const std::string effectId = materialToEffect.contains(matId) ? materialToEffect.at(matId) : "";
            if (effectId.empty() || !effects.contains(effectId)) {
                return MeshPhongMaterial::create();
            }
            const EffectData& effect = effects.at(effectId);
            auto mat = buildMaterial(effect);
            if (!effect.diffuseTextureId.empty() && images.contains(effect.diffuseTextureId)) {
                auto tex = texLoader.load(images.at(effect.diffuseTextureId));
                if (tex) {
                    tex->wrapS = effect.wrapS;
                    tex->wrapT = effect.wrapT;
                    tex->needsUpdate();
                    mat->map = tex;
                }
            }
            return mat;
        };

        // Build scene
        auto sceneGroup = Group::create();

        const auto visualScenes = root.child("library_visual_scenes");
        const auto& scene = root.child("scene");
        std::string sceneId;
        if (scene) {
            const auto instScene = scene.child("instance_visual_scene");
            if (instScene) sceneId = stripHash(instScene.attribute("url").value());
        }

        pugi::xml_node visualScene;
        for (const auto& vs : visualScenes.children("visual_scene")) {
            if (sceneId.empty() || std::string(vs.attribute("id").value()) == sceneId) {
                visualScene = vs;
                break;
            }
        }

        if (!visualScene) {
            std::cerr << "[ColladaLoader] No visual scene found." << std::endl;
            return sceneGroup;
        }

        // Helper: parse a node recursively
        std::function<void(const pugi::xml_node&, Group&)> parseNode;
        parseNode = [&](const pugi::xml_node& nodeXml, Group& parent) {
            auto group = Group::create();
            group->name = nodeXml.attribute("name").value();

            Matrix4 localMatrix = buildNodeTransform(nodeXml);
            Vector3 pos, scale;
            Quaternion quat;
            localMatrix.decompose(pos, quat, scale);
            group->position.copy(pos);
            group->quaternion.copy(quat);
            group->scale.copy(scale);

            // instance_geometry
            for (const auto& instGeom : nodeXml.children("instance_geometry")) {
                const std::string geomId = stripHash(instGeom.attribute("url").value());
                auto geomIt = geometries.find(geomId);
                if (geomIt == geometries.end()) continue;
                const GeometryData& geodata = geomIt->second;

                // Build material symbol -> material id map from bind_material
                std::unordered_map<std::string, std::string> symbolToMaterialId;
                const auto bindMat = instGeom.child("bind_material");
                if (bindMat) {
                    for (const auto& instMat : bindMat.child("technique_common").children("instance_material")) {
                        const std::string symbol = instMat.attribute("symbol").value();
                        const std::string target = stripHash(instMat.attribute("target").value());
                        symbolToMaterialId[symbol] = target;
                    }
                }

                // Create a mesh per primitive
                for (const auto& prim : geodata.primitives) {
                    auto geom = buildGeometry(geodata, prim.material);
                    if (!geom) continue;

                    const std::string matId = symbolToMaterialId.contains(prim.material) ? symbolToMaterialId.at(prim.material) : prim.material;
                    auto mesh = Mesh::create(geom, loadMaterial(matId));
                    mesh->name = geodata.name;
                    group->add(mesh);
                }
            }

            // instance_controller (skinned meshes — instantiate the referenced geometry without skinning)
            for (const auto& instCtrl : nodeXml.children("instance_controller")) {
                const std::string ctrlId = stripHash(instCtrl.attribute("url").value());
                const auto ctrlGeomIt = controllerToGeometry.find(ctrlId);
                if (ctrlGeomIt == controllerToGeometry.end()) continue;

                const std::string geomId = ctrlGeomIt->second;
                const auto geomIt = geometries.find(geomId);
                if (geomIt == geometries.end()) continue;
                const GeometryData& geodata = geomIt->second;

                std::unordered_map<std::string, std::string> symbolToMaterialId;
                if (const auto bindMat = instCtrl.child("bind_material")) {
                    for (const auto& instMat : bindMat.child("technique_common").children("instance_material")) {
                        symbolToMaterialId[instMat.attribute("symbol").value()] = stripHash(instMat.attribute("target").value());
                    }
                }

                for (const auto& prim : geodata.primitives) {
                    auto geom = buildGeometry(geodata, prim.material);
                    if (!geom) continue;

                    const std::string matId = symbolToMaterialId.contains(prim.material) ? symbolToMaterialId.at(prim.material) : prim.material;
                    auto mesh = Mesh::create(geom, loadMaterial(matId));
                    mesh->name = geodata.name;
                    group->add(mesh);
                }
            }

            // Recurse into child nodes
            for (const auto& child : nodeXml.children("node")) {
                parseNode(child, *group);
            }

            parent.add(group);
        };


        for (const auto& node : visualScene.children("node")) {
            parseNode(node, *sceneGroup);
        }

        return sceneGroup;
    }
};

ColladaLoader::ColladaLoader()
    : pimpl_(std::make_unique<Impl>()) {}

std::shared_ptr<Group> ColladaLoader::load(const std::filesystem::path& path) {
    return pimpl_->load(path);
}

ColladaLoader::~ColladaLoader() = default;