
#include "threepp/loaders/OBJLoader.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/MTLLoader.hpp"
#include "threepp/materials/materials.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

using namespace threepp;

namespace {

    size_t parseVertexOrNormalIndex(const std::string& value, size_t len) {
        int index = utils::parseInt(value);
        return (index > 0 ? (index - 1) : index + len / 3) * 3;
    }

    size_t parseUvIndex(const std::string& value, size_t len) {
        int index = utils::parseInt(value);
        return (index > 0 ? (index - 1) : index + len / 2) * 2;
    }

    struct OBJGeometry {
        std::string type;
        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> colors;
        std::vector<float> uvs;
    };

    struct OBJMaterial {
        std::optional<size_t> index;
        std::string name;
        std::string mtlib;
        bool smooth = false;
        int groupStart = 0;
        int groupEnd = -1;
        int groupCount = -1;
        bool inherited = false;
    };

    struct OBJObject {
        std::string name;
        bool fromDeclaration;

        bool smooth = false;

        OBJGeometry geometry;
        std::vector<std::shared_ptr<OBJMaterial>> materials;

        OBJObject(std::string name, bool fromDeclaration, bool smooth)
            : name(std::move(name)),
              fromDeclaration(fromDeclaration),
              smooth(smooth) {}

        std::shared_ptr<OBJMaterial> currentMaterial() {
            if (materials.empty()) return nullptr;
            return materials.back();
        }

        std::shared_ptr<OBJMaterial> startMaterial(const std::string& mname, const std::vector<std::string>& libraries) {

            auto previous = finalize(false);

            if (previous) {
                if (previous->inherited || previous->groupCount <= 0) {
                    materials.erase(materials.begin() + static_cast<int>(previous->index.value()));
                }
            }

            auto material = std::make_shared<OBJMaterial>();
            material->index = materials.size();
            material->name = mname;
            material->mtlib = !libraries.empty() ? libraries.back() : "",
            material->smooth = previous ? previous->smooth : smooth,
            material->groupStart = previous ? previous->groupEnd : 0,
            material->groupCount = -1,
            material->groupEnd = -1,
            material->inherited = false;

            materials.emplace_back(material);

            return materials.back();
        }

        std::shared_ptr<OBJMaterial> finalize(bool end) {
            auto lastMultiMaterial = currentMaterial();
            if (lastMultiMaterial && lastMultiMaterial->groupEnd == -1) {

                lastMultiMaterial->groupEnd = static_cast<int>(geometry.vertices.size()) / 3;
                lastMultiMaterial->groupCount = lastMultiMaterial->groupEnd - lastMultiMaterial->groupStart;
                lastMultiMaterial->inherited = false;
            }

            if (end && !materials.empty()) {
                for (int mi = static_cast<int>(materials.size()) - 1; mi >= 0; --mi) {
                    if (materials.at(mi)->groupCount <= 0) {
                        materials.erase(materials.begin() + mi);
                    }
                }
            }

            if (end && materials.empty()) {
                auto m = std::make_shared<OBJMaterial>();
                m->name = "";
                m->smooth = smooth;
                materials.emplace_back(m);
            }

            return lastMultiMaterial;
        }
    };


    class ParserState {

    public:
        std::shared_ptr<OBJObject> object;
        std::vector<std::shared_ptr<OBJObject>> objects;

        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> colors;
        std::vector<float> uvs;

        std::vector<std::string> materialLibraries;

        ParserState() {
            startObject("", false);
        }

        void startObject(const std::string& name, bool fromDeclaration = false) {

            if (object) {
                if (!object->fromDeclaration) {
                    object->name = name;
                    object->fromDeclaration = fromDeclaration;
                    return;
                }
            }

            std::shared_ptr<OBJMaterial> previousMaterial;
            if (object) {
                previousMaterial = object->currentMaterial();
                object->finalize(true);
            }

            object = std::make_shared<OBJObject>(name, fromDeclaration, true);
            if (previousMaterial && !previousMaterial->name.empty()) {

                auto declared = std::make_shared<OBJMaterial>(*previousMaterial);
                declared->index = 0;
                declared->inherited = true;
                object->materials.emplace_back(declared);
            }

            objects.emplace_back(object);
        }

        void finalize() const {
            object->finalize(true);
        }

        void addVertex(size_t a, size_t b, size_t c) {

            auto& src = vertices;
            auto& dst = object->geometry.vertices;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2],
                                   src[b + 0], src[b + 1], src[b + 2],
                                   src[c + 0], src[c + 1], src[c + 2]});
        }

        void addVertexPointOrLine(size_t a) {

            auto& src = vertices;
            auto& dst = object->geometry.vertices;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2]});
        }

        void addNormal(size_t a, size_t b, size_t c) {

            auto& src = normals;
            auto& dst = object->geometry.normals;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2],
                                   src[b + 0], src[b + 1], src[b + 2],
                                   src[c + 0], src[c + 1], src[c + 2]});
        }

        void addColor(size_t a, size_t b, size_t c) {

            auto& src = colors;
            auto& dst = object->geometry.colors;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2],
                                   src[b + 0], src[b + 1], src[b + 2],
                                   src[c + 0], src[c + 1], src[c + 2]});
        }

        void addUv(size_t a, size_t b, size_t c) {

            auto& src = uvs;
            auto& dst = object->geometry.uvs;

            dst.insert(dst.end(), {src[a + 0], src[a + 1],
                                   src[b + 0], src[b + 1],
                                   src[c + 0], src[c + 1]});
        }

        void addUvLine(size_t a) {

            auto& src = uvs;
            auto& dst = object->geometry.uvs;

            dst.insert(dst.end(), {src[a + 0], src[a + 1]});
        }

        void addFace(const std::string& a, const std::string& b, const std::string& c,
                     const std::string& ua, const std::string& ub, const std::string& uc,
                     const std::string& na, const std::string& nb, const std::string& nc) {

            auto vLen = vertices.size();

            auto ia = parseVertexOrNormalIndex(a, vLen);
            auto ib = parseVertexOrNormalIndex(b, vLen);
            auto ic = parseVertexOrNormalIndex(c, vLen);

            addVertex(ia, ib, ic);

            if (!ua.empty()) {

                auto uvlen = uvs.size();
                ia = parseUvIndex(ua, uvlen);
                ib = parseUvIndex(ub, uvlen);
                ic = parseUvIndex(uc, uvlen);

                addUv(ia, ib, ic);
            }

            if (!na.empty()) {

                auto nLen = normals.size();
                ia = parseVertexOrNormalIndex(na, nLen);

                ib = (na == nb) ? ia : parseVertexOrNormalIndex(nb, nLen);
                ic = (na == nc) ? ia : parseVertexOrNormalIndex(nc, nLen);

                addNormal(ia, ib, ic);
            }

            if (!colors.empty()) {

                addColor(ia, ib, ic);
            }
        }

        void addPointGeometry(const std::vector<std::string>& verts) {

            object->geometry.type = "Points";

            auto vLen = verts.size();

            for (const auto& v : verts) {
                addVertexPointOrLine(parseVertexOrNormalIndex(v, vLen));
            }
        }
    };

}// namespace

struct OBJLoader::Impl {

    OBJLoader& scope;
    std::shared_ptr<MaterialCreator> materials;
    std::unordered_map<std::string, std::weak_ptr<Group>> cache_;

    explicit Impl(OBJLoader& scope): scope(scope) {}

    std::shared_ptr<Group> load(const std::filesystem::path& path, bool tryLoadMtl) {

        if (scope.useCache && cache_.contains(path.string())) {

            auto cached = cache_.at(path.string());
            if (!cached.expired()) {
                return cached.lock()->clone<Group>();
            } else {
                cache_.erase(path.string());
            }
        }

        if (!std::filesystem::exists(path)) {
            std::cerr << "[OBJLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
            return nullptr;
        }

        std::ifstream in(path);

        if (tryLoadMtl) {
            std::filesystem::path mtlFile{path.parent_path() / (path.stem().string() + ".mtl")};
            if (std::filesystem::exists(mtlFile)) {
                materials = MTLLoader().load(absolute(mtlFile));
                materials->preload();
            }
        }

        ParserState state;

        std::string line;
        while (std::getline(in, line)) {
            utils::trimInplace(line);

            auto lineLength = line.size();

            if (lineLength == 0) continue;

            auto lineFirstChar = line.front();

            if (lineFirstChar == '#') continue;

            if (lineFirstChar == 'v') {

                auto data = utils::split(line, ' ');

                if (data[0] == "v") {
                    state.vertices.insert(state.vertices.end(), {utils::parseFloat(data[1]),
                                                                 utils::parseFloat(data[2]),
                                                                 utils::parseFloat(data[3])});

                    if (data.size() == 8) {
                        state.colors.insert(state.colors.end(), {utils::parseFloat(data[4]),
                                                                 utils::parseFloat(data[5]),
                                                                 utils::parseFloat(data[6])});
                    }
                } else if (data[0] == "vn") {
                    state.normals.insert(state.normals.end(), {utils::parseFloat(data[1]),
                                                               utils::parseFloat(data[2]),
                                                               utils::parseFloat(data[3])});
                } else if (data[0] == "vt") {
                    state.uvs.insert(state.uvs.end(), {utils::parseFloat(data[1]),
                                                       utils::parseFloat(data[2])});
                }
            } else if (lineFirstChar == 'f') {

                std::string lineData{line.begin() + 1, line.end()};
                utils::trimInplace(lineData);
                auto vertexData = utils::split(lineData, ' ');
                std::vector<std::vector<std::string>> faceVertices;

                for (const auto& vertex : vertexData) {

                    if (!vertexData.empty()) {
                        auto vertexParts = utils::split(vertex, '/');
                        faceVertices.insert(faceVertices.end(), vertexParts);
                    }
                }

                auto& v1 = faceVertices[0];

                for (unsigned j = 1; j < faceVertices.size() - 1; ++j) {

                    auto& v2 = faceVertices[j];
                    auto& v3 = faceVertices[j + 1];

                    state.addFace(v1[0], v2[0], v3[0],
                                  v1[1], v2[1], v3[1],
                                  v1[2], v2[2], v3[2]);
                }


            } else if (lineFirstChar == 'l') {

                // TODO

            } else if (lineFirstChar == 'p') {

                auto lineData = utils::trim(line.substr(1));
                auto pointData = utils::split(lineData, ' ');

                state.addPointGeometry(pointData);

            } else if (lineFirstChar == 's') {

                auto result = utils::split(line, ' ');

                if (!result.empty()) {

                    auto value = utils::trim(result[1]);
                    utils::toLowerInplace(value);
                    state.object->smooth = (value != "0" && value != "off");

                } else {

                    state.object->smooth = true;
                }

                auto material = state.object->currentMaterial();
                if (material) {
                    material->smooth = state.object->smooth;
                }

            } else {
                static std::regex r("^[og]\\s+(.*)");
                std::smatch match;
                if (std::regex_match(line, match, r)) {
                    std::string name = utils::trim(match[1]);
                    state.startObject(name);
                    continue;
                }
                if (line.find("usemtl ") != std::string::npos) {
                    state.object->startMaterial(utils::trim(line.substr(7)), state.materialLibraries);
                    continue;
                }
                if (line.find("mtllib ") != std::string::npos) {
                    state.materialLibraries.emplace_back(utils::trim(line.substr(7)));
                    continue;
                }


                if (line == "\\0") continue;

                std::cerr << "[OBJLoader] Unexpected line: " << line << ":" << line.size() << std::endl;
            }
        }

        state.finalize();

        auto container = Group::create();

        for (const auto& object : state.objects) {

            auto& geometry = object->geometry;
            auto& materials = object->materials;
            bool isLine = geometry.type == "Line";
            bool isPoints = geometry.type == "Points";
            bool hasVertexColors = false;

            if (geometry.vertices.empty()) continue;

            auto bufferGeometry = BufferGeometry::create();

            bufferGeometry->setAttribute("position", FloatBufferAttribute::create(geometry.vertices, 3));

            if (!geometry.normals.empty()) {

                bufferGeometry->setAttribute("normal", FloatBufferAttribute::create(geometry.normals, 3));

            } else {

                //TODO
            }

            if (!geometry.colors.empty()) {

                bufferGeometry->setAttribute("color", FloatBufferAttribute::create(geometry.colors, 3));
            }

            if (!geometry.uvs.empty()) {

                bufferGeometry->setAttribute("uv", FloatBufferAttribute::create(geometry.uvs, 2));
            }

            std::vector<std::shared_ptr<Material>> createdMaterials;

            for (auto& sourceMaterial : materials) {

                std::shared_ptr<Material> material;

                if (this->materials) {
                    material = this->materials->create(sourceMaterial->name);

                    if (isLine && material && !material->is<LineBasicMaterial>()) {

                        // TODO

                    } else if (isPoints && material && !material->is<PointsMaterial>()) {

                        // TODO
                    }
                }

                if (!material) {

                    if (isLine) {
                        material = LineBasicMaterial::create();
                    } else if (isPoints) {
                        auto pointsMaterial = PointsMaterial::create();
                        pointsMaterial->size = 1;
                        pointsMaterial->sizeAttenuation = false;
                        material = std::move(pointsMaterial);
                    } else {
                        material = MeshPhongMaterial::create();
                    }

                    material->name = sourceMaterial->name;
                }

                material->vertexColors = hasVertexColors;
                if (material->is<MaterialWithFlatShading>()) {
                    std::dynamic_pointer_cast<MaterialWithFlatShading>(material)->flatShading = !sourceMaterial->smooth;
                }

                createdMaterials.emplace_back(material);
            }

            std::shared_ptr<Object3D> mesh;

            if (!createdMaterials.empty()) {

                for (unsigned mi = 0; mi < materials.size(); ++mi) {

                    auto& sourceMaterial = materials.at(mi);
                    bufferGeometry->addGroup(sourceMaterial->groupStart, sourceMaterial->groupCount, mi);
                }

                if (isLine) {
                    mesh = LineSegments::create(bufferGeometry, createdMaterials.front());
                } else if (isPoints) {
                    mesh = Points::create(bufferGeometry, createdMaterials.front());
                } else {
                    mesh = Mesh::create(bufferGeometry, createdMaterials);
                }

            } else {

                if (isLine) {
                    mesh = LineSegments::create(bufferGeometry, createdMaterials.front());
                } else if (isPoints) {
                    mesh = Points::create(bufferGeometry, createdMaterials.front());
                } else {
                    mesh = Mesh::create(bufferGeometry, createdMaterials.front());
                }
            }

            mesh->name = object->name;

            container->add(mesh);
        }

        if (scope.useCache) cache_[path.string()] = container;

        return container;
    }
};

OBJLoader::OBJLoader()
    : pimpl_(std::make_unique<Impl>(*this)) {}

std::shared_ptr<Group> OBJLoader::load(const std::filesystem::path& path, bool tryLoadMtl) {

    return pimpl_->load(path, tryLoadMtl);
}

void OBJLoader::clearCache() {

    pimpl_->cache_.clear();
}

OBJLoader& OBJLoader::setMaterials(const std::shared_ptr<MaterialCreator>& materials) {

    pimpl_->materials = materials;

    return *this;
}

OBJLoader::~OBJLoader() = default;
