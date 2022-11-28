
#include "threepp/loaders/OBJLoader.hpp"

#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/utils/StringUtils.hpp"
#include "threepp/utils/regex_util.hpp"

using namespace threepp;

namespace {

    template<class T>
    void splice(std::vector<T> list, int start, int deleteCount, const std::optional<std::vector<T>> &elements = std::nullopt) {
        if (deleteCount > 0) {
            list.erase(list.begin() + start, list.begin() + start + deleteCount);
        }

        if (elements) {
            for (const T &t : *elements) {
                list.emplace_back(t);
            }
        }
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
        std::optional<std::string> name;
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

        std::shared_ptr<OBJMaterial> startMaterial(const std::string &mname, const std::vector<std::string> &libraries) {

            auto previous = finalize(false);

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
            if (lastMultiMaterial && lastMultiMaterial->groupCount == 1) {
                lastMultiMaterial->groupEnd = static_cast<int>(geometry.vertices.size()) / 3;
                lastMultiMaterial->groupCount = lastMultiMaterial->groupEnd - lastMultiMaterial->groupStart;
                lastMultiMaterial->inherited = false;
            }

            if (end && !materials.empty()) {
                for (int mi = static_cast<int>(materials.size()) - 1; mi >= 0; --mi) {
                    if (materials.at(mi)->groupCount <= 0) {
                        splice(materials, mi, 1);
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

        void startObject(const std::string &name, bool fromDeclaration = false) {

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
            if (previousMaterial && previousMaterial->name) {

                auto declared = std::make_shared<OBJMaterial>(*previousMaterial);
                declared->index = 0;
                declared->inherited = true;
                object->materials.emplace_back(declared);
            }

            objects.emplace_back(object);
        }

        void finalize() {
            object->finalize(true);
        }

        size_t parseVertexIndex(const std::string &value, size_t len) {
            int index = std::stoi(value);
            return (index > 0 ? (index - 1) : index + len / 3) * 3;
        }

        size_t parseNormalIndex(const std::string &value, size_t len) {
            int index = std::stoi(value);
            return (index > 0 ? (index - 1) : index + len / 3) * 3;
        }

        size_t parseUvIndex(const std::string &value, size_t len) {
            int index = std::stoi(value);
            return (index > 0 ? (index - 1) : index + len / 2) * 2;
        }

        void addVertex(size_t a, size_t b, size_t c) {

            auto &src = vertices;
            auto &dst = object->geometry.vertices;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2],
                                   src[b + 0], src[b + 1], src[b + 2],
                                   src[c + 0], src[c + 1], src[c + 2]});
        }

        void addVertexPointOrLine(size_t a) {

            auto &src = vertices;
            auto &dst = object->geometry.vertices;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2]});
        }

        void addNormal(size_t a, size_t b, size_t c) {

            auto &src = normals;
            auto &dst = object->geometry.normals;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2],
                                   src[b + 0], src[b + 1], src[b + 2],
                                   src[c + 0], src[c + 1], src[c + 2]});
        }

        void addColor(size_t a, size_t b, size_t c) {

            auto &src = colors;
            auto &dst = object->geometry.colors;

            dst.insert(dst.end(), {src[a + 0], src[a + 1], src[a + 2],
                                   src[b + 0], src[b + 1], src[b + 2],
                                   src[c + 0], src[c + 1], src[c + 2]});
        }

        void addUv(size_t a, size_t b, size_t c) {

            auto &src = uvs;
            auto &dst = object->geometry.uvs;

            dst.insert(dst.end(), {src[a + 0], src[a + 1],
                                   src[b + 0], src[b + 1],
                                   src[c + 0], src[c + 1]});
        }

        void addUvLine(size_t a) {

            auto &src = uvs;
            auto &dst = object->geometry.uvs;

            dst.insert(dst.end(), {src[a + 0], src[a + 1]});
        }

        void addFace(const std::string &a, const std::string &b, const std::string &c,
                     const std::string &ua, const std::string &ub, const std::string &uc,
                     const std::string &na, const std::string &nb, const std::string &nc) {

            auto vLen = vertices.size();

            auto ia = parseVertexIndex(a, vLen);
            auto ib = parseVertexIndex(b, vLen);
            auto ic = parseVertexIndex(c, vLen);

            addVertex(ia, ib, ic);

            if (!ua.empty()) {

                auto uvlen = uvs.size();
                ia = parseUvIndex(ua, uvlen);
                ib = parseUvIndex(ua, uvlen);
                ic = parseUvIndex(ua, uvlen);

                addUv(ia, ib, ic);
            }

            if (!na.empty()) {

                auto nLen = normals.size();
                ia = parseNormalIndex(na, nLen);

                ib = (na == nb) ? ia : parseNormalIndex(nb, nLen);
                ic = (na == nc) ? ia : parseNormalIndex(nc, nLen);

                addNormal(ia, ib, ic);
            }

            if (!colors.empty()) {

                addColor(ia, ib, ic);
            }
        }

        void addPointGeometry(const std::vector<std::string> &verts) {

            object->geometry.type = "Points";

            auto vLen = verts.size();

            for (const auto &v : verts) {
                addVertexPointOrLine(parseVertexIndex(v, vLen));
            }
        }
    };

    std::shared_ptr<Group> parse(const std::string &text) {

        ParserState state;

        auto text2 = utils::replaceAll(text, "\r\n", "\n");

        auto lines = utils::split(text2, '\n');

        for (auto line : lines) {
            line = utils::trimStart(line);

            auto lineLength = line.size();

            if (lineLength == 0) continue;

            auto &lineFirstChar = line[0];

            if (lineFirstChar == '#') continue;

            if (lineFirstChar == 'v') {

                auto data = utils::regexSplit(line, std::regex("\\s+"));

                if (data[0] == "v") {
                    state.vertices.insert(state.vertices.end(), {std::stof(data[1]),
                                                                   std::stof(data[2]),
                                                                   std::stof(data[3])});

                    if (data.size() == 8) {
                        state.colors.insert(state.colors.end(), {std::stof(data[4]),
                                                                   std::stof(data[5]),
                                                                   std::stof(data[6])});
                    }
                } else if (data[0] == "vn") {
                    state.normals.insert(state.normals.end(), {std::stof(data[1]),
                                                                  std::stof(data[2]),
                                                                  std::stof(data[3])});
                } else if (data[0] == "vt") {
                    state.uvs.insert(state.uvs.end(), {std::stof(data[1]),
                                                         std::stof(data[2])});
                }
            } else if (lineFirstChar == 'f') {

                std::string lineData{line.begin() + 1, line.end()};
                lineData = utils::trim(lineData);
                auto vertexData = utils::regexSplit(lineData, std::regex("\\s+"));
                std::vector<std::vector<std::string>> faceVertices;

                for (const auto &vertex : vertexData) {

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

            } else if (lineFirstChar == 'p') {

                auto lineData = utils::trim(line.substr(1));
                auto pointData = utils::split(lineData, ' ');

                state.addPointGeometry(pointData);

            } else if (lineFirstChar == 's') {

                auto result = utils::split(line, ' ');

                if (!result.empty()) {

                    auto value = utils::trim(result[1]);
                    value = utils::toLower(value);
                    state.object->smooth = (value != "0" && value != "off");

                } else {

                    state.object->smooth = true;
                }

                auto material = state.object->currentMaterial();
                if (material) {
                    material->smooth = state.object->smooth;
                }

            } else {

                std::vector<std::string> result = findAll(line, std::regex("^[og]\\s*(.+)?"));
                if (!result.empty()) {
                    std::string name = utils::trim(result.front().substr(1));
                    state.startObject(name);
                    continue;
                }

                result = findAll(line, std::regex("^usemtl "));
                if (!result.empty()) {
                    state.object->startMaterial(utils::trim(line.substr(7)), state.materialLibraries);
                    continue;
                }
                result = findAll(line, std::regex("^mtllib "));
                if (!result.empty()) {
                    state.materialLibraries.emplace_back(utils::trim(line.substr(7)));
                    continue;
                }


                if (line == "\\0") continue;

                throw std::runtime_error("Unexpected line: " + line);
            }
        }

        state.finalize();

        auto container = Group::create();

        for (const auto &object : state.objects) {

            auto &geometry = object->geometry;
            auto &materials = object->materials;
            bool isLine = geometry.type == "Line";
            bool isPoints = geometry.type == "Points";
            bool hasVertexColors = false;

            if (geometry.vertices.empty()) continue;

            auto bufferGeometry = BufferGeometry::create();

            bufferGeometry->setAttribute("position", FloatBufferAttribute::create(geometry.vertices, 3));

            if (!geometry.normals.empty()) {

                bufferGeometry->setAttribute("normal", FloatBufferAttribute::create(geometry.normals, 3));
            } else {


            }

            if (!geometry.colors.empty()) {

                bufferGeometry->setAttribute("color", FloatBufferAttribute::create(geometry.colors, 3));
            }

            if (!geometry.uvs.empty()) {

                bufferGeometry->setAttribute("uv", FloatBufferAttribute::create(geometry.uvs, 2));
            }

            std::vector<std::shared_ptr<Material>> createdMaterials;

            for (auto &sourceMaterial : materials) {

                std::shared_ptr<Material> material;

                //TODO
            }

            std::shared_ptr<Object3D> mesh;

            if (!createdMaterials.empty()) {
                //TODO
            } else {

                if (isLine) {
                    mesh = LineSegments::create(bufferGeometry, createdMaterials.front());
                } else if (isPoints) {
                    mesh = Points::create(bufferGeometry, createdMaterials.front());
                } else {
                    mesh = Mesh::create(bufferGeometry, MeshBasicMaterial::create());//TODO
                }
            }

            mesh->name = object->name;

            container->add(mesh);
        }

        return container;
    }

}// namespace


std::shared_ptr<Group> OBJLoader::load(const std::filesystem::path &path) {

    std::ifstream file(path);
    std::stringstream strStream;
    strStream << file.rdbuf();

    return parse(strStream.str());
}
