
#include "GLUniforms.hpp"

#include "UniformUtils.hpp"

#include <regex>

using namespace threepp;
using namespace threepp::gl;

namespace {

    void ensureCapacity(std::vector<float> &v, unsigned int size) {

        while (v.size() < size) {
            v.emplace_back();
        }
    }

    // helper type for the visitor
    template<class... Ts>
    struct overloaded : Ts... { using Ts::operator()...; };
    // explicit deduction guide (not needed as of C++20)
    template<class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    struct ActiveUniformInfo {

        int size{};
        unsigned int type{};
        std::string name;

        ActiveUniformInfo(unsigned int program, unsigned int index) {

            int length;
            char nameBuffer[256];
            glGetActiveUniform(program, index, 256, &length, &size, &type, nameBuffer);
            name.assign(nameBuffer, length);
        }
    };

    struct SingleUniform : UniformObject {

        explicit SingleUniform(std::string id, ActiveUniformInfo activeInfo, int addr)
            : UniformObject(std::move(id)),
              activeInfo(std::move(activeInfo)),
              addr(addr),
              setValueFun(getSingularSetter(this->activeInfo.type)) {
        }

        void setValue(const UniformValue &value, GLTextures *textures) override {

            setValueFun(value, textures);
        }


    private:
        int addr;
        std::vector<float> cache;
        ActiveUniformInfo activeInfo;
        std::function<void(const UniformValue &, GLTextures *)> setValueFun;

        void setValueV1f(const UniformValue &value) {

            float f = std::get<float>(value);

            ensureCapacity(cache, 1);
            if (cache[0] == f) return;

            glUniform1f(addr, f);
            cache[0] = f;
        }

        template<class ArrayLike>
        void setValueV2fHelper(ArrayLike &value) {

            float x = value[0];
            float y = value[1];

            ensureCapacity(cache, 2);
            if (cache[0] != x || cache[1] != y) {

                glUniform2f(addr, x, y);

                cache[0] = x;
                cache[1] = y;
            }
        }

        void setValueV2f(const UniformValue &value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cout << "setValueV2f: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Vector2 arg) { setValueV2fHelper(arg); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueV3fHelper(ArrayLike &value) {

            float x = value[0];
            float y = value[1];
            float z = value[2];

            ensureCapacity(cache, 3);
            if (cache[0] != x || cache[1] != y || cache[2] != z) {

                glUniform3f(addr, x, y, z);

                cache[0] = x;
                cache[1] = y;
                cache[2] = z;
            }
        }

        void setValueV3f(const UniformValue &value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cout << "setValueV3f: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Vector3 arg) { setValueV3fHelper(arg); },
                               [&](Color arg) { setValueV3fHelper(arg); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueV4fHelper(ArrayLike &value) {

            float x = value[0];
            float y = value[1];
            float z = value[2];
            float w = value[3];

            ensureCapacity(cache, 4);
            if (cache[0] != x || cache[1] != y && cache[2] != z || cache[3] != w) {

                glUniform4f(addr, x, y, z, w);

                cache[0] = x;
                cache[1] = y;
                cache[2] = z;
                cache[3] = w;
            }
        }

        void setValueV4f(const UniformValue &value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cout << "setValueV4f: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Vector4 arg) { setValueV4fHelper(arg); },
                               [&](Quaternion arg) { setValueV4fHelper(arg); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueM3Helper(ArrayLike &value) {

            if (arraysEqual(cache, value)) return;

            glUniformMatrix3fv(addr, 1, false, value.data());

            ensureCapacity(cache, 9);
            copyArray(cache, value);
        }

        void setValueM3(const UniformValue &value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cout << "setValueM3: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Matrix3 arg) { setValueM3Helper(arg.elements); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueM4Helper(ArrayLike &value) {

            if (arraysEqual(cache, value)) return;

            glUniformMatrix4fv(addr, 1, false, value.data());

            ensureCapacity(cache, 16);
            copyArray(cache, value);
        }

        void setValueM4(const UniformValue &value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cout << "setValueM4: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Matrix4 arg) { setValueM4Helper(arg.elements); },
                       },
                       value);
        }

        std::function<void(const UniformValue &, GLTextures *)> getSingularSetter(int type) {

            switch (type) {

                case 0x1406:
                    return [&](const UniformValue &value, GLTextures *) { setValueV1f(value); };

                case 0x8b50:
                    return [&](const UniformValue &value, GLTextures *) { setValueV2f(value); };

                case 0x8b51:
                    return [&](const UniformValue &value, GLTextures *) { setValueV3f(value); };

                case 0x8b52:
                    return [&](const UniformValue &value, GLTextures *) { setValueV4f(value); };

                case 0x8b5a:
                    return [&](const UniformValue &value, GLTextures *) { setValueM3(value); };

                case 0x8b5c:
                    return [&](const UniformValue &value, GLTextures *) { setValueM4(value); };

                default:
                    return [&](const UniformValue &value, GLTextures *) {
                        std::cout << "TODO type:" << type << std::endl;
                    };
            }
        }
    };

    struct PureArrayUniform : UniformObject {

        explicit PureArrayUniform(std::string id, ActiveUniformInfo activeInfo, int addr)
            : UniformObject(std::move(id)),
              activeInfo(std::move(activeInfo)),
              addr(addr) {}

        void setValue(const UniformValue &value, GLTextures *textures) override {
            std::cout << "PureArrayUniform '" << id << "'" << std::endl;
        }

    private:
        int addr;
        ActiveUniformInfo activeInfo;
    };

    struct StructuredUniform : UniformObject, Container {

        explicit StructuredUniform(std::string id) : UniformObject(std::move(id)) {}

        void setValue(const UniformValue &value, GLTextures *textures) override {
            std::cout << "StructuredUniform '" << id << "'" << std::endl;
        }
    };

    void addUniform(Container *container, const std::shared_ptr<UniformObject> &uniformObject) {

        container->seq.emplace_back(uniformObject);
        container->map[uniformObject->id] = uniformObject;
    }

    void parseUniform(ActiveUniformInfo &activeInfo, int addr, Container *container) {

        static std::regex rex(R"(([\w\d_]+)(\])?(\[|\.)?)");

        const auto name = activeInfo.name;

        std::sregex_iterator rex_it(name.cbegin(), name.cend(), rex);
        std::sregex_iterator rex_end;

        while (rex_it != rex_end) {
            std::smatch match = *rex_it;

            std::string id = match[1];
            bool isIndex = match[2] == "]";
            std::string subscript = match[3];

            if (isIndex) id = std::to_string(std::atoi(id.c_str()) | 0);

            if (!match[3].matched || subscript == "[" && match[3].second == name.end()) {

                // bare name or "pure" bottom-level array "[0]" suffix
                if (!match[3].matched) {
                    addUniform(container, std::make_shared<SingleUniform>(id, activeInfo, addr));
                } else {
                    addUniform(container, std::make_shared<PureArrayUniform>(id, activeInfo, addr));
                }
                break;

            } else {

                if (!container->map.count(id)) {
                    addUniform(container, std::make_shared<StructuredUniform>(id));
                }

                container = dynamic_cast<Container *>(container->map[id].get());
            }

            rex_it++;
        }
    }


}// namespace


GLUniforms::GLUniforms(unsigned int program) {

    int n;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &n);

    for (int i = 0; i < n; i++) {

        ActiveUniformInfo info(program, i);
        GLint addr = glGetUniformLocation(program, info.name.c_str());

        parseUniform(info, addr, dynamic_cast<Container *>(this));
    }
}

void GLUniforms::setValue(const std::string &name, const UniformValue &value, GLTextures *textures) {

    if (map.count(name)) {

        map.at(name)->setValue(value, textures);
    }
}

void GLUniforms::upload(std::vector<std::shared_ptr<UniformObject>> &seq, std::shared_ptr<UniformMap> &values, GLTextures *textures) {

    for (auto &u : seq) {

        Uniform &v = values->at(u->id);

        if (!v.needsUpdate || (v.needsUpdate && v.needsUpdate.value())) {

            // note: always updating when .needsUpdate is undefined
            u->setValue(v.value(), textures);
        }
    }
}

std::vector<std::shared_ptr<UniformObject>> GLUniforms::seqWithValue(std::vector<std::shared_ptr<UniformObject>> &seq, std::shared_ptr<UniformMap> &values) {

    std::vector<std::shared_ptr<UniformObject>> r;

    for (const auto &u : seq) {

        if (values->count(u->id)) r.emplace_back(u);
    }

    return r;
}
