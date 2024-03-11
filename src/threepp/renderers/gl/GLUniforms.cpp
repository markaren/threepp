
#include "threepp/renderers/gl/GLUniforms.hpp"

#include "threepp/renderers/gl/UniformUtils.hpp"
#include "threepp/utils/StringUtils.hpp"

#ifndef EMSCRIPTEN
#include <glad/glad.h>
#else
#include <GL/glew.h>
#endif

#include <iostream>
#include <regex>

using namespace threepp;
using namespace threepp::gl;

namespace {

    // helper type for the visitor
    template<class... Ts>
    struct overloaded: Ts... {
        using Ts::operator()...;
    };
    // explicit deduction guide (not needed as of C++20)
    template<class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    struct ActiveUniformInfo {

        int size;
        unsigned int type;
        std::string name;

        ActiveUniformInfo(unsigned int program, unsigned int index) {

            int length;
            char nameBuffer[256];
            glGetActiveUniform(program, index, 256, &length, &size, &type, nameBuffer);
            name.assign(nameBuffer, length);
        }
    };

    struct SingleUniform: UniformObject {

        explicit SingleUniform(std::string id, ActiveUniformInfo activeInfo, int addr)
            : UniformObject(std::move(id)),
              activeInfo(std::move(activeInfo)),
              addr(addr),
              setValueFun(getSingularSetter()) {
        }

        void setValue(const UniformValue& value, GLTextures* textures) override {
            setValueFun(value, textures);
        }

    private:
        int addr;
        std::vector<float> cache;
        ActiveUniformInfo activeInfo;
        std::function<void(const UniformValue&, GLTextures*)> setValueFun;

        std::function<void(const UniformValue&, GLTextures*)> getSingularSetter() {

            switch (activeInfo.type) {

                case 0x1406:// FLOAT
                    return [&](const UniformValue& value, GLTextures*) { setValueV1f(value); };

                case 0x8b50:// _VEC2
                    return [&](const UniformValue& value, GLTextures*) { setValueV2f(value); };

                case 0x8b51:// _VEC3
                    return [&](const UniformValue& value, GLTextures*) { setValueV3f(value); };

                case 0x8b52:// _VEC4
                    return [&](const UniformValue& value, GLTextures*) { setValueV4f(value); };

                case 0x8b5b:// _MAT3
                    return [&](const UniformValue& value, GLTextures*) { setValueM3(value); };

                case 0x8b5c:// _MAT4
                    return [&](const UniformValue& value, GLTextures*) { setValueM4(value); };

                case 0x8b5e:// SAMPLER_2D
                case 0x8d66:// SAMPLER_EXTERNAL_OES
                case 0x8dca:// INT_SAMPLER_2D
                case 0x8dd2:// UNSIGNED_INT_SAMPLER_2D
                case 0x8b62:// SAMPLER_2D_SHADOW
                    return [&](const UniformValue& value, GLTextures* textures) { setValueT1(value, textures); };

                case 0x8b5f:// SAMPLER_3D
                case 0x8dcb:// INT_SAMPLER_3D
                case 0x8dd3:// UNSIGNED_INT_SAMPLER_3D
                    return [&](const UniformValue& value, GLTextures* textures) { setValueT3D1(value, textures); };

                case 0x8b60:// SAMPLER_CUBE
                case 0x8dcc:// INT_SAMPLER_CUBE
                case 0x8dd4:// UNSIGNED_INT_SAMPLER_CUBE
                case 0x8dc5:// SAMPLER_CUBE_SHADOW
                    return [&](const UniformValue& value, GLTextures* textures) { setValueT6(value, textures); };

                case 0x1404:// INT, BOOL
                case 0x8b56:
                    return [&](const UniformValue& value, GLTextures*) { setValueV1i(value); };

                default:
                    return [&](const UniformValue& value, GLTextures*) {
                        std::cout << "SingleUniform TODO: "
                                  << "name=" << activeInfo.name << ",type=" << activeInfo.type << std::endl;
                    };
            }
        }

        // Single texture (2D / Cube)

        void setValueT1(const UniformValue& value, GLTextures* textures) const {
            const auto unit = textures->allocateTextureUnit();
            glUniform1i(addr, unit);
            auto tex = std::get<Texture*>(value);
            textures->setTexture2D(*tex, unit);
        }

        void setValueT3D1(const UniformValue& value, GLTextures* textures) const {
            const auto unit = textures->allocateTextureUnit();
            glUniform1i(addr, unit);
            auto tex = std::get<Texture*>(value);
            textures->setTexture3D(*tex, unit);
        }

        void setValueT6(const UniformValue& value, GLTextures* textures) const {
            const auto unit = textures->allocateTextureUnit();
            glUniform1i(addr, unit);
            auto tex = std::get<Texture*>(value);
            textures->setTextureCube(*tex, unit);
        }

        void setValueV1i(const UniformValue& value) const {

            if (std::holds_alternative<bool>(value)) {
                bool b = std::get<bool>(value);
                glUniform1i(addr, b);
            } else if (std::holds_alternative<int>(value)) {
                int i = std::get<int>(value);
                glUniform1i(addr, i);
            } else {
                throw std::runtime_error("Illegal variant index: " + std::to_string(value.index()));
            }
        }

        void setValueV1f(const UniformValue& value) {

            float f;
            if (std::holds_alternative<bool>(value)) {
                f = std::get<bool>(value);
            } else if (std::holds_alternative<float>(value)) {
                f = std::get<float>(value);
            } else {
                throw std::runtime_error("Illegal variant index: " + std::to_string(value.index()));
            }

            ensureCapacity(cache, 1);
            if (cache[0] == f) return;

            glUniform1f(addr, f);
            cache[0] = f;
        }

        template<class ArrayLike>
        void setValueV2fHelper(ArrayLike& value) {

            float x = value[0];
            float y = value[1];

            ensureCapacity(cache, 2);
            if (cache[0] != x || cache[1] != y) {

                glUniform2f(addr, x, y);

                cache[0] = x;
                cache[1] = y;
            }
        }

        void setValueV2f(const UniformValue& value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cerr << "setValueV2f: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Vector2 arg) { setValueV2fHelper(arg); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueV3fHelper(ArrayLike& value) {

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

        void setValueV3f(const UniformValue& value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cerr << "setValueV3f: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Vector3 arg) { setValueV3fHelper(arg); },
                               [&](Vector3* arg) { setValueV3fHelper(*arg); },
                               [&](Color arg) { setValueV3fHelper(arg); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueV4fHelper(ArrayLike& value) {

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

        void setValueV4f(const UniformValue& value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cerr << "setValueV4f: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Vector4 arg) { setValueV4fHelper(arg); },
                               [&](Quaternion arg) { setValueV4fHelper(arg); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueM3Helper(ArrayLike& value) {

            if (arraysEqual(cache, value)) return;

            glUniformMatrix3fv(addr, 1, false, value.data());

            ensureCapacity(cache, 9);
            copyArray(cache, value);
        }

        void setValueM3(const UniformValue& value) {

            std::visit(overloaded{
                               [&](auto) { std::cerr << "setValueM3: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Matrix3 arg) { setValueM3Helper(arg.elements); },
                       },
                       value);
        }

        template<class ArrayLike>
        void setValueM4Helper(ArrayLike& value) {

            if (arraysEqual(cache, value)) return;

            glUniformMatrix4fv(addr, 1, false, value.data());

            ensureCapacity(cache, 16);
            copyArray(cache, value);
        }

        void setValueM4(const UniformValue& value) {

            std::visit(overloaded{
                               [&](auto arg) { std::cerr << "setValueM4: unsupported variant at index: " << value.index() << std::endl; },
                               [&](Matrix4 arg) { setValueM4Helper(arg.elements); },
                               [&](Matrix4* arg) { setValueM4Helper(arg->elements); }},
                       value);
        }
    };

    struct PureArrayUniform: UniformObject {

        explicit PureArrayUniform(std::string id, ActiveUniformInfo activeInfo, int addr)
            : UniformObject(std::move(id)),
              activeInfo(std::move(activeInfo)),
              addr(addr),
              setValueFun(getPureArraySetter()) {}

        void setValue(const UniformValue& value, GLTextures* textures) override {
            setValueFun(value, textures);
        }

        [[nodiscard]] std::function<void(const UniformValue&, GLTextures*)> getPureArraySetter() const {

            switch (activeInfo.type) {

                case 0x1406:// FLOAT
                    return [&](const UniformValue& value, GLTextures*) {
                        auto& data = std::get<std::vector<float>>(value);
                        glUniform1fv(addr, activeInfo.size, data.data());
                    };
                case 0x8b50:// VEC2
                    return [&](const UniformValue& value, GLTextures*) {
                        auto& data = std::get<std::vector<Vector2>>(value);
                        glUniform2fv(addr, activeInfo.size, flatten(data, activeInfo.size, 2).data());
                    };
                case 0x8b51:// VEC3
                    return [&](const UniformValue& value, GLTextures*) {
                        auto& data = std::get<std::vector<Vector3>>(value);
                        glUniform3fv(addr, activeInfo.size, flatten(data, activeInfo.size, 3).data());
                    };
                case 0x8b52:// VEC4
                    return [&](const UniformValue& value, GLTextures*) {
                        auto& data = std::get<std::vector<float>>(value);
                        glUniform4fv(addr, activeInfo.size, data.data());
                    };

                case 0x8b5b:// MAT3
                    return [&](const UniformValue& value, GLTextures*) {
                        auto& data = std::get<std::vector<Matrix3>>(value);
                        glUniformMatrix3fv(addr, activeInfo.size, false, flatten(data, activeInfo.size, 9).data());
                    };
                case 0x8b5c:// MAT4
                    return [&](const UniformValue& value, GLTextures*) {
                        std::visit(overloaded{
                                           [&](auto arg) { std::cerr << "setValueM4: unsupported variant at index: " << value.index() << std::endl; },
                                           [&](std::vector<float> arg) { glUniformMatrix4fv(addr, activeInfo.size, false, arg.data()); },
                                           [&](std::vector<Matrix4> arg) { glUniformMatrix4fv(addr, activeInfo.size, false, flatten(arg, activeInfo.size, 16).data()); },
                                           [&](std::vector<Matrix4*> arg) { glUniformMatrix4fv(addr, activeInfo.size, false, flattenP(arg, activeInfo.size, 16).data()); }},
                                   value);
                    };
                case 0x8b5e:// SAMPLER_2D
                case 0x8d66:// SAMPLER_EXTERNAL_OES
                case 0x8dca:// INT_SAMPLER_2D
                case 0x8dd2:// UNSIGNED_INT_SAMPLER_2D
                case 0x8b62:// SAMPLER_2D_SHADOW
                    return [&](const UniformValue& value, GLTextures* textures) {
                        auto& data = std::get<std::vector<Texture*>>(value);
                        const auto n = data.size();
                        auto units = allocTexUnits(*textures, n);

                        glUniform1iv(addr, static_cast<int>(n), units.data());

                        for (unsigned i = 0; i != n; ++i) {
                            auto value = data[i];
                            if (!value) continue;
                            textures->setTexture2D(*data[i], units[i]);
                        }
                    };
                default:
                    return [&](const UniformValue& value, GLTextures*) {
                        std::cout << "PureArrayUniform TODO: "
                                  << "name=" << activeInfo.name << ",type=" << activeInfo.type << std::endl;
                    };
            }
        }

    private:
        int addr;
        ActiveUniformInfo activeInfo;
        std::function<void(const UniformValue&, GLTextures*)> setValueFun;
    };

    struct StructuredUniform: UniformObject, Container {

        explicit StructuredUniform(std::string id, ActiveUniformInfo activeInfo)
            : UniformObject(std::move(id)),
              activeInfo(std::move(activeInfo)) {}

        void setValue(const UniformValue& value, GLTextures* textures) override {
            //            std::cout << "StructuredUniform '" << id << "', index=" << value.index() << std::endl;

            std::visit(
                    overloaded{
                            [&](auto) { std::cout << "StructuredUniform '" << activeInfo.name << "': unsupported variant at index: " << value.index() << std::endl; },
                            [&](std::unordered_map<std::string, NestedUniformValue> arg) {
                                for (auto& u : seq) {
                                    NestedUniformValue& v = arg.at(u->id);
                                    std::visit(overloaded{
                                                       [&](auto) { std::cout << "Warning: Unhandled NestedUniformValue!" << std::endl; },
                                                       [&](int arg) { u->setValue(arg, textures); },
                                                       [&](float arg) { u->setValue(arg, textures); },
                                                       [&](Vector2 arg) { u->setValue(arg, textures); },
                                                       [&](Vector3 arg) { u->setValue(arg, textures); },
                                                       [&](Color arg) { u->setValue(arg, textures); }},
                                               v);
                                }
                            },
                            [&](std::vector<std::unordered_map<std::string, NestedUniformValue>*> arg) {
                                for (auto& u : seq) {
                                    int index = utils::parseInt(u->id);
                                    auto value = arg[index];
                                    if (!value) continue;
                                    u->setValue(*arg[index], textures);
                                }
                            }},
                    value);
        }

    private:
        ActiveUniformInfo activeInfo;
    };

    void addUniform(Container* container, std::unique_ptr<UniformObject> uniformObject) {

        const auto id = uniformObject->id;
        container->seq.emplace_back(std::move(uniformObject));
        container->map[id] = container->seq.back().get();
    }

    void parseUniform(ActiveUniformInfo& activeInfo, int addr, Container* container) {

        static std::regex rex(R"(([\w\d_]+)(\])?(\[|\.)?)");

        const auto path = activeInfo.name;
        const auto pathLength = path.size();

        std::sregex_iterator rex_it(path.cbegin(), path.cend(), rex);
        std::sregex_iterator rex_end;

        while (rex_it != rex_end) {
            std::smatch match = *rex_it;

            const auto matchEnd = match.length();

            std::string id = match[1];
            bool isIndex = match[2] == "]";
            std::string subscript = match[3];

            if (isIndex) id = std::to_string(utils::parseInt(id) | 0);

            if (!match[3].matched || subscript == "[" && matchEnd + 2 == pathLength) {

                // bare name or "pure" bottom-level array "[0]" suffix
                if (!match[3].matched) {
                    addUniform(container, std::make_unique<SingleUniform>(id, activeInfo, addr));
                } else {
                    addUniform(container, std::make_unique<PureArrayUniform>(id, activeInfo, addr));
                }
                break;

            } else {

                if (!container->map.count(id)) {
                    addUniform(container, std::make_unique<StructuredUniform>(id, activeInfo));
                }

                container = dynamic_cast<Container*>(container->map.at(id));
            }

            ++rex_it;
        }
    }


}// namespace


GLUniforms::GLUniforms(unsigned int program) {

    int n{};
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &n);

    for (int i = 0; i < n; ++i) {

        ActiveUniformInfo info(program, i);
        GLint addr = glGetUniformLocation(program, info.name.c_str());

        parseUniform(info, addr, dynamic_cast<Container*>(this));
    }
}

void GLUniforms::setValue(const std::string& name, const UniformValue& value, GLTextures* textures) {

    if (map.count(name)) {

        map.at(name)->setValue(value, textures);
    }
}

void GLUniforms::upload(std::vector<UniformObject*>& seq, UniformMap& values, GLTextures* textures) {

    for (const auto& u : seq) {

        Uniform& v = values.at(u->id);

        if (!v.needsUpdate || (v.needsUpdate && v.needsUpdate.value())) {

            // note: always updating when .needsUpdate is undefined
            u->setValue(v.value(), textures);
        }
    }
}

std::vector<UniformObject*> GLUniforms::seqWithValue(const std::vector<std::unique_ptr<UniformObject>>& seq, UniformMap& values) {

    std::vector<UniformObject*> r;

    for (const auto& u : seq) {

        if (values.count(u->id)) r.emplace_back(u.get());
    }

    return r;
}
