
#include "GLUniforms.hpp"

#include "UniformUtils.hpp"

#include <regex>

using namespace threepp;
using namespace threepp::gl;

namespace {

    //    void setValueV1f(int addr, float value) {
    //
    //        glUniform1f(addr, value);
    //    }
    //
    //    template <class ArrayLike>
    //    void setValuev2f(int addr, ArrayLike value) {
    //
    //        glUniform2f(addr, value);
    //    }

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
        ActiveUniformInfo activeInfo;
        std::function<void(const UniformValue &, GLTextures *)> setValueFun;
        std::vector<float> cache;

        void setValueV1f(const UniformValue &value) const {

            float f = std::get<float>(value);
            std::cout << "value=" << f << std::endl;
            glUniform1f(addr, f);
        }

        std::function<void(const UniformValue &, GLTextures *)> getSingularSetter(int type) {

            switch (type) {

                case 0x1406:
                    return [&](const UniformValue &value, GLTextures *) { setValueV1f(value); };

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

    void addUniform(Container *container, std::shared_ptr<UniformObject> uniformObject) {

        container->seq.emplace_back(uniformObject);
        container->map[uniformObject->id] = uniformObject;
    }

    void parseUniform(ActiveUniformInfo &activeInfo, int addr, Container *container) {

        static std::regex rex(R"(([\w\d_]+)(\])?(\[|\.)?)");

        auto name = activeInfo.name;

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

        map[name]->setValue(value, textures);
    }
}

void GLUniforms::upload(std::vector<std::shared_ptr<UniformObject>> &seq, std::unordered_map<std::string, Uniform> &values, GLTextures *textures) {

    for (auto &u : seq) {

        Uniform &v = values[u->id];

        if (!v.needsUpdate || *v.needsUpdate) {

            // note: always updating when .needsUpdate is undefined
            u->setValue(v.value(), textures);
        }
    }
}

std::vector<std::shared_ptr<UniformObject>> GLUniforms::seqWithValue(std::vector<std::shared_ptr<UniformObject>> &seq, std::unordered_map<std::string, Uniform> &values) {

    std::vector<std::shared_ptr<UniformObject>> r;

    for (auto &u : seq) {

        if (values.count(u->id)) r.emplace_back(u);
    }

    return r;
}
