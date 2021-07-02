
#include "GLUniforms.hpp"

#include "UniformUtils.hpp"

using namespace threepp;
using namespace threepp::gl;


namespace {

    struct ActiveUniformInfo {

        int size;
        unsigned int type;
        std::string name;

        ActiveUniformInfo(unsigned int program, unsigned int index) {

            int length;
            char nameBuffer[256];
            glGetActiveUniform(program, index, 256, &length, &size, &type, nameBuffer);
            name.assign(name, length);
        }
    };

    void addUniform(Container *container, std::shared_ptr<UniformObject> &uniformObject) {

        container->seq.emplace_back(uniformObject);
        container->map[uniformObject->id] = uniformObject;
    }

    void parseUniform(ActiveUniformInfo &activeInfo, int addr, Container *container) {
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

    struct SingleUniform : UniformObject {

        explicit SingleUniform(std::string id) : UniformObject(std::move(id)) {}

        void setValue(const UniformValue &value, GLTextures *textures) override {
        }
    };

    struct PureArrayUniform : UniformObject {

        explicit PureArrayUniform(std::string id) : UniformObject(std::move(id)) {}
    };

    struct StructuredUniform : UniformObject, Container {

        explicit StructuredUniform(std::string id) : UniformObject(std::move(id)) {}

        void setValue(const UniformValue &value, GLTextures *textures) override {
        }
    };
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
