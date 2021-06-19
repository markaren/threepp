// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderChunk.js

#ifndef THREEPP_SHADERCHUNK_HPP
#define THREEPP_SHADERCHUNK_HPP

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace threepp::shaders {

    class ShaderChunk {

    public:

        ShaderChunk(const ShaderChunk&) = delete;
        void operator=(const ShaderChunk&) = delete;

        static ShaderChunk &instance() {
            static ShaderChunk instance;
            return instance;
        }

        std::string get(const std::string &key) {
            if (data_.count(key) == 0) {
                std::string line;
                std::string text;
                std::ifstream file("ShaderChunk/" + key + ".glsl");
                if (file) {
                    while (!file.eof()) {
                        std::getline(file, line);
                        text.append(line + "\n");
                    }
                } else {
                    throw std::runtime_error("No such file: " + key);
                }

                data_[key] = text;
            }
            return data_[key];
        }

    private:
        std::unordered_map<std::string, std::string> data_;

        ShaderChunk() = default;
        ~ShaderChunk() = default;
    };

}// namespace threepp::shaders

#endif//THREEPP_SHADERCHUNK_HPP
