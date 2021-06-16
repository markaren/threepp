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
        static ShaderChunk *getInstance() {
            if (!instance_) {
                instance_ = new ShaderChunk();
            }
            return instance_;
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
        static ShaderChunk *instance_;

        std::unordered_map<std::string, std::string> data_;

        ShaderChunk() = default;
    };

    ShaderChunk *ShaderChunk::instance_ = nullptr;

}// namespace threepp::shaders

#endif//THREEPP_SHADERCHUNK_HPP
