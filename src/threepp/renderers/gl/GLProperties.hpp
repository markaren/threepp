//

#ifndef THREEPP_GLPROPERTIES_HPP
#define THREEPP_GLPROPERTIES_HPP

#include <string>
#include <unordered_map>

namespace threepp::gl {

    template <class T>
    class GLProperties {

    public:

        GLProperties() = default;

        std::unordered_map<T, GLProperties> &get(const T &object) {

            return properties_[object];
        }

        void remove(const T &object) {

            properties_.erase(object);
        }

        void update(const T &object, const std::string &key, const std::any& value) {

            properties_.at(object)[key] = value;
        }

        void dispose() {

            properties_.clear();
        }

    private:
        std::unordered_map<T, std::unordered_map<std::string, std::any>> properties_;

    };

}

#endif//THREEPP_GLPROPERTIES_HPP
