// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferGeometry.js

#ifndef THREEPP_BUFFERGEOMETRY_HPP
#define THREEPP_BUFFERGEOMETRY_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/Group.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"

#include <any>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp {

    class BufferGeometry {

    public:
        const unsigned int id = _id++;

        std::string uuid = generateUUID();

        std::string name;
        std::string type = "BufferGeometry";

        std::vector<Group> groups;

        std::vector<int> &getIndex() {

            return this->index;
        }

        BufferGeometry &setIndex(const std::vector<int> &index) {

            this->index = index;

            return *this;
        }

        template<class T>
        BufferAttribute<T> &getAttribute(const std::string &name) {

            if (!hasAttribute(name)) throw std::runtime_error("No attribute named: " + name);

            return std::any_cast<BufferAttribute<T> &>(attributes_[name]);
        }

        template<class T>
        void setAttribute(const std::string &name, BufferAttribute<T> attribute) {

            attributes_[name] = std::move(attribute);
        }

        bool hasAttribute(const std::string &name) {

            return attributes_.count(name) != 0;
        }

        void addGroup(unsigned int start, unsigned int count, unsigned int materialIndex = 0) {

            groups.emplace_back(Group{start, count, materialIndex});
        }

        void clearGroups() {

            groups.clear();
        }

        BufferGeometry &applyMatrix4(const Matrix4 &matrix) {

            if (this->attributes_.count("position")) {

                auto position = std::any_cast<BufferAttribute<float>>(this->attributes_["position"]);

                position.applyMatrix4(matrix);

                position.needsUpdate();
            }


            if (this->attributes_.count("normal")) {

                auto normal = std::any_cast<BufferAttribute<float>>(this->attributes_["normal"]);

                auto normalMatrix = Matrix3().getNormalMatrix(matrix);

                normal.applyNormalMatrix(normalMatrix);

                normal.needsUpdate();
            }


            if (this->attributes_.count("tangent")) {

                auto tangent = std::any_cast<BufferAttribute<float>>(this->attributes_["tangent"]);

                tangent.transformDirection(matrix);

                tangent.needsUpdate();
            }

            //            if ( this.boundingBox !== null ) {
            //
            //                this.computeBoundingBox();
            //
            //            }
            //
            //            if ( this.boundingSphere !== null ) {
            //
            //                this.computeBoundingSphere();
            //
            //            }

            return *this;
        }

    private:
        std::vector<int> index;
        std::unordered_map<std::string, std::any> attributes_;

        inline static unsigned int _id = 0;
    };


}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRY_HPP
