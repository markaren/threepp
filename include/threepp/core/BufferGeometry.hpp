// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferGeometry.js

#ifndef THREEPP_BUFFERGEOMETRY_HPP
#define THREEPP_BUFFERGEOMETRY_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/core/Group.hpp"

#include <utility>
#include <vector>
#include <unordered_map>

namespace threepp {

    class BufferGeometry {

    public:

        const unsigned int id = _id++;

        std::string uuid = generateUUID();

        std::string name;
        std::string type = "BufferGeometry";

        std::vector<Group> groups;

        std::any &getAttribute(const std::string &name) {
            return attributes_[name];
        }

        void setAttribute(const std::string &name, std::any attribute) {
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

        BufferGeometry &applyMatrix4( const Matrix4 &matrix ) {



            if ( this->attributes_.count("position") ) {

                //auto position = std::any_cast<Vector3>(this->attributes_["position"]);

//                position.applyMatrix4( matrix );
//
//                position.needsUpdate = true;

            }


            if ( this->attributes_.count("normal") ) {

//                const normal = this.attributes.normal;
//
//                const normalMatrix = new Matrix3().getNormalMatrix( matrix );
//
//                normal.applyNormalMatrix( normalMatrix );
//
//                normal.needsUpdate = true;

            }



            if ( this->attributes_.count("tangent") ) {

//                const tangent = this.attributes.tangent;
//
//                tangent.transformDirection( matrix );
//
//                tangent.needsUpdate = true;

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

        std::unordered_map<std::string, std::any> attributes_;

        static unsigned int _id;

    };

    unsigned int BufferGeometry::_id = 0;

}

#endif//THREEPP_BUFFERGEOMETRY_HPP
