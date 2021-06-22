//

#ifndef THREEPP_GLRENDERLIST_HPP
#define THREEPP_GLRENDERLIST_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp::gl {

    struct RenderItem {

        unsigned int id;
        std::shared_ptr<Object3D> object;
        std::shared_ptr<BufferGeometry> geometry;
        std::shared_ptr<Material> material;


    };

    int painterSortStable( a, b ) {

        if ( a.groupOrder != b.groupOrder ) {

            return a.groupOrder - b.groupOrder;

        } else if ( a.renderOrder != b.renderOrder ) {

            return a.renderOrder - b.renderOrder;

        } else if ( a.program != b.program ) {

            return a.program.id - b.program.id;

        } else if ( a.material.id != b.material.id ) {

            return a.material.id - b.material.id;

        } else if ( a.z !== b.z ) {

            return a.z - b.z;

        } else {

            return a.id - b.id;

        }

    }

    int reversePainterSortStable( a, b ) {

        if ( a.groupOrder != b.groupOrder ) {

            return a.groupOrder - b.groupOrder;

        } else if ( a.renderOrder != b.renderOrder ) {

            return a.renderOrder - b.renderOrder;

        } else if ( a.z !== b.z ) {

            return b.z - a.z;

        } else {

            return a.id - b.id;

        }

    }

    class GLRenderLists {

    };

}

#endif//THREEPP_GLRENDERLIST_HPP
