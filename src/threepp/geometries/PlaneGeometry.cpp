
#include "threepp/geometries/PlaneGeometry.hpp"

using namespace threepp;


PlaneGeometry::PlaneGeometry(float width, float height, int widthSegments, int heightSegments) {

    const auto width_half = width / 2;
    const auto height_half = height / 2;

    const auto gridX = std::floor( widthSegments );
    const auto gridY = std::floor( heightSegments );

    const auto gridX1 = gridX + 1;
    const auto gridY1 = gridY + 1;

    const auto segment_width = width / gridX;
    const auto segment_height = height / gridY;

    //

    std::vector<int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    for ( int iy = 0; iy < gridY1; iy ++ ) {

        const auto y = (float) (iy * segment_height - height_half);

        for ( int ix = 0; ix < gridX1; ix ++ ) {

            const auto x = (float) (ix * segment_width - width_half);

            vertices.insert(vertices.end(), {x, - y, 0} );

            normals.insert(normals.end(), { 0, 0, 1 } );

            uvs.emplace_back( (float) (ix / gridX) );
            uvs.emplace_back( (float) (1 - ( iy / gridY )) );

        }

    }

    for ( int iy = 0; iy < gridY; iy ++ ) {

        for ( int ix = 0; ix < gridX; ix ++ ) {

            const auto a = (int) (ix + gridX1 * iy);
            const auto b = (int) (ix + gridX1 * ( iy + 1 ));
            const auto c = (int) (( ix + 1 ) + gridX1 * ( iy + 1 ));
            const auto d = (int) (( ix + 1 ) + gridX1 * iy);

            indices.insert(indices.end(), {a, b, d} );
            indices.insert(indices.end(), {b, c, d} );

        }

    }

    this->setIndex( indices );
    this->setAttribute( "position", FloatBufferAttribute::create( vertices, 3 ) );
    this->setAttribute( "normal", FloatBufferAttribute::create( normals, 3 ) );
    this->setAttribute( "uv", FloatBufferAttribute::create( uvs, 2 ) );

}
