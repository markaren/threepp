
#include "threepp/geometries/CylinderGeometry.hpp"

using namespace threepp;

CylinderGeometry::CylinderGeometry(float radiusTop, float radiusBottom, float height, int radialSegments, int heightSegments, bool openEnded, float thetaStart, float thetaLength) {

    std::vector<int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    int index = 0;
    const auto halfHeight = height / 2;
    std::vector<std::vector<int>> indexArray;
    int groupStart = 0;

    auto generateTorso = [&]{

      Vector3 normal;
      Vector3 vertex;

      int groupCount = 0;

      // this will be used to calculate the normal
      const auto slope = ( radiusBottom - radiusTop ) / height;

      // generate vertices, normals and uvs

      for ( int y = 0; y <= heightSegments; y ++ ) {

          std::vector<int> indexRow;

          const auto v = (float) y / heightSegments;

          // calculate the radius of the current row

          const auto radius = v * ( radiusBottom - radiusTop ) + radiusTop;

          for ( int x = 0; x <= radialSegments; x ++ ) {

              const auto u =  (float) x / radialSegments;

              const auto theta = u * thetaLength + thetaStart;

              const auto sinTheta = std::sin( theta );
              const auto  cosTheta = std::cos( theta );

              // vertex

              vertex.x = radius * sinTheta;
              vertex.y = - v * height + halfHeight;
              vertex.z = radius * cosTheta;
              vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z} );

              // normal

              normal.set( sinTheta, slope, cosTheta ).normalize();
              normals.insert(normals.end(), {normal.x, normal.y, normal.z} );

              // uv

              uvs.insert(uvs.end(), {u, 1 - v} );

              // save index of vertex in respective row

              indexRow.emplace_back( index ++ );

          }

          // now save vertices of the row in our index array

          indexArray.emplace_back( indexRow );

      }

      // generate indices

      for ( int x = 0; x < radialSegments; x ++ ) {

          for ( int y = 0; y < heightSegments; y ++ ) {

              // we use the index array to access the correct indices

              const auto a = indexArray[ y ][ x ];
              const auto b = indexArray[ y + 1 ][ x ];
              const auto c = indexArray[ y + 1 ][ x + 1 ];
              const auto d = indexArray[ y ][ x + 1 ];

              // faces

              indices.insert(indices.end(), {a, b, d} );
              indices.insert(indices.end(), {b, c, d} );

              // update group counter

              groupCount += 6;

          }

      }

      // add a group to the geometry. this will ensure multi material support

      addGroup( groupStart, groupCount, 0 );

      // calculate new start value for groups

      groupStart += groupCount;
    };

    generateTorso();


    auto generateCap = [](bool top) {

    };

    if ( !openEnded ) {

        if ( radiusTop > 0 ) generateCap( true );
        if ( radiusBottom > 0 ) generateCap( false );

    }

    // build geometry

    this->setIndex( indices );
    this->setAttribute( "position", BufferAttribute<float>( vertices, 3 ) );
    this->setAttribute( "normal", BufferAttribute<float>( normals, 3 ) );
    this->setAttribute( "uv", BufferAttribute<float>( uvs, 2 ) );

}

