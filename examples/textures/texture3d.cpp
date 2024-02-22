
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/math/ImprovedNoise.hpp"
#include "threepp/textures/DataTexture3D.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;


std::string vertexSource();
std::string fragmentSource();

namespace {

    auto createMaterial(Texture* texture) {

        auto m = RawShaderMaterial::create();
        m->vertexShader = vertexSource();
        m->fragmentShader = fragmentSource();
        m->side = Side::Back;
        m->transparent = true;

        m->uniforms = {
                {"base", Uniform(Color(0x798aa0))},
                {"map", Uniform(texture)},
                {"cameraPos", Uniform(Vector3())},
                {"threshold", Uniform(0.25f)},
                {"opacity", Uniform(0.25f)},
                {"range", Uniform(0.1f)},
                {"steps", Uniform(100)},
                {"frame", Uniform(1)}};

        return m;
    }

    auto createTextureData(unsigned int size) {

        int i = 0;
        float scale = 0.05f;
        Vector3 vector;
        math::ImprovedNoise perlin;
        std::vector<unsigned char> data(size * size * size);
        for (unsigned z = 0; z < size; z++) {
            for (unsigned y = 0; y < size; y++) {
                for (unsigned x = 0; x < size; x++) {

                    auto d = 1.f - vector.set(x, y, z).subScalar(size / 2).divideScalar(size).length();
                    data[i] = (128 + 128 * perlin.noise(x * scale / 1.5f, y * scale, z * scale / 1.5f)) * d * d;
                    ++i;
                }
            }
        }

        return data;
    }

}// namespace

int main() {

    Canvas canvas("DataTexture3D", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;
    renderer.setClearColor(Color::blue);

    Scene scene;
    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 100);
    camera.position.z = 1.5f;

    OrbitControls controls{camera, canvas};

    unsigned int size = 128;
    auto data = createTextureData(size);
    auto texture = DataTexture3D::create(data, size, size, size);
    texture->format = Format::Red;
    texture->minFilter = Filter::Linear;
    texture->magFilter = Filter::Linear;
    texture->unpackAlignment = 1;

    auto material = createMaterial(texture.get());
    auto geometry = BoxGeometry::create(1, 1, 1);

    auto mesh = Mesh::create(geometry, material);
    scene.add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });


    Clock clock;
    canvas.animate([&]() {
        float t = clock.getElapsedTime();

        material->uniforms.at("cameraPos").value<Vector3>().copy(camera.position);
        material->uniforms.at("frame").value<int>()++;

        int step = std::floor(50 * std::sin(math::TWO_PI * 0.1f * t) + 50);
        material->uniforms.at("steps").value<int>() = std::max(1, step);

        renderer.render(scene, camera);
    });
}

std::string vertexSource() {

    return R"(
        #version 330 core

        in vec3 position;

        uniform mat4 modelMatrix;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform vec3 cameraPos;

        out vec3 vOrigin;
        out vec3 vDirection;

        void main() {
            vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );

            vOrigin = vec3( inverse( modelMatrix ) * vec4( cameraPos, 1.0 ) ).xyz;
            vDirection = position - vOrigin;

            gl_Position = projectionMatrix * mvPosition;
        })";
}

std::string fragmentSource() {

    return R"(
        #version 330 core

        precision highp float;
        precision highp sampler3D;

        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;

        in vec3 vOrigin;
        in vec3 vDirection;

        out vec4 color;

        uniform vec3 base;
        uniform sampler3D map;

        uniform float threshold;
        uniform float range;
        uniform float opacity;
        uniform int steps;
        uniform int frame;

        uint wang_hash(uint seed)
        {
                seed = (seed ^ 61u) ^ (seed >> 16u);
                seed *= 9u;
                seed = seed ^ (seed >> 4u);
                seed *= 0x27d4eb2du;
                seed = seed ^ (seed >> 15u);
                return seed;
        }

        float randomFloat(inout uint seed)
        {
                return float(wang_hash(seed)) / 4294967296.;
        }

        vec2 hitBox( vec3 orig, vec3 dir ) {
            const vec3 box_min = vec3( - 0.5 );
            const vec3 box_max = vec3( 0.5 );
            vec3 inv_dir = 1.0 / dir;
            vec3 tmin_tmp = ( box_min - orig ) * inv_dir;
            vec3 tmax_tmp = ( box_max - orig ) * inv_dir;
            vec3 tmin = min( tmin_tmp, tmax_tmp );
            vec3 tmax = max( tmin_tmp, tmax_tmp );
            float t0 = max( tmin.x, max( tmin.y, tmin.z ) );
            float t1 = min( tmax.x, min( tmax.y, tmax.z ) );
            return vec2( t0, t1 );
        }

        float sample1( vec3 p ) {
            return texture( map, p ).r;
        }

        float shading( vec3 coord ) {
            float step = 0.01;
            return sample1( coord + vec3( - step ) ) - sample1( coord + vec3( step ) );
        }

        void main(){
            vec3 rayDir = normalize( vDirection );
            vec2 bounds = hitBox( vOrigin, rayDir );

            if ( bounds.x > bounds.y ) discard;

            bounds.x = max( bounds.x, 0.0 );

            vec3 p = vOrigin + bounds.x * rayDir;
            vec3 inc = 1.0 / abs( rayDir );
            float delta = min( inc.x, min( inc.y, inc.z ) );
            delta /= steps;

            // Jitter

            // Nice little seed from
            // https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
            uint seed = uint( gl_FragCoord.x ) * uint( 1973 ) + uint( gl_FragCoord.y ) * uint( 9277 ) + uint( frame ) * uint( 26699 );
            vec3 size = vec3( textureSize( map, 0 ) );
            float randNum = randomFloat( seed ) * 2.0 - 1.0;
            p += rayDir * randNum * ( 1.0 / size );

            //

            vec4 ac = vec4( base, 0.0 );

            for ( float t = bounds.x; t < bounds.y; t += delta ) {

                float d = sample1( p + 0.5 );

                d = smoothstep( threshold - range, threshold + range, d ) * opacity;

                float col = shading( p + 0.5 ) * 3.0 + ( ( p.x + p.y ) * 0.25 ) + 0.2;

                ac.rgb += ( 1.0 - ac.a ) * d * col;

                ac.a += ( 1.0 - ac.a ) * d;

                if ( ac.a >= 0.95 ) break;

                p += rayDir * delta;

            }

            color = ac;

            if ( color.a == 0.0 ) discard;

        })";
}
