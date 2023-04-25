#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/loaders/CubeTextureLoader.hpp"
#include "threepp/threepp.hpp"

#include <string>

namespace {

std::string vertexSource() {

        return R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;

            out vec3 texCoords;

            uniform mat4 projectionMatrix;
            uniform mat4 modelViewMatrix;

            void main()
            {
                vec4 pos = projectionMatrix * modelViewMatrix * vec4(aPos, 1.0f);
                // Having z equal w will always result in a depth of 1.0f
                gl_Position = vec4(pos.x, pos.y, pos.w, pos.w);
                // We want to flip the z axis due to the different coordinate systems (left hand vs right hand)
                texCoords = vec3(aPos.x, aPos.y, -aPos.z);
            }   )";
    }
    std::string fragmentSource() {

        return R"(
            #version 330 core
            out vec4 FragColor;

            in vec3 texCoords;

            uniform samplerCube skybox;

            void main()
            {    
                FragColor = texture(skybox, texCoords);
            })";
    }
}// namespace

using namespace threepp;

int main() {

    Canvas canvas("Cubmap demo", {{"antialiasing", 4}});

    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 1, 10);
    camera->position.z = 2;

    std::vector<unsigned int> skyboxIndices
        { // Right
          1, 2, 6, 
          6, 5, 1,
          // Left
          0, 4, 7, 
          7, 3, 0,
          // Top
          4, 5, 6, 
          6, 7, 4,
          // Bottom
          0, 3, 2,
          2, 1, 0, 
          // Back
          0, 1, 5, 
          5, 3, 0,
          // Front
          3, 7, 6, 
          6, 2, 3 };
        std::vector<float> skyboxVertices
        { -1.0f, -1.0f, 1.0f,
           1.0f, -1.0f, 1.0f,
           1.0f, -1.0f, -1.0f,
           -1.0f, -1.0f, -1.0f,
           -1.0f, 1.0f, 1.0f,
           1.0f,  1.0f, 1.0f,
           1.0f,  1.0f, -1.0f,
           -1.0f, 1.0f, -1.0f };


        std::list<std::filesystem::path> urls = 
        {
            "data/textures/cubemap/colloseum/right.png",
            "data/textures/cubemap/colloseum/left.png",
            "data/textures/cubemap/colloseum/top.png",
            "data/textures/cubemap/colloseum/bottom.png",
            "data/textures/cubemap/colloseum/front.png",
            "data/textures/cubemap/colloseum/back.png"
        };
        CubeTextureLoader loader;
        auto cubeTexture = loader.load(urls);

        // Scene    
        auto skyboxGeometry = BufferGeometry::create();
        skyboxGeometry->setIndex(skyboxIndices);
        skyboxGeometry->setAttribute("position", FloatBufferAttribute::create(skyboxVertices, 3));
        
        auto material = RawShaderMaterial::create();  
        (*material->uniforms)["skybox"] = Uniform(); 
        material->vertexShader = vertexSource();
        material->fragmentShader = fragmentSource();
        material->side = FrontSide;
        material->transparent = false;
        material->depthFunc = LessEqualDepth;
        material->depthTest = true;
        material->depthWrite = true;
        material->fog = false;
        
        material->uniforms->at("skybox").setValue(cubeTexture.get()); 
        Matrix4 identify; 
        auto mesh = Mesh::create(skyboxGeometry, material);
        scene->add(mesh);


    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) { 
        renderer.render(scene, camera);
    });
}
