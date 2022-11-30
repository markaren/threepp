
#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/loaders/TextureLoader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

using namespace threepp;

struct AssimpLoader::Impl {

    std::shared_ptr<Group> load(const std::filesystem::path &path) {

        auto aiScene = importer_.ReadFile(path.string().c_str(), aiProcessPreset_TargetRealtime_Quality);

        if (!aiScene) {
            throw std::runtime_error(importer_.GetErrorString());
        }

        auto group = Group::create();

        if (aiScene->HasMeshes()) {
            auto aiMeshes = aiScene->mMeshes;

            for (unsigned i = 0; i < aiScene->mNumMeshes; ++i) {

                auto aiMesh = aiMeshes[i];
                auto geometry = BufferGeometry::create();

                std::vector<float> vertices;
                std::vector<float> normals;
                std::vector<float> colors;
                std::vector<float> uvs;

                if (aiMesh->HasFaces()) {
                    auto numFaces = aiMesh->mNumFaces;
                    for (unsigned j = 0; j < numFaces; ++j) {
                        auto face = aiMesh->mFaces[j];
                        auto numIndices = face.mNumIndices;
                        if (numIndices == 3) {
                            for (unsigned n = 0; n < numIndices; ++n) {
                                auto vertexIndex = face.mIndices[n];
                                if (aiMesh->HasNormals()) {
                                    normals.emplace_back(aiMesh->mNormals[vertexIndex].x);
                                    normals.emplace_back(aiMesh->mNormals[vertexIndex].y);
                                    normals.emplace_back(aiMesh->mNormals[vertexIndex].z);
                                }
                                if (aiMesh->HasTextureCoords(0)) {
                                    auto texCoord = aiMesh->mTextureCoords[0][vertexIndex];
                                    uvs.emplace_back(texCoord.x);
                                    uvs.emplace_back(texCoord.y);
                                }
                                if (aiMesh->mColors[0]) {
                                    colors.emplace_back(aiMesh->mColors[0][vertexIndex].r);
                                    colors.emplace_back(aiMesh->mColors[0][vertexIndex].g);
                                    colors.emplace_back(aiMesh->mColors[0][vertexIndex].b);
                                }

                                vertices.emplace_back(aiMesh->mVertices[vertexIndex].x);
                                vertices.emplace_back(aiMesh->mVertices[vertexIndex].y);
                                vertices.emplace_back(aiMesh->mVertices[vertexIndex].z);
                            }
                        }
                    }

                }

                auto material = MeshBasicMaterial::create();

                geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
                if (!normals.empty()) {
                    geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
                }
                if (!colors.empty()) {
                    material->vertexColors = true;
                    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
                }
                if (!uvs.empty()) {
                    geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
                }


                if (aiScene->HasMaterials()) {
                    aiString p;
                    auto mi = aiMesh->mMaterialIndex;
                    auto mat = aiScene->mMaterials[mi];
                    if (aiGetMaterialTextureCount(mat, aiTextureType_DIFFUSE) > 0) {
                        if (aiGetMaterialTexture(mat, aiTextureType_DIFFUSE, 0, &p) == aiReturn_SUCCESS) {
                            auto texPath = path.parent_path() / p.C_Str();
                            auto tex = TextureLoader().loadTexture(texPath);
//                            tex->encoding = sRGBEncoding;
                            material->map = tex;
                        }
                    } else {
                        C_STRUCT aiColor4D diffuse;
                        if(AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse)) {
                            material->color.setRGB(diffuse.r, diffuse.g, diffuse.b);
                        }
                    }
                    if (aiGetMaterialTextureCount(mat, aiTextureType_SPECULAR) > 0) {
                        if (aiGetMaterialTexture(mat, aiTextureType_SPECULAR, 0, &p) == aiReturn_SUCCESS) {
                            auto texPath = path.parent_path() / p.C_Str();
                            auto tex = TextureLoader().loadTexture(texPath);
//                            tex->encoding = sRGBEncoding;
                            material->specularMap = tex;
                        }
                    } else {
//                        C_STRUCT aiColor4D specular;
//                        if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_SPECULAR, &specular)) {
//                            material->specular.setRGB(specular.r, specular.g, specular.b);
//                        }
                    }

                }

                auto mesh = Mesh::create(geometry, material);
                group->add(mesh);
            }
        }

        return group;
    }

    ~Impl() = default;

private:
    Assimp::Importer importer_;
};

AssimpLoader::AssimpLoader()
    : pimpl_(std::make_unique<Impl>()) {}

std::shared_ptr<Group> AssimpLoader::load(const std::filesystem::path &path) {
    return pimpl_->load(path);
}

AssimpLoader::~AssimpLoader() = default;
