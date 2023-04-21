
#ifndef THREEPP_ASSIMPLOADER_HPP
#define THREEPP_ASSIMPLOADER_HPP

#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <filesystem>

namespace threepp {

    class AssimpLoader {

    public:
        std::shared_ptr<Group> load(const std::filesystem::path& path) {

            auto aiScene = importer_.ReadFile(path.string().c_str(), aiProcessPreset_TargetRealtime_Quality);

            if (!aiScene) {
                throw std::runtime_error(importer_.GetErrorString());
            }

            auto group = Group::create();
            parseNodes(path, aiScene, aiScene->mRootNode, *group);

            return group;
        }

    private:
        TextureLoader texLoader_;
        Assimp::Importer importer_;

        void parseNodes(const std::filesystem::path& path, const aiScene* aiScene, aiNode* aiNode, Object3D& parent) {

            auto group = Group::create();
            group->name = aiNode->mName.C_Str();

            for (unsigned i = 0; i < aiNode->mNumMeshes; ++i) {
                auto geometry = BufferGeometry::create();

                auto aiMesh = aiScene->mMeshes[aiNode->mMeshes[i]];

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

                auto material = MeshPhongMaterial::create();

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
                            auto tex = texLoader_.load(texPath);
                            //                        tex->encoding = sRGBEncoding;
                            std::dynamic_pointer_cast<MaterialWithMap>(material)->map = tex;
                        }
                    } else {
                        C_STRUCT aiColor4D diffuse;
                        if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse)) {
                            std::dynamic_pointer_cast<MaterialWithColor>(material)->color.setRGB(diffuse.r, diffuse.g, diffuse.b);
                        }
                    }

                    auto m = material->as<MeshPhongMaterial>();

                    if (aiGetMaterialTextureCount(mat, aiTextureType_EMISSIVE) > 0) {
                        if (aiGetMaterialTexture(mat, aiTextureType_EMISSIVE, 0, &p) == aiReturn_SUCCESS) {
                            auto texPath = path.parent_path() / p.C_Str();
                            auto tex = texLoader_.load(texPath);
                            //                        tex->encoding = sRGBEncoding;
                            m->emissiveMap = tex;
                        }
                    } else {
                        C_STRUCT aiColor4D emissive;
                        if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_EMISSIVE, &emissive)) {
                            m->emissive.setRGB(emissive.r, emissive.g, emissive.b);
                        }
                    }

                    if (aiGetMaterialTextureCount(mat, aiTextureType_SPECULAR) > 0) {
                        if (aiGetMaterialTexture(mat, aiTextureType_SPECULAR, 0, &p) == aiReturn_SUCCESS) {
                            auto texPath = path.parent_path() / p.C_Str();
                            auto tex = texLoader_.load(texPath);
                            //                        tex->encoding = sRGBEncoding;
                            m->specularMap = tex;
                        }
                    } else {
                        C_STRUCT aiColor4D specular;
                        if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_SPECULAR, &specular)) {
                            m->specular.setRGB(specular.r, specular.g, specular.b);
                        }
                    }

                    float shininess;
                    if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_SHININESS, &shininess)) {
                        m->shininess = shininess;
                    }

                    float emmisiveIntensity;
                    if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_EMISSIVE_INTENSITY, &emmisiveIntensity)) {
                        m->emissiveIntensity = emmisiveIntensity;
                    }

//                    C_STRUCT aiColor4D ambient;
//                    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_AMBIENT, &ambient)) {
//                        std::dynamic_pointer_cast<MaterialWithColor>(material)->color.add(Color().setRGB(ambient.r, ambient.g, ambient.b));
//                    }
//
//                    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &ambient)) {
//                        std::dynamic_pointer_cast<MaterialWithColor>(material)->color.setRGB(ambient.r, ambient.g, ambient.b);
//                    }

                    float opacity;
                    if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_OPACITY, &opacity)) {
                        material->transparent = true;
                        material->opacity = opacity;
                    }
                }

                auto mesh = Mesh::create(geometry, material);
                mesh->name = aiMesh->mName.C_Str();
                group->add(mesh);
            }

            auto t = aiNode->mTransformation;
            Matrix4 m;
            m.set(t.a1, t.a2, t.a3, t.a4,
                  t.b1, t.b2, t.b3, t.b4,
                  t.c1, t.c2, t.c3, t.c4,
                  t.d1, t.d2, t.d3, t.d4);
            group->applyMatrix4(m);

            parent.add(group);

            for (unsigned i = 0; i < aiNode->mNumChildren; ++i) {
                parseNodes(path, aiScene, aiNode->mChildren[i], *group);
            }
        }
    };

}// namespace threepp

#endif//THREEPP_ASSIMPLOADER_HPP
