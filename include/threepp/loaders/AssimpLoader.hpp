
#ifndef THREEPP_ASSIMPLOADER_HPP
#define THREEPP_ASSIMPLOADER_HPP

#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <filesystem>
#include <sstream>

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
            setTransform(*group, aiNode->mTransformation);
            parent.add(group);

            for (unsigned i = 0; i < aiNode->mNumMeshes; ++i) {

                auto aiMesh = aiScene->mMeshes[aiNode->mMeshes[i]];

                auto geometry = BufferGeometry::create();
                auto material = MeshStandardMaterial::create();
                setupMaterial(path, aiScene, aiMesh, *material);

                auto mesh = Mesh::create(geometry, material);
                mesh->name = aiMesh->mName.C_Str();

                std::vector<unsigned int> indices;
                std::vector<float> vertices;
                std::vector<float> normals;
                std::vector<float> colors;
                std::vector<float> uvs;
                std::vector<std::vector<float>> morphPositions(aiMesh->mNumAnimMeshes);

                if (aiMesh->HasFaces()) {

                    // Populate the index buffer
                    for (unsigned j = 0; j < aiMesh->mNumFaces; j++) {
                        const aiFace& face = aiMesh->mFaces[j];
                        indices.push_back(face.mIndices[0]);
                        indices.push_back(face.mIndices[1]);
                        indices.push_back(face.mIndices[2]);
                    }

                    const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
                    // Populate the vertex attribute vectors
                    for (unsigned j = 0; j < aiMesh->mNumVertices; j++) {
                        const auto pos = aiMesh->mVertices[j];
                        const auto normal = aiMesh->mNormals[j];
                        const auto texCoord = aiMesh->HasTextureCoords(0) ? aiMesh->mTextureCoords[0][j] : Zero3D;
                        if (aiMesh->HasVertexColors(0)) {
                            const auto color = aiMesh->mColors[0][j];
                            colors.insert(colors.end(), {color.r, color.g, color.b, color.a});
                        }

                        for (auto k = 0; k < aiMesh->mNumAnimMeshes; k++) {
                            auto& list = morphPositions[k];
                            const auto mp = aiMesh->mAnimMeshes[k]->mVertices[j];
                            list.insert(list.end(), {mp.x, mp.y, mp.z});
                        }

                        vertices.insert(vertices.end(), {pos.x, pos.y, pos.z});
                        normals.insert(normals.end(), {normal.x, normal.y, normal.z});
                        uvs.insert(uvs.end(), {texCoord.x, texCoord.y});
                    }
                }

                if (!indices.empty()) {
                    geometry->setIndex(indices);
                }

                geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
                if (!normals.empty()) {
                    geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
                }
                if (!colors.empty()) {
                    material->vertexColors = true;
                    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 4));
                }
                if (!uvs.empty()) {
                    geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
                }

                if (!morphPositions.empty()) {
                    for (const auto& pos : morphPositions) {
                        geometry->getOrCreateMorphAttribute("position")->emplace_back(FloatBufferAttribute::create(pos, 3));
                        mesh->morphTargetInfluences().emplace_back();
                    }
                }

                group->add(mesh);
            }

            for (unsigned i = 0; i < aiNode->mNumChildren; ++i) {
                parseNodes(path, aiScene, aiNode->mChildren[i], *group);
            }
        }

        void setupMaterial(const std::filesystem::path& path, const aiScene* aiScene, const aiMesh* aiMesh, MeshStandardMaterial& material) {
            if (aiScene->HasMaterials()) {
                aiString p;
                auto mi = aiMesh->mMaterialIndex;
                auto mat = aiScene->mMaterials[mi];

                if (aiGetMaterialTextureCount(mat, aiTextureType_DIFFUSE) > 0) {
                    if (aiGetMaterialTexture(mat, aiTextureType_DIFFUSE, 0, &p) == aiReturn_SUCCESS) {
                        auto tex = loadTexture(aiScene, path, p.C_Str());
                        material.map = tex;
                    }
                } else {
                    C_STRUCT aiColor4D diffuse;
                    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse)) {
                        material.color.setRGB(diffuse.r, diffuse.g, diffuse.b);
                    }
                }

                if (aiGetMaterialTextureCount(mat, aiTextureType_EMISSIVE) > 0) {
                    if (aiGetMaterialTexture(mat, aiTextureType_EMISSIVE, 0, &p) == aiReturn_SUCCESS) {
                        auto tex = loadTexture(aiScene, path, p.C_Str());
                        material.emissiveMap = tex;
                    }
                } else {
                    C_STRUCT aiColor4D emissive;
                    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_EMISSIVE, &emissive)) {
                        material.emissive.setRGB(emissive.r, emissive.g, emissive.b);
                    }
                }

                //                if (aiGetMaterialTextureCount(mat, aiTextureType_SPECULAR) > 0) {
                //                    if (aiGetMaterialTexture(mat, aiTextureType_SPECULAR, 0, &p) == aiReturn_SUCCESS) {
                //                        auto tex = loadTexture(aiScene, path, p.C_Str());
                //                        material.specularMap = tex;
                //                    }
                //                } else {
                //                    C_STRUCT aiColor4D specular;
                //                    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_SPECULAR, &specular)) {
                //                        material.specular.setRGB(specular.r, specular.g, specular.b);
                //                    }
                //                }

                //                float shininess;
                //                if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_SHININESS, &shininess)) {
                //                    material.shininess = shininess;
                //                }

                float emmisiveIntensity;
                if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_EMISSIVE_INTENSITY, &emmisiveIntensity)) {
                    material.emissiveIntensity = emmisiveIntensity;
                }

                // should this be added?
                //                C_STRUCT aiColor4D ambient;
                //                if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_AMBIENT, &ambient)) {
                //                    material.color.add(Color().setRGB(ambient.r, ambient.g, ambient.b));
                //                }
                //
                //                if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &ambient)) {
                //                    material.color.setRGB(ambient.r, ambient.g, ambient.b);
                //                }

                float opacity;
                if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_OPACITY, &opacity)) {
                    material.transparent = true;
                    material.opacity = opacity;
                }
            }
        }

        std::shared_ptr<Texture> loadTexture(const aiScene* aiScene, const std::filesystem::path& path, const std::string& name) {

            std::shared_ptr<Texture> tex;

            if (name[0] == '*') {

                // embedded texture
                const auto embed = aiScene->GetEmbeddedTexture(name.c_str());

                std::stringstream ss;
                ss << embed->mFilename.C_Str() << "." << embed->achFormatHint;

                if (embed->mHeight == 0) {

                    std::vector<unsigned char> data(embed->mWidth);
                    std::copy((unsigned char*) embed->pcData, (unsigned char*) embed->pcData + data.size(), data.begin());
                    tex = texLoader_.loadFromMemory(ss.str(), data);

                } else {

                    std::vector<unsigned char> data(embed->mWidth * embed->mHeight);
                    std::copy((unsigned char*) embed->pcData, (unsigned char*) embed->pcData + data.size(), data.begin());
                    tex = texLoader_.loadFromMemory(ss.str(), data);
                }
            } else {

                auto texPath = path.parent_path() / name;
                tex = texLoader_.load(texPath);
            }

            return tex;
        }

        void setTransform(Object3D& obj, const aiMatrix4x4& t) {
            aiVector3t<float> pos;
            aiQuaterniont<float> quat;
            aiVector3t<float> scale;
            t.Decompose(scale, quat, pos);

            Matrix4 m;
            m.makeRotationFromQuaternion(Quaternion{quat.x, quat.y, quat.z, quat.w});
            m.setPosition({pos.x, pos.y, pos.z});

            obj.applyMatrix4(m);
            obj.scale.set(scale.x, scale.y, scale.z);
        }
    };

}// namespace threepp

#endif//THREEPP_ASSIMPLOADER_HPP
