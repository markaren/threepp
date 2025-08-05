
#ifndef THREEPP_ASSIMPLOADER_HPP
#define THREEPP_ASSIMPLOADER_HPP

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/loaders/Loader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

#include <filesystem>
#include <sstream>
#include <utility>

namespace threepp {

    class AssimpLoader: public Loader<Group> {

    public:
        struct AssimpVersion {
            unsigned int major;
            unsigned int minor;
            unsigned int patch;

            friend std::ostream& operator<<(std::ostream& os, const AssimpVersion& v) {
                os << v.major << "." << v.minor << "." << v.patch;
                return os;
            }
        };

        [[nodiscard]] static AssimpVersion getVersion() {
            return {aiGetVersionMajor(), aiGetVersionMinor(), aiGetVersionPatch()};
        }


        std::shared_ptr<Group> load(const std::filesystem::path& path) override {

            auto aiScene = importer_.ReadFile(path.string().c_str(), aiProcessPreset_TargetRealtime_Quality);

            if (!aiScene) {
                throw std::runtime_error(importer_.GetErrorString());
            }

            SceneInfo info(path);
            preParse(info, aiScene, aiScene->mRootNode);

            auto group = Group::create();
            group->name = path.filename().stem().string();
            parseNodes(info, aiScene, aiScene->mRootNode, *group);

            for (unsigned i = 0; i < aiScene->mNumAnimations; i++) {
                const auto aiAnim = aiScene->mAnimations[i];

                std::string name(aiAnim->mName.data);
                auto duration = static_cast<float>(aiAnim->mDuration / aiAnim->mTicksPerSecond);

                std::vector<std::shared_ptr<KeyframeTrack>> tracks;
                for (unsigned j = 0; j < aiAnim->mNumChannels; j++) {

                    const auto aiNodeAnim = aiAnim->mChannels[j];

                    if (auto rotationTrack = loadRotationTrack(aiNodeAnim)){
                        tracks.emplace_back(std::move(rotationTrack));
                    }

                    if (auto positionTrack = loadPositionTrack(aiNodeAnim)){
                        tracks.emplace_back(std::move(positionTrack));
                    }

                    if (auto scaleTrack = loadScaleTrack(aiNodeAnim)){
                        tracks.emplace_back(std::move(scaleTrack));
                    }
                }

                auto clip = std::make_shared<AnimationClip>(name, duration, tracks);
                group->animations.emplace_back(clip);
            }

            return group;
        }

    private:
        TextureLoader texLoader_;
        Assimp::Importer importer_;

        struct SceneInfo;

        static std::unique_ptr<KeyframeTrack> loadRotationTrack(const aiNodeAnim* aiNodeAnim) {
            std::vector<float> times;
            std::vector<float> values;
            std::string name(aiNodeAnim->mNodeName.data);
            // std::erase(name, '.');

            for (auto k = 0; k < aiNodeAnim->mNumRotationKeys; k++) {

                const auto key = aiNodeAnim->mRotationKeys[k];
                times.emplace_back(static_cast<float>(key.mTime / 1000));
                values.insert(values.end(), {static_cast<float>(key.mValue.x), static_cast<float>(key.mValue.y),
                                             static_cast<float>(key.mValue.z), static_cast<float>(key.mValue.w)});
            }

            return std::make_unique<QuaternionKeyframeTrack>(name + ".quaternion", times, values);
        }

        static std::unique_ptr<KeyframeTrack> loadPositionTrack(const aiNodeAnim* aiNodeAnim) {
            std::vector<float> times;
            std::vector<float> values;
            std::string name(aiNodeAnim->mNodeName.data);
            // std::erase(name, '.');

            for (auto k = 0; k < aiNodeAnim->mNumPositionKeys; k++) {

                const auto key = aiNodeAnim->mPositionKeys[k];
                times.emplace_back(static_cast<float>(key.mTime / 1000));
                values.insert(values.end(), {static_cast<float>(key.mValue.x), static_cast<float>(key.mValue.y),
                                             static_cast<float>(key.mValue.z)});
            }

            return std::make_unique<VectorKeyframeTrack>(name + ".position", times, values);
        }

        static std::unique_ptr<KeyframeTrack> loadScaleTrack(const aiNodeAnim* aiNodeAnim) {
            std::vector<float> times;
            std::vector<float> values;
            std::string name(aiNodeAnim->mNodeName.data);
            // std::erase(name, '.');

            for (auto k = 0; k < aiNodeAnim->mNumScalingKeys; k++) {

                const auto key = aiNodeAnim->mScalingKeys[k];
                times.emplace_back(static_cast<float>(key.mTime / 1000));
                values.insert(values.end(), {static_cast<float>(key.mValue.x), static_cast<float>(key.mValue.y),
                                             static_cast<float>(key.mValue.z)});
            }

            return std::make_unique<VectorKeyframeTrack>(name + ".scale", times, values);
        }


        void parseNodes(const SceneInfo& info, const aiScene* aiScene, aiNode* aiNode, Object3D& parent) {

            std::string nodeName(aiNode->mName.data);

            std::shared_ptr<Object3D> group = info.getBone(nodeName);
            if (!group) group = Group::create();
            group->name = nodeName;
            setTransform(*group, aiNode->mTransformation);
            parent.add(group);

            auto meshes = parseNodeMeshes(info, aiScene, aiNode);
            for (const auto& mesh : meshes) group->add(mesh);

            for (unsigned i = 0; i < aiNode->mNumChildren; ++i) {
                parseNodes(info, aiScene, aiNode->mChildren[i], *group);
            }
        }

        std::vector<std::shared_ptr<Mesh>> parseNodeMeshes(const SceneInfo& info, const aiScene* aiScene, const aiNode* aiNode) {

            std::vector<std::shared_ptr<Mesh>> children;

            for (unsigned i = 0; i < aiNode->mNumMeshes; ++i) {

                const auto meshIndex = aiNode->mMeshes[i];
                const auto aiMesh = aiScene->mMeshes[meshIndex];

                auto geometry = BufferGeometry::create();
                auto material = MeshStandardMaterial::create();
                setupMaterial(info.path, aiScene, aiMesh, *material);

                std::shared_ptr<Mesh> mesh;
                if (info.hasSkeleton(meshIndex)) {

                    const auto boneData = info.boneData.at(meshIndex);

                    geometry->setAttribute("skinIndex", FloatBufferAttribute::create(boneData.boneIndices, 4));
                    geometry->setAttribute("skinWeight", FloatBufferAttribute::create(boneData.boneWeights, 4));

                    auto skinnedMesh = SkinnedMesh::create(geometry, material);
                    skinnedMesh->normalizeSkinWeights();

                    auto skeleton = Skeleton::create(boneData.bones, boneData.boneInverses);
                    skinnedMesh->bind(skeleton, Matrix4());

                    mesh = skinnedMesh;
                } else {

                    mesh = Mesh::create(geometry, material);
                }
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
                children.emplace_back(mesh);
            }

            return children;
        }

        struct BoneData {

            std::vector<float> boneIndices;
            std::vector<float> boneWeights;

            std::vector<Matrix4> boneInverses;
            std::vector<std::shared_ptr<Bone>> bones;
        };

        struct SceneInfo {

            std::filesystem::path path;
            std::unordered_map<unsigned int, BoneData> boneData;

            explicit SceneInfo(std::filesystem::path path): path(std::move(path)) {}

            [[nodiscard]] bool hasSkeleton(unsigned int meshIndex) const {
                return boneData.count(meshIndex);
            }

            [[nodiscard]] std::shared_ptr<Bone> getBone(const std::string& name) const {
                for (const auto& [idx, data] : boneData) {
                    for (const auto& bone : data.bones) {
                        if (bone->name.substr(5) == name) {
                            return bone;
                        }
                    }
                }
                return nullptr;
            }
        };

        void preParse(SceneInfo& info, const aiScene* aiScene, aiNode* aiNode) {

            for (unsigned i = 0; i < aiNode->mNumMeshes; ++i) {
                auto meshIndex = aiNode->mMeshes[i];
                auto aiMesh = aiScene->mMeshes[meshIndex];

                if (aiMesh->HasBones()) {
                    BoneData data;

                    std::vector<std::vector<float>> boneIndices;
                    std::vector<std::vector<float>> boneWeights;

                    for (auto j = 0; j < aiMesh->mNumBones; j++) {

                        const auto aiBone = aiMesh->mBones[j];
                        std::string boneName(aiBone->mName.data);

                        std::shared_ptr<Bone> bone;
                        if (auto oldBone = info.getBone(boneName)) {
                            bone = oldBone;
                        } else {
                            bone = Bone::create();
                            bone->name = "Bone:" + boneName;
                        }

                        data.bones.emplace_back(bone);

                        data.boneInverses.emplace_back(aiMatrixToMatrix4(aiBone->mOffsetMatrix));

                        for (auto k = 0; k < aiBone->mNumWeights; k++) {
                            const auto aiWeight = aiBone->mWeights[k];

                            while (boneWeights.size() <= aiWeight.mVertexId) boneWeights.emplace_back();
                            while (boneIndices.size() <= aiWeight.mVertexId) boneIndices.emplace_back();

                            boneWeights[aiWeight.mVertexId].emplace_back(aiWeight.mWeight);
                            boneIndices[aiWeight.mVertexId].emplace_back(static_cast<float>(j));
                        }
                    }

                    for (unsigned j = 0; j < boneIndices.size(); j++) {

                        sortWeights(boneIndices[j], boneWeights[j]);
                    }

                    for (unsigned j = 0; j < boneWeights.size(); j++) {

                        for (unsigned k = 0; k < 4; k++) {

                            if (!boneWeights[j].empty() && !boneIndices[j].empty()) {

                                const auto weight = boneWeights[j][k];
                                const auto index = boneIndices[j][k];

                                data.boneWeights.emplace_back(weight);
                                data.boneIndices.emplace_back(index);

                            } else {

                                data.boneWeights.emplace_back(0.f);
                                data.boneIndices.emplace_back(0.f);
                            }
                        }
                    }
                    info.boneData[meshIndex] = data;
                }
            }

            for (unsigned i = 0; i < aiNode->mNumChildren; ++i) {
                preParse(info, aiScene, aiNode->mChildren[i]);
            }
        }


        static void sortWeights(std::vector<float>& indexes, std::vector<float>& weights) {

            std::vector<std::pair<float, float>> pairs;

            for (unsigned i = 0; i < indexes.size(); i++) {

                pairs.emplace_back(indexes[i], weights[i]);
            }

            std::ranges::stable_sort(pairs, [](const auto& a, const auto& b) {
                return b.second < a.second;
            });

            while (pairs.size() < 4) pairs.emplace_back(0.f, 0.f);
            while (pairs.size() > 4) pairs.pop_back();

            float sum = 0;
            for (unsigned i = 0; i < 4; i++) {

                sum += pairs[i].second * pairs[i].second;
            }
            sum = std::sqrt(sum);

            for (unsigned i = 0; i < 4; i++) {

                pairs[i].second = pairs[i].second / sum;

                while (indexes.size() <= i) indexes.emplace_back();
                while (weights.size() <= i) weights.emplace_back();

                indexes[i] = pairs[i].first;
                weights[i] = pairs[i].second;
            }
        }

        static void handleWrapping(const aiMaterial* mat, aiTextureType mode, Texture& tex) {

            aiTextureMapMode wrapS;
            if (AI_SUCCESS == mat->Get(AI_MATKEY_MAPPINGMODE_U(mode, 0), wrapS)) {
                switch (wrapS) {
                    case aiTextureMapMode_Wrap:
                        tex.wrapS = TextureWrapping::Repeat;
                        break;
                    case aiTextureMapMode_Mirror:
                        tex.wrapS = TextureWrapping::MirroredRepeat;
                        break;
                    case aiTextureMapMode_Clamp:
                        tex.wrapS = TextureWrapping::ClampToEdge;
                        break;
                }
            }
            aiTextureMapMode wrapT;
            if (AI_SUCCESS == mat->Get(AI_MATKEY_MAPPINGMODE_V(mode, 0), wrapT)) {
                switch (wrapT) {
                    case aiTextureMapMode_Wrap:
                        tex.wrapT = TextureWrapping::Repeat;
                        break;
                    case aiTextureMapMode_Mirror:
                        tex.wrapT = TextureWrapping::MirroredRepeat;
                        break;
                    case aiTextureMapMode_Clamp:
                        tex.wrapT = TextureWrapping::ClampToEdge;
                        break;
                }
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

                        handleWrapping(mat, aiTextureType_DIFFUSE, *tex);
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

                        handleWrapping(mat, aiTextureType_EMISSIVE, *tex);
                    }
                } else {
                    C_STRUCT aiColor4D emissive;
                    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_EMISSIVE, &emissive)) {
                        material.emissive.setRGB(emissive.r, emissive.g, emissive.b);
                    }
                }

                float emmisiveIntensity;
                if (AI_SUCCESS == aiGetMaterialFloat(mat, AI_MATKEY_EMISSIVE_INTENSITY, &emmisiveIntensity)) {
                    material.emissiveIntensity = emmisiveIntensity;
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
                    material.transparent = (opacity < 1.f);
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

        static Matrix4 aiMatrixToMatrix4(const aiMatrix4x4& t) {
            Matrix4 m;
            m.set(t.a1, t.a2, t.a3, t.a4,
                  t.b1, t.b2, t.b3, t.b4,
                  t.c1, t.c2, t.c3, t.c4,
                  t.d1, t.d2, t.d3, t.d4);

            return m;
        }

        static void setTransform(Object3D& obj, const aiMatrix4x4& t) {
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
