
#ifndef THREEPP_TEXT_HPP
#define THREEPP_TEXT_HPP

#include "threepp/geometries/ExtrudeTextGeometry.hpp"
#include "threepp/geometries/TextGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class Text2D: public Mesh {

    public:
        Text2D(const TextGeometry::Options& opts, const std::string& str = "", const std::shared_ptr<Material>& material = nullptr)
            : Mesh(TextGeometry::create(str, opts), material ? material : SpriteMaterial::create()), options(opts) {}

        void setColor(const Color& color) {

            if (auto m = materials_.front()->as<MaterialWithColor>()) {
                m->color = color;
            }
        }

        void setText(const std::string& str) {

            auto geometry = TextGeometry::create(str, options);
            setGeometry(geometry);
        }

        void setText(const std::string& str, const TextGeometry::Options& opts) {

            auto geometry = TextGeometry::create(str, opts);
            setGeometry(geometry);
        }

        static std::shared_ptr<Text2D> create(const TextGeometry::Options& opts, const std::string& str = "", const std::shared_ptr<Material>& material = nullptr) {

            return std::make_shared<Text2D>(opts, str, material);
        }

    private:
        TextGeometry::Options options;
    };

    class Text3D: public Mesh {

    public:
        Text3D(const ExtrudeTextGeometry::Options& opts, const std::string& str = "", const std::shared_ptr<Material>& material = nullptr)
            : Mesh(ExtrudeTextGeometry::create(str, opts), material ? material : MeshBasicMaterial::create()) {}

        void setColor(const Color& color) {

            if (auto m = materials_.front()->as<MaterialWithColor>()) {
                m->color = color;
            }
        }

        void setText(const std::string& str, const ExtrudeTextGeometry::Options& opts) {

            auto geometry = ExtrudeTextGeometry::create(str, opts);
            setGeometry(geometry);
        }

        static std::shared_ptr<Text3D> create(const ExtrudeTextGeometry::Options& opts, const std::string& str = "", const std::shared_ptr<Material>& material = nullptr) {

            return std::make_shared<Text3D>(opts, str, material);
        }
    };

}// namespace threepp

#endif//THREEPP_TEXT_HPP
