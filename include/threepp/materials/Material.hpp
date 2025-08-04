// https://github.com/mrdoob/three.js/blob/r129/src/materials/Material.js

#ifndef THREEPP_MATERIAL_HPP
#define THREEPP_MATERIAL_HPP

#include "threepp/constants.hpp"
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/core/Uniform.hpp"
#include "threepp/math/Plane.hpp"

#include <optional>
#include <variant>

namespace threepp {

    typedef std::variant<bool, int, float, Vector2, Side, Blending, BlendFactor, BlendEquation, StencilFunc, StencilOp, CombineOperation, DepthFunc, NormalMapType, Color, std::string, std::shared_ptr<Texture>> MaterialValue;

    class Material: public EventDispatcher {

    public:
        const unsigned int id = materialId++;

        std::string name;

        bool fog = true;

        Blending blending = Blending::Normal;
        Side side{Side::Front};
        bool vertexColors = false;

        float opacity = 1;
        bool transparent = false;

        BlendFactor blendSrc = BlendFactor::SrcAlpha;
        BlendFactor blendDst = BlendFactor::OneMinusSrcAlpha;
        BlendEquation blendEquation = BlendEquation::Add;
        std::optional<BlendFactor> blendSrcAlpha;
        std::optional<BlendFactor> blendDstAlpha;
        std::optional<BlendEquation> blendEquationAlpha;

        DepthFunc depthFunc{DepthFunc::LessEqual};
        bool depthTest = true;
        bool depthWrite = true;

        int stencilWriteMask = 0xff;
        StencilFunc stencilFunc{StencilFunc::Always};
        int stencilRef = 0;
        int stencilFuncMask = 0xff;
        StencilOp stencilFail{StencilOp::Keep};
        StencilOp stencilZFail{StencilOp::Keep};
        StencilOp stencilZPass{StencilOp::Keep};
        bool stencilWrite = false;

        std::vector<Plane> clippingPlanes;
        bool clipIntersection = false;
        bool clipShadows = false;
        bool clipping = false;

        std::optional<Side> shadowSide{};

        bool colorWrite = true;

        bool polygonOffset = false;
        float polygonOffsetFactor = 0;
        float polygonOffsetUnits = 0;

        bool dithering = false;

        float alphaTest = 0;
        bool alphaToCoverage = false;
        bool premultipliedAlpha = false;

        bool visible = true;

        bool toneMapped = true;

        std::unordered_map<std::string, UniformValue> defaultAttributeValues;

        Material(Material&&) = delete;
        Material& operator=(Material&&) = delete;
        Material(const Material&) = delete;
        Material& operator=(const Material&) = delete;

        [[nodiscard]] std::string uuid() const;

        [[nodiscard]] unsigned int version() const;

        void setValues(const std::unordered_map<std::string, MaterialValue>& values);

        void dispose();

        void needsUpdate();

        [[nodiscard]] virtual std::string type() const = 0;

        template<class T>
            requires std::derived_from<T, Material>
        T* as() {

            return dynamic_cast<T*>(this);
        }

        template<class T>
            requires std::derived_from<T, Material>
        [[nodiscard]] bool is() const {

            return dynamic_cast<const T*>(this) != nullptr;
        }

        template<class T = Material>
            requires std::derived_from<T, Material>
        [[nodiscard]] std::shared_ptr<T> clone() const {

            auto clone = createDefault();
            copyInto(*clone);

            return std::dynamic_pointer_cast<T>(clone);
        }

        ~Material() override;

    protected:
        Material();

        virtual std::shared_ptr<Material> createDefault() const = 0;

        virtual void copyInto(Material& material) const;

        static Color extractColor(const MaterialValue& value) {
            if (std::holds_alternative<int>(value)) {
                return std::get<int>(value);
            } else {
                return std::get<Color>(value);
            }
        }

        static float extractFloat(const MaterialValue& value) {
            if (std::holds_alternative<int>(value)) {
                return std::get<int>(value);
            } else {
                return std::get<float>(value);
            }
        }

        virtual bool setValue(const std::string& key, const MaterialValue& value);

    private:
        bool disposed_ = false;
        std::string uuid_;
        unsigned int version_ = 0;
        inline static unsigned int materialId = 0;
    };


}// namespace threepp

#endif//THREEPP_MATERIAL_HPP
