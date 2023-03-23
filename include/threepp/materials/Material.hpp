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

    typedef std::variant<bool, int, float, Color, std::string, std::shared_ptr<Texture>> MaterialValue;

    class Material: public EventDispatcher, public std::enable_shared_from_this<Material> {

    public:
        const unsigned int id = materialId++;

        std::string name;

        bool fog = true;

        int blending = NormalBlending;
        int side = FrontSide;
        bool vertexColors = false;

        float opacity = 1;
        bool transparent = false;

        int blendSrc = SrcAlphaFactor;
        int blendDst = OneMinusSrcAlphaFactor;
        int blendEquation = AddEquation;
        std::optional<int> blendSrcAlpha;
        std::optional<int> blendDstAlpha;
        std::optional<int> blendEquationAlpha;

        int depthFunc = LessEqualDepth;
        bool depthTest = true;
        bool depthWrite = true;

        int stencilWriteMask = 0xff;
        int stencilFunc = AlwaysStencilFunc;
        int stencilRef = 0;
        int stencilFuncMask = 0xff;
        int stencilFail = KeepStencilOp;
        int stencilZFail = KeepStencilOp;
        int stencilZPass = KeepStencilOp;
        bool stencilWrite = false;

        std::vector<Plane> clippingPlanes;
        bool clipIntersection = false;
        bool clipShadows = false;
        bool clipping = false;

        std::optional<int> shadowSide{};

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

        unsigned int version = 0;

        Material(const Material&) = delete;

        std::string uuid() const;

        void setValues(const std::unordered_map<std::string, MaterialValue>& values);

        void dispose();

        void needsUpdate();

        [[nodiscard]] virtual std::string type() const = 0;

        template<class T>
        std::shared_ptr<T> as() {

            auto m = shared_from_this();
            return std::dynamic_pointer_cast<T>(m);
        }

        template<class T>
        bool is() {

            return dynamic_cast<T*>(this) != nullptr;
        }

        virtual std::shared_ptr<Material> clone() const { return nullptr; };

        ~Material() override;

    protected:
        Material();

        void copyInto(Material* m) const;

        virtual bool setValue(const std::string& key, const MaterialValue& value);

    private:
        bool disposed_ = false;
        std::string uuid_;
        inline static unsigned int materialId = 0;
    };


}// namespace threepp

#endif//THREEPP_MATERIAL_HPP
