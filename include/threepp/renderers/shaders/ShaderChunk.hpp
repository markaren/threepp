// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderChunk.js

#ifndef THREEPP_SHADERCHUNK_HPP
#define THREEPP_SHADERCHUNK_HPP

#include <string>
#include <unordered_map>

namespace threepp::shaders {

    class ShaderChunk {

    public:
        ShaderChunk(const ShaderChunk&) = delete;
        void operator=(const ShaderChunk&) = delete;

        const std::string& default_vertex() {
            return get("default_vertex");
        }

        const std::string& default_fragment() {
            return get("default_fragment");
        }

        const std::string& alphamap_fragment() {
            return get("alphamap_fragment");
        }

        const std::string& alphamap_pars_fragment() {
            return get("alphamap_pars_fragment");
        }

        const std::string& alphatest_fragment() {
            return get("alphatest_fragment");
        }

        const std::string& aomap_fragment() {
            return get("aomap_fragment");
        }

        const std::string& aomap_pars_fragment() {
            return get("aomap_pars_fragment");
        }

        const std::string& begin_vertex() {
            return get("begin_vertex");
        }

        const std::string& beginnormal_vertex() {
            return get("beginnormal_vertex");
        }

        const std::string& bsdfs() {
            return get("bsdfs");
        }

        const std::string& bumpmap_pars_fragment() {
            return get("bumpmap_pars_fragment");
        }

        const std::string& clipping_planes_fragment() {
            return get("clipping_planes_fragment");
        }

        const std::string& clipping_planes_pars_fragment() {
            return get("clipping_planes_pars_fragment");
        }

        const std::string& clipping_planes_pars_vertex() {
            return get("clipping_planes_pars_vertex");
        }

        const std::string& clipping_planes_vertex() {
            return get("clipping_planes_vertex");
        }

        const std::string& color_fragment() {
            return get("color_fragment");
        }

        const std::string& color_pars_fragment() {
            return get("color_pars_fragment");
        }

        const std::string& color_pars_vertex() {
            return get("color_pars_vertex");
        }

        const std::string& color_vertex() {
            return get("color_vertex");
        }

        const std::string& common() {
            return get("common");
        }

        const std::string& cube_uv_reflection_fragment() {
            return get("cube_uv_reflection_fragment");
        }

        const std::string& defaultnormal_vertex() {
            return get("defaultnormal_vertex");
        }

        const std::string& displacementmap_pars_vertex() {
            return get("displacementmap_pars_vertex");
        }

        const std::string& displacementmap_vertex() {
            return get("displacementmap_vertex");
        }

        const std::string& emissivemap_fragment() {
            return get("emissivemap_fragment");
        }

        const std::string& emissivemap_pars_fragment() {
            return get("emissivemap_pars_fragment");
        }

        const std::string& encodings_fragment() {
            return get("encodings_fragment");
        }

        const std::string& encodings_pars_fragment() {
            return get("encodings_pars_fragment");
        }

        const std::string& envmap_fragment() {
            return get("envmap_fragment");
        }

        const std::string& envmap_common_pars_fragment() {
            return get("envmap_common_pars_fragment");
        }

        const std::string& envmap_pars_fragment() {
            return get("envmap_pars_fragment");
        }

        const std::string& envmap_pars_vertex() {
            return get("envmap_pars_vertex");
        }

        const std::string& envmap_physical_pars_fragment() {
            return get("envmap_physical_pars_fragment");
        }

        const std::string& envmap_vertex() {
            return get("envmap_vertex");
        }

        const std::string& fog_vertex() {
            return get("fog_vertex");
        }

        const std::string& fog_pars_vertex() {
            return get("fog_pars_vertex");
        }

        const std::string& fog_fragment() {
            return get("fog_fragment");
        }

        const std::string& fog_pars_fragment() {
            return get("fog_pars_fragment");
        }

        const std::string& gradientmap_pars_fragment() {
            return get("gradientmap_pars_fragment");
        }

        const std::string& lightmap_fragment() {
            return get("lightmap_fragment");
        }

        const std::string& lightmap_pars_fragment() {
            return get("lightmap_pars_fragment");
        }

        const std::string& lights_lambert_vertex() {
            return get("lights_lambert_vertex");
        }

        const std::string& lights_pars_begin() {
            return get("lights_pars_begin");
        }

        const std::string& lights_toon_fragment() {
            return get("lights_toon_fragment");
        }

        const std::string& lights_toon_pars_fragment() {
            return get("lights_toon_pars_fragment");
        }

        const std::string& lights_phong_fragment() {
            return get("lights_phong_fragment");
        }

        const std::string& lights_phong_pars_fragment() {
            return get("lights_phong_pars_fragment");
        }

        const std::string& lights_physical_fragment() {
            return get("lights_physical_fragment");
        }

        const std::string& lights_physical_pars_fragment() {
            return get("lights_physical_pars_fragment");
        }

        const std::string& lights_fragment_begin() {
            return get("lights_fragment_begin");
        }

        const std::string& lights_fragment_maps() {
            return get("lights_fragment_maps");
        }

        const std::string& lights_fragment_end() {
            return get("lights_fragment_end");
        }

        const std::string& logdepthbuf_fragment() {
            return get("logdepthbuf_fragment");
        }

        const std::string& logdepthbuf_pars_fragment() {
            return get("logdepthbuf_pars_fragment");
        }

        const std::string& logdepthbuf_pars_vertex() {
            return get("logdepthbuf_pars_vertex");
        }

        const std::string& logdepthbuf_vertex() {
            return get("logdepthbuf_vertex");
        }

        const std::string& map_fragment() {
            return get("map_fragment");
        }

        const std::string& map_pars_fragment() {
            return get("map_pars_fragment");
        }

        const std::string& map_particle_fragment() {
            return get("map_particle_fragment");
        }

        const std::string& map_particle_pars_fragment() {
            return get("map_particle_pars_fragment");
        }

        const std::string& metalnessmap_fragment() {
            return get("metalnessmap_fragment");
        }

        const std::string& metalnessmap_pars_fragment() {
            return get("metalnessmap_pars_fragment");
        }

        const std::string& morphnormal_vertex() {
            return get("morphnormal_vertex");
        }

        const std::string& morphtarget_pars_vertex() {
            return get("morphtarget_pars_vertex");
        }

        const std::string& morphtarget_vertex() {
            return get("morphtarget_vertex");
        }

        const std::string& normal_fragment_begin() {
            return get("normal_fragment_begin");
        }

        const std::string& normal_fragment_maps() {
            return get("normal_fragment_maps");
        }

        const std::string& normalmap_pars_fragment() {
            return get("normalmap_pars_fragment");
        }

        const std::string& clearcoat_normal_fragment_begin() {
            return get("clearcoat_normal_fragment_begin");
        }

        const std::string& clearcoat_normal_fragment_maps() {
            return get("clearcoat_normal_fragment_maps");
        }

        const std::string& clearcoat_pars_fragment() {
            return get("clearcoat_pars_fragment");
        }

        const std::string& packing() {
            return get("packing");
        }

        const std::string& premultiplied_alpha_fragment() {
            return get("premultiplied_alpha_fragment");
        }

        const std::string& project_vertex() {
            return get("project_vertex");
        }

        const std::string& dithering_fragment() {
            return get("dithering_fragment");
        }

        const std::string& dithering_pars_fragment() {
            return get("dithering_pars_fragment");
        }

        const std::string& roughnessmap_fragment() {
            return get("roughnessmap_fragment");
        }

        const std::string& roughnessmap_pars_fragment() {
            return get("roughnessmap_pars_fragment");
        }

        const std::string& shadowmap_pars_fragment() {
            return get("shadowmap_pars_fragment");
        }

        const std::string& shadowmap_pars_vertex() {
            return get("shadowmap_pars_vertex");
        }

        const std::string& shadowmap_vertex() {
            return get("shadowmap_vertex");
        }

        const std::string& shadowmask_pars_fragment() {
            return get("shadowmask_pars_fragment");
        }

        const std::string& skinbase_vertex() {
            return get("skinbase_vertex");
        }

        const std::string& skinning_pars_vertex() {
            return get("skinning_pars_vertex");
        }

        const std::string& skinning_vertex() {
            return get("skinning_vertex");
        }

        const std::string& skinnormal_vertex() {
            return get("skinnormal_vertex");
        }

        const std::string& specularmap_fragment() {
            return get("specularmap_fragment");
        }

        const std::string& specularmap_pars_fragment() {
            return get("specularmap_pars_fragment");
        }

        const std::string& tonemapping_fragment() {
            return get("tonemapping_fragment");
        }

        const std::string& tonemapping_pars_fragment() {
            return get("tonemapping_pars_fragment");
        }

        const std::string& transmission_fragment() {
            return get("transmission_fragment");
        }

        const std::string& transmission_pars_fragment() {
            return get("transmission_pars_fragment");
        }

        const std::string& uv_pars_fragment() {
            return get("uv_pars_fragment");
        }

        const std::string& uv_pars_vertex() {
            return get("uv_pars_vertex");
        }

        const std::string& uv_vertex() {
            return get("uv_vertex");
        }

        const std::string& uv2_pars_fragment() {
            return get("uv2_pars_fragment");
        }

        const std::string& uv2_pars_vertex() {
            return get("uv2_pars_vertex");
        }

        const std::string& uv2_vertex() {
            return get("uv2_vertex");
        }

        const std::string& worldpos_vertex() {
            return get("worldpos_vertex");
        }

        const std::string& background_frag() {
            return get("background_frag");
        }

        const std::string& background_vert() {
            return get("background_vert");
        }

        const std::string& cube_frag() {
            return get("cube_frag");
        }

        const std::string& cube_vert() {
            return get("cube_vert");
        }

        const std::string& depth_frag() {
            return get("depth_frag");
        }

        const std::string& depth_vert() {
            return get("depth_vert");
        }

        const std::string& distanceRGBA_frag() {
            return get("distanceRGBA_frag");
        }

        const std::string& distanceRGBA_vert() {
            return get("distanceRGBA_vert");
        }

        const std::string& equirect_frag() {
            return get("equirect_frag");
        }

        const std::string& equirect_vert() {
            return get("equirect_vert");
        }

        const std::string& linedashed_frag() {
            return get("linedashed_frag");
        }

        const std::string& linedashed_vert() {
            return get("linedashed_vert");
        }

        const std::string& meshbasic_frag() {
            return get("meshbasic_frag");
        }

        const std::string& meshbasic_vert() {
            return get("meshbasic_vert");
        }

        const std::string& meshlambert_frag() {
            return get("meshlambert_frag");
        }

        const std::string& meshlambert_vert() {
            return get("meshlambert_vert");
        }

        const std::string& meshmatcap_frag() {
            return get("meshmatcap_frag");
        }

        const std::string& meshmatcap_vert() {
            return get("meshmatcap_vert");
        }

        const std::string& meshtoon_frag() {
            return get("meshtoon_frag");
        }

        const std::string& meshtoon_vert() {
            return get("meshtoon_vert");
        }

        const std::string& meshphong_frag() {
            return get("meshphong_frag");
        }

        const std::string& meshphong_vert() {
            return get("meshphong_vert");
        }

        const std::string& meshphysical_frag() {
            return get("meshphysical_frag");
        }

        const std::string& meshphysical_vert() {
            return get("meshphysical_vert");
        }

        const std::string& normal_frag() {
            return get("normal_frag");
        }

        const std::string& normal_vert() {
            return get("normal_vert");
        }

        const std::string& points_frag() {
            return get("points_frag");
        }

        const std::string& points_vert() {
            return get("points_vert");
        }

        const std::string& shadow_frag() {
            return get("shadow_frag");
        }

        const std::string& shadow_vert() {
            return get("shadow_vert");
        }

        const std::string& sprite_frag() {
            return get("sprite_frag");
        }

        const std::string& sprite_vert() {
            return get("sprite_vert");
        }

        const std::string& get(const std::string& key) {
            return data_.at(key);
        }

        static ShaderChunk& instance() {
            static ShaderChunk instance;
            return instance;
        }

    private:
        std::unordered_map<std::string, std::string> data_;

        ShaderChunk();

        ~ShaderChunk() = default;
    };

}// namespace threepp::shaders

#endif//THREEPP_SHADERCHUNK_HPP
