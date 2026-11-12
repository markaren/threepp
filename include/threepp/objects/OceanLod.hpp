#ifndef THREEPP_OCEAN_LOD_HPP
#define THREEPP_OCEAN_LOD_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace threepp::oceanlod {
// Fixed patch slots and indices. Rest coordinates live on a world lattice;
// only the CDLOD transition vertices move. The contact square never coarsens.
inline constexpr unsigned cells = 16, side = cells + 1, slots = 6144;
inline constexpr unsigned vertices = side * side * slots;
struct Patch { float x, z, spacing, active; };
// Retain <=64 m cells through 9 km before morphing: the 12 s swell must
// remain above geometric Nyquist. Beyond that the field fades to curvature.
inline float range(float spacing) { return spacing>=64.f ? 156.25f*spacing : 64.f*spacing; }
// THREEPP_OCEAN_CDLOD_FINE=<floor m>[,<px>] (unset: nothing here changes). The finest cells were fixed at 0.5 m, so a camera a
// metre or two over the water saw cells hundreds of pixels wide. With the switch the levels under 2 m follow the
// projected width: a level is used out to where its cell is <px> pixels wide (of a 1080-line frame; 24 is what the
// fixed 64-cell rule gives a 40 deg lens), and the quadtree goes on halving near the camera down to <floor>
// (the fine cascade's texels are 4 cm: 6-12 cm cells are what it can displace). Each level stays inside the next
// coarser one's range (0.875 of it), and the selection drops its finest level while the patch slots do not suffice.
// The extra refinement is kept to the lens's horizontal wedge (1.25 of its half angle + 3 deg): a 5 deg lens asks for
// 6 cm cells 60 m out, which only a thin wedge can have within the patch slots. (At the wedge's sides a refined patch
// meets an unrefined one unmorphed: a possible crack, off screen by the margin.)
struct Fine { bool on=false; float floor=.5f, px=24.f, proj=2.7475f, dirX=0.f, dirZ=-1.f, halfAngle=4.f; };
inline Fine& fine() {
    static Fine f=[]{
        Fine g;
        if (const char* v=std::getenv("THREEPP_OCEAN_CDLOD_FINE")) {
            char* e=nullptr; const float fl=std::strtof(v,&e);
            if (fl>0.f && fl<=.5f) { g.on=true; g.floor=fl; if (e && *e==',') { const float p=std::strtof(e+1,nullptr); if (p>=2.f) g.px=p; } }
        }
        return g;
    }();
    return f;
}
inline float fineReach() { return std::clamp(fine().proj*540.f/fine().px,64.f,1024.f); }// cells a level reaches out to
inline float fineRange(float spacing) {
    float r=128.f; const float reach=fineReach();
    for (float t=1.f; t>=spacing*.999f; t*=.5f) r=std::min(std::max(reach*t,64.f*t),.875f*r);
    return r;
}
inline bool fineInView(float x,float z,float width,float eyeX,float eyeZ) {
    const auto& f=fine();
    if (f.halfAngle>=3.f) return true;
    const float cx=x+.5f*width-eyeX, cz=z+.5f*width-eyeZ, rad=.7072f*width, dist=std::hypot(cx,cz);
    if (dist<=rad) return true;
    return std::acos(std::clamp((cx*f.dirX+cz*f.dirZ)/dist,-1.f,1.f)) <= f.halfAngle+std::asin(rad/dist);
}
inline int fineMinLevel() { return fine().on ? -int(std::ceil(std::log2(.5f/fine().floor)-1e-4f)) : 0; }
inline float altitudeBlend(float height) {float a=std::clamp((height-80.f)/100.f,0.f,1.f);return a*a*(3.f-2.f*a);}
inline float cameraRange(float spacing,float height) {
    const float a=altitudeBlend(height);
    return spacing<2.f ? (fine().on?fineRange(spacing):range(spacing))*(1.f-a)+(spacing>=1.f?32.f*a:0.f) : range(spacing)+a*std::min(128.f*spacing,768.f);
}
inline float normalizedDistance(float x,float z,float eyeX,float eyeZ,float spacing,float height) {
    const float camera=std::max(std::abs(x-eyeX),std::abs(z-eyeZ))/std::max(1e-6f,cameraRange(spacing,height));
    const float contact=std::max(0.f,std::max(std::abs(x),std::abs(z))-128.f)/range(spacing);
    return spacing<.5f ? camera : std::min(camera,contact);// (the levels under 0.5 m are the camera's only)
}
inline bool refine(float x,float z,float width,float eyeX,float eyeZ,float spacing,float height) {
    const float camera=std::max({x-eyeX,eyeX-x-width,z-eyeZ,eyeZ-z-width,0.f});
    const float contact=std::max({x-128.f,-128.f-x-width,z-128.f,-128.f-z-width,0.f});
    if (fine().on && spacing<2.f) {
        // (as without the switch out to the fixed range; beyond it, and under 0.5 m, inside the lens's wedge only)
        const float a=altitudeBlend(height);
        const float fixed=spacing>=.5f ? range(spacing)*(1.f-a)+(spacing>=1.f?32.f*a:0.f) : 0.f;
        return camera<fixed || (camera<cameraRange(spacing,height) && fineInView(x,z,width,eyeX,eyeZ)) || (spacing>=.5f && contact<range(spacing));
    }
    return camera<cameraRange(spacing,height) || contact<range(spacing);
}
inline float footprint(float x,float z,float eyeX,float eyeZ,float height) {
    auto base=[](float d){return d<=2048.f?std::max(.5f,d/64.f):(d<10000.f?32.f*std::pow(d/2048.f,std::log(2.f)/std::log(10000.f/2048.f)):d/156.25f);};
    float d=std::max(std::abs(x-eyeX),std::abs(z-eyeZ));
    const float a=altitudeBlend(height);
    float camera=base(d);
    if(a>0.f){
        float lo=.5f,prev=0.f;
        for(float s=.5f;s<=1024.f;s*=2.f){float r=cameraRange(s,height);if(d<=r){camera=prev>0.f ? lo*std::pow(std::max(d,prev)/prev,std::log(s/lo)/std::log(r/prev)) : lo*std::pow(s/lo,std::clamp(d/std::max(1e-6f,r),0.f,1.f));break;}lo=s;prev=r;}
    }
    return std::min(camera,base(std::max(0.f,std::max(std::abs(x),std::abs(z))-128.f)));
}
inline float distance(float x, float z, float eyeX, float eyeZ) {
    const float camera = std::max(std::abs(x-eyeX), std::abs(z-eyeZ));
    const float contact = std::max(0.f, std::max(std::abs(x), std::abs(z))-128.f);
    return std::min(camera, contact);
}
inline float boxDistance(float x, float z, float width, float eyeX, float eyeZ) {
    const float camera = std::max({x-eyeX, eyeX-x-width, z-eyeZ, eyeZ-z-width, 0.f});
    const float contact = std::max({x-128.f, -128.f-x-width, z-128.f, -128.f-z-width, 0.f});
    return std::min(camera, contact);
}
inline int& selectedMinLevel() { static int l=0; return l; }// the finest level the last select() kept
inline std::vector<Patch> select(float eyeX, float eyeZ, float height=0.f) {
    std::vector<Patch> out;
    int minLevel=fineMinLevel();
    auto visit = [&](auto&& self, float x, float z, int level) -> void {
        const float spacing = std::ldexp(.5f, level);
        const float width = cells * spacing;
        if (level>minLevel && refine(x,z,width,eyeX,eyeZ,.5f*spacing,height)) {
            const float h = width*.5f;
            self(self,x,z,level-1); self(self,x+h,z,level-1);
            self(self,x,z+h,level-1); self(self,x+h,z+h,level-1);
        } else out.push_back({x,z,spacing,1.f});
    };
    // A fixed 131 km square includes the curved horizon for all judging views.
    visit(visit,-65536.f,-65536.f,14);
    while (out.size() > slots && minLevel < 0) { ++minLevel; out.clear(); visit(visit,-65536.f,-65536.f,14); }
    selectedMinLevel()=minLevel;
    if (out.size() > slots) throw std::runtime_error("CDLOD patch budget exceeded");
    return out;
}
inline std::array<float,2> morph(float x, float z, float spacing, float eyeX, float eyeZ, float height=0.f) {
    const float a = std::clamp((normalizedDistance(x,z,eyeX,eyeZ,spacing,height)-.9f)/.1f,0.f,1.f);
    const float coarse = 2.f*spacing;
    return {x + a*(std::floor(x/coarse)*coarse-x), z + a*(std::floor(z/coarse)*coarse-z)};
}
inline Patch locate(float qx, float qz, float eyeX, float eyeZ, float height=0.f) {
    float x=-65536.f,z=-65536.f;
    int level=14;
    const int minLevel=fine().on ? selectedMinLevel() : 0;
    for (;;) {
        const float spacing=std::ldexp(.5f,level),width=cells*spacing;
        if (level<=minLevel || !refine(x,z,width,eyeX,eyeZ,.5f*spacing,height))
            return {x,z,spacing,1.f};
        const float half=.5f*width;
        if(qx>=x+half)x+=half;
        if(qz>=z+half)z+=half;
        --level;
    }
}
}
#endif
