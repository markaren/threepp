#pragma once
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string_view>

namespace threepp {
// Uniform current. Direction TO, degrees from simulation +X toward +Y; scene
// coordinates are (X,Z)=(x,-y). Physics opts in separately via THREEPP_OCEAN_CURRENT_PHYS.
struct UniformCurrent {
    bool enabled = false;
    float x = 0, z = 0;
    static UniformCurrent parse(const char* text) {
        if (!text) return {};
        const std::string_view s(text);
        const auto comma=s.find(',');
        if(comma==s.npos || comma==0 || comma+1==s.size()) return {};
        auto number=[](std::string_view token,double& v) {
            const auto r=std::from_chars(token.data(),token.data()+token.size(),v);
            return r.ec==std::errc{} && r.ptr==token.data()+token.size() && std::isfinite(v);
        };
        double speed,angle;
        if(!number(s.substr(0,comma),speed) || !number(s.substr(comma+1),angle) || speed<0 || speed>100) return {};
        // Bounded numerical prototype; 100 m/s is a parser safety limit, not
        // a physically validated sea/current range. Zero is exactly OFF.
        if(speed==0) return {};
        const double a=std::remainder(angle,360.)*3.14159265358979323846/180.;
        return {true,float(speed*std::cos(a)),float(-speed*std::sin(a))};
    }
    static const UniformCurrent& environment() {
        static const auto u=parse(std::getenv("THREEPP_OCEAN_CURRENT"));
        return u;
    }
};
}
