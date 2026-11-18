// The weather of a scene, in physical units: one description that every
// effect derived from it reads.
//
//     auto weather = std::make_shared<Weather>();
//     weather->windSpeed = 9.6f;
//     weather->fetch = 8000.f;
//     weather->rain.push_back({{800.f, 0.f, -400.f}});
//     ocean->setWeather(*weather);// the ocean's spectrum
//     radar->weather = weather;   // the radar's sea clutter and rain
//
// Wind, fetch, swell and rain cells are authored here and nowhere else. The
// ocean and the radar both read it; neither reads the other. It is a pure
// function of scenario time: a moving rain cell is where its velocity has
// carried it at the time asked for.
#ifndef THREEPP_WEATHER_HPP
#define THREEPP_WEATHER_HPP

#include "threepp/math/Vector3.hpp"

#include <vector>

namespace threepp {

    /**
     * A rain shower: a column of rain from the sea up to `top`, with a
     * Gaussian rain-rate profile across (sigma = `radius`) peaking at
     * `rainRate` mm/h, moving at `velocity`.
     */
    struct RainCell {
        Vector3 centre;  // x, z at time 0; y unused
        Vector3 velocity;// m/s; x, z used
        float radius = 300.f;
        float top = 1500.f;  // world y
        float rainRate = 8.f;// mm/h at the centre

        // The centre at scenario time `time` (s): centre + velocity * time in x, z.
        [[nodiscard]] Vector3 centreAt(double time) const;
    };

    /**
     * The weather of a scene in physical units, a pure function of scenario
     * time. One instance is shared by everything that derives an effect from
     * it (the ocean's spectrum, the radar's sea clutter and rain).
     */
    struct Weather {
        // Wind at 10 m: speed in m/s, and the direction it blows TOWARD in the
        // world XZ plane, (cos windTheta, sin windTheta): Ocean's convention,
        // see DisplacedMesh::Params::windTheta.
        float windSpeed = 10.f;
        float windTheta = 0.6f;
        // Metres of open water upwind; 0 = a fully developed sea
        // (DisplacedMesh::Params::fetch).
        float fetch = 0.f;
        // Swell from a distant storm: its significant height (m), period (s)
        // and the direction it travels toward (radians, as windTheta). See
        // DisplacedMesh::Params::swellHeight.
        float swellHeight = 0.f;
        float swellPeriod = 12.f;
        float swellTheta = 0.f;
        std::vector<RainCell> rain;

        /**
         * Significant wave height (m) of the sea this wind, fetch and swell
         * raise. It is the analytic height of the spectrum the ocean draws
         * (phillips_spectrum.comp): Hs = 0.21 U^2 / g fully developed, and
         * with a fetch the shader's JONSWAP scaling of the length and level
         * plus the peak's share of the variance; below 4.5 m/s the shader's
         * wind-growth factor at the peak (0 under 1 m/s) and a light air's
         * centimetre scales (under half a millimetre); the swell adds in
         * quadrature.
         * It leaves out the ocean's short-wave cut-off, waveScale (1 =
         * physical) and a site's WaveClimate map, and it has not been
         * compared against the rendered height field.
         */
        [[nodiscard]] double significantWaveHeight() const;

        // Rain rate (mm/h) at the surface point (x, z) at `time`: the cells' sum.
        [[nodiscard]] double rainRateAt(double x, double z, double time) const;
    };

}// namespace threepp

#endif//THREEPP_WEATHER_HPP
