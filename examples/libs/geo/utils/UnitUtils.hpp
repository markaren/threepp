// https://github.com/tentone/geo-three/blob/master/source/utils/UnitsUtils.ts

#ifndef THREEPP_UNITUTILS_HPP
#define THREEPP_UNITUTILS_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector2.hpp"

#include <cmath>

namespace threepp::utils {

    /**
	 * Average radius of earth in meters.
	 */
    constexpr float EARTH_RADIUS = 6371008.f;

    /**
	 * Earth radius in semi-major axis A as defined in WGS84.
	 */
    const float EARTH_RADIUS_A = 6378137.0f;

    /**
	 * Earth radius in semi-minor axis B as defined in WGS84.
	 */
    const float EARTH_RADIUS_B = 6356752.314245f;

    /**
	 * Earth equator perimeter in meters.
	 */
    const float EARTH_PERIMETER = 2 * math::PI * EARTH_RADIUS;

    /**
	 * Earth equator perimeter in meters.
	 */
    const float EARTH_ORIGIN = EARTH_PERIMETER / 2.0f;

    inline Vector2 datumsToSpherical(float latitude, float longitude) {
        const auto x = longitude * EARTH_ORIGIN / 180.0f;
        auto y = std::log(std::tan((90 + latitude) * math::PI / 360.0f)) / (math::PI / 180.0f);

        y = y * EARTH_ORIGIN / 180.0f;

        return {x, y};
    }

}// namespace threepp::utils

#endif//THREEPP_UNITUTILS_HPP
