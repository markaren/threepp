
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/vegetation/CanopyForest.hpp"

#include <cmath>
#include <vector>

using namespace threepp;

// The detector's whole contract: one site per crown. A synthetic canopy height
// model with N well-separated Gaussian bumps must yield exactly N sites — not
// N·(cells per bump), which is what a naive "canopy > threshold" scan gives.
TEST_CASE("detectTreeSites finds one site per planted canopy peak", "[vegetation]") {
    constexpr int dim = 200;
    constexpr float world = 200.f;// ~1 m cells, like a 1 m DTM pack

    std::vector<float> chm(static_cast<size_t>(dim) * dim, 0.f);
    std::vector<float> dem(static_cast<size_t>(dim) * dim, 50.f);// flat, well above sea

    const int px[] = {30, 70, 110, 150};
    const int pz[] = {40, 90, 140};
    int planted = 0;
    for (int cz : pz)
        for (int cx : px) {
            ++planted;
            // sigma 2 cells: the bump falls below the 2.5 m gate ~3.3 cells out,
            // so no two bumps touch and each is the max of its own 5×5 window.
            for (int iz = cz - 8; iz <= cz + 8; ++iz)
                for (int ix = cx - 8; ix <= cx + 8; ++ix) {
                    const float d2 = static_cast<float>((ix - cx) * (ix - cx) + (iz - cz) * (iz - cz));
                    chm[static_cast<size_t>(iz) * dim + ix] += 10.f * std::exp(-d2 / 8.f);
                }
        }

    const terrain::HeightGrid canopy(chm, dim, world);
    const terrain::HeightGrid ground(dem, dim, world);

    const auto sites = vegetation::detectTreeSites(canopy, ground, {});

    REQUIRE(static_cast<int>(sites.size()) == planted);
}
