// The watermill's shared surfaces, painted once at start-up (Textures.hpp) and worn by every part that is made of
// the same stuff. Colours given to a material multiply its painted map, so one grey board texture serves the
// silvered siding, the darker structural timber and the wet wood by the water.

#ifndef THREEPP_EXAMPLE_WATERMILL_MATERIALS_HPP
#define THREEPP_EXAMPLE_WATERMILL_MATERIALS_HPP

#include "threepp/materials/MeshStandardMaterial.hpp"

#include "Textures.hpp"

namespace mill {

    struct Materials {
        TexSet woodWeathered;// silvered outdoor boards
        TexSet woodWorn;     // indoor timber: browner, less eaten, dusty
        TexSet stone;        // dressed building stone
        TexSet rock;         // water-worn sandstone
        TexSet ground;       // forest floor
        TexSet rust;         // the wheel's iron
        TexSet glass;        // window panes
        TexSet lime;         // mortar and render

        std::shared_ptr<threepp::MeshStandardMaterial> siding;      // outer wall boards
        std::shared_ptr<threepp::MeshStandardMaterial> timber;      // posts, beams, the trestle
        std::shared_ptr<threepp::MeshStandardMaterial> timberWet;   // the flume, the wheel's buckets, wood near water
        std::shared_ptr<threepp::MeshStandardMaterial> trim;        // window frames and sashes (old white paint gone grey)
        std::shared_ptr<threepp::MeshStandardMaterial> shingles;    // roof shakes (vertex colours: moss)
        std::shared_ptr<threepp::MeshStandardMaterial> interior;    // inside timber
        std::shared_ptr<threepp::MeshStandardMaterial> floorBoards; // inside floors
        std::shared_ptr<threepp::MeshStandardMaterial> masonry;     // stones of walls and piers (vertex colours per stone)
        std::shared_ptr<threepp::MeshStandardMaterial> mortar;      // the lime between them
        std::shared_ptr<threepp::MeshStandardMaterial> rockMat;     // boulders and outcrops (vertex colours)
        std::shared_ptr<threepp::MeshStandardMaterial> groundMat;   // the terrain (vertex colours)
        std::shared_ptr<threepp::MeshStandardMaterial> iron;        // rusted iron
        std::shared_ptr<threepp::MeshStandardMaterial> blackIron;   // bolts, straps, bearings: dark, a little oily
        std::shared_ptr<threepp::MeshStandardMaterial> glassMat;    // grimy panes

        Materials() {
            using namespace threepp;
            woodWeathered = woodTexture(31, 0.7f, 1.f);
            woodWorn = woodTexture(47, 0.15f, 0.f);
            stone = stoneTexture(53);
            rock = rockTexture(9);
            ground = groundTexture(5);
            rust = rustTexture(61);
            glass = glassTexture(71);
            lime = limeTexture(83);

            const auto tinted = [](const TexSet& set, const V3& c, float normalScale = 1.f) {
                auto m = set.material(normalScale);
                m->color.setRGB(c.r, c.g, c.b);
                return m;
            };
            siding = tinted(woodWeathered, V3(1.f, 1.f, 1.f));
            timber = tinted(woodWeathered, V3(0.78f, 0.74f, 0.7f));
            timberWet = tinted(woodWeathered, V3(0.5f, 0.46f, 0.4f));
            trim = tinted(woodWeathered, V3(1.25f, 1.25f, 1.22f), 0.6f);
            shingles = tinted(woodWeathered, V3(0.62f, 0.6f, 0.58f), 1.3f);
            shingles->vertexColors = true;
            interior = tinted(woodWorn, V3(0.66f, 0.6f, 0.54f));// old timber indoors: dark, greyed, dusty
            floorBoards = tinted(woodWorn, V3(0.74f, 0.68f, 0.6f));
            // every made thing carries a colour per vertex (its piece's tint and its weathering)
            for (const auto& w : {siding, timber, timberWet, trim, interior, floorBoards}) w->vertexColors = true;
            masonry = tinted(stone, V3(1.f, 1.f, 1.f), 1.2f);
            masonry->vertexColors = true;
            mortar = tinted(lime, V3(1.f, 1.f, 1.f), 0.8f);
            mortar->vertexColors = true;
            rockMat = rock.material(1.f);
            rockMat->vertexColors = true;
            groundMat = ground.material(1.f);
            groundMat->vertexColors = true;
            iron = rust.material(1.f);
            iron->vertexColors = true;
            blackIron = MeshStandardMaterial::create();
            blackIron->color.setRGB(0.035f, 0.032f, 0.03f);
            blackIron->roughness = 0.55f;
            blackIron->metalness = 0.6f;
            glassMat = glass.material(1.f);
            glassMat->transparent = true;
            glassMat->depthWrite = false;
            glassMat->side = Side::Double;
            glassMat->metalness = 0.f;
            glassMat->metalnessMap = nullptr;
            glassMat->vertexColors = true;
        }
    };

}// namespace mill

#endif
