// The pelican: its skin and bill, eyes, legs, wings and plumage, assembled on the bicycle.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_PELICAN_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_PELICAN_HPP

#include "Bicycle.hpp"

#include <chrono>
#include <iostream>
#include "Legs.hpp"
#include "PelicanBody.hpp"
#include "Plumage.hpp"
#include "Wings.hpp"

namespace pb {

    struct Pelican {
        PelicanBody anatomy;
        Plumage plumage;
        Wings wings;
        Leg legs[2];   // 0 right, 1 left
        std::shared_ptr<Group> root = Group::create();

        std::shared_ptr<MeshStandardMaterial> skinMat = standard(Color(1, 1, 1), 0.92f);
        std::shared_ptr<MeshPhysicalMaterial> billMat = physical(Color(1, 1, 1), 0.34f);
        std::shared_ptr<MeshStandardMaterial> pouchMat = standard(Color(1, 1, 1), 0.5f);
        std::shared_ptr<MeshStandardMaterial> eyeMat = standard(lin(12, 10, 10), 0.08f);
        std::shared_ptr<MeshStandardMaterial> irisMat = standard(lin(232, 226, 200), 0.2f);
        std::shared_ptr<MeshStandardMaterial> lidMat = standard(lin(238, 150, 50), 0.6f);
        std::shared_ptr<MeshStandardMaterial> legBall = standard(lin(230, 126, 44), 0.55f);
        std::shared_ptr<MeshStandardMaterial> thighWhite = standard(lin(247, 243, 234), 0.92f);
        std::shared_ptr<MeshStandardMaterial> legMat = standard(Color(1, 1, 1), 0.55f);
        std::shared_ptr<MeshStandardMaterial> footMat = standard(Color(1, 1, 1), 0.6f);
        std::shared_ptr<MeshStandardMaterial> webMat = standard(Color(1, 1, 1), 0.55f);

        struct Tex {
            TexSet contour, covert, primary, secondary, crest, tail;
        };
        std::unique_ptr<Tex> tex;

        Pelican() {
            skinMat->vertexColors = true;
            billMat->vertexColors = true;
            billMat->clearcoat = 0.35f;
            billMat->clearcoatRoughness = 0.25f;
            pouchMat->vertexColors = true;
            pouchMat->side = Side::Double;
            legMat->vertexColors = true;
            footMat->vertexColors = true;
            webMat->vertexColors = true;
            webMat->side = Side::Double;

            MeshBuild body;
            anatomy.buildBodyMesh(body);
            root->add(body.mesh(skinMat));

            MeshBuild bill;
            anatomy.buildUpperMesh(bill);
            anatomy.buildLowerMesh(bill);
            root->add(bill.mesh(billMat));

            MeshBuild pouch;
            anatomy.buildPouchMesh(pouch, true);
            root->add(pouch.mesh(pouchMat));

            irisMat->color = lin(206, 190, 150);
            lidMat->color = lin(238, 150, 50);
            for (int s : {1, -1}) {
                const Vector3 p = anatomy.eyePosition(s);
                Vector3 outward(0.25f, 0.12f, static_cast<float>(s));
                outward.normalize();
                // the eyeball sunk into the head, its dark pupil a little proud, a ring of bare skin round it
                root->add(ball(p - outward * 0.0012f, 0.0078f, irisMat, 20));
                root->add(ball(p + outward * 0.0016f, 0.0056f, eyeMat, 16));
                auto lid = Mesh::create(TorusGeometry::create(0.0088f, 0.0026f, 10, 28), lidMat);
                lid->quaternion.setFromUnitVectors(Vector3(0, 0, 1), outward);
                lid->position.copy(p - outward * 0.0008f);
                lid->scale.set(1.f, 0.9f, 0.7f);
                root->add(finish(lid));
            }

            for (int i = 0; i < 2; ++i) {
                legs[i].side = i == 0 ? 1.f : -1.f;
                legs[i].hip = Vector3(-0.03f, -0.03f, legs[i].side * 0.085f);   // deep in the belly: the thigh comes out through the plumage
                legs[i].build(legMat, footMat, webMat, legBall, thighWhite);
                root->add(legs[i].root);
            }

            // ---- plumage -----------------------------------------------------------------------------------------------
            const auto tPaint = std::chrono::steady_clock::now();
            tex = std::make_unique<Tex>();
            tex->contour = featherTexture(FeatherKind::Contour, 384, 512, 11);
            tex->covert = featherTexture(FeatherKind::Covert, 384, 768, 12);
            tex->primary = featherTexture(FeatherKind::Primary, 192, 1536, 13);
            tex->secondary = featherTexture(FeatherKind::Secondary, 256, 1280, 14);
            tex->crest = featherTexture(FeatherKind::Crest, 64, 512, 15);
            tex->tail = featherTexture(FeatherKind::Tail, 256, 768, 16);
            std::cout << "feather textures " << std::chrono::duration<double>(std::chrono::steady_clock::now() - tPaint).count() << " s" << std::endl;
            plumage.layContour(anatomy, 21);
            plumage.layTail(anatomy, 22);
            plumage.layCrest(anatomy, 23);
        }

        // set the bird on the bike (bike frame position, pitch about z, scale) and fit its arms to the grips
        void mount(const Bicycle& bike, const Vector3& position, float pitch, float scale) {
            root->position.copy(position);
            root->rotation.z = pitch;
            root->scale.set(scale, scale, scale);
            root->updateMatrixWorld(true);
            Matrix4 inv = *root->matrixWorld;
            inv.invert();
            // wrists rest on top of the grips
            const Vector3 gr = Vector3(bike.gripR).add(Vector3(0.f, 0.018f, 0.f)).applyMatrix4(inv);
            const Vector3 gl = Vector3(bike.gripL).add(Vector3(0.f, 0.018f, 0.f)).applyMatrix4(inv);
            wings.build(anatomy, gr, gl, 31);
            addPlumage();
        }

        void addPlumage() {
            const auto contourGeo = featherCard(0.16f, 0.5f);
            const auto flightGeo = featherCard(0.06f, 0.25f, 4, 16);
            const auto contourMat = featherMaterial(tex->contour);
            const auto covertMat = featherMaterial(tex->covert);
            const auto primMat = featherMaterial(tex->primary);
            const auto secMat = featherMaterial(tex->secondary);
            const auto crestMat = featherMaterial(tex->crest);
            const auto tailMat = featherMaterial(tex->tail);
            for (auto& leg : legs) leg.addFeathers(contourGeo, contourMat, 41u + static_cast<uint32_t>(leg.side > 0 ? 0 : 7));
            root->add(plumage.body.build(contourGeo, contourMat));
            root->add(plumage.neck.build(contourGeo, contourMat));
            root->add(plumage.head.build(contourGeo, contourMat));
            root->add(plumage.crest.build(contourGeo, crestMat));
            root->add(plumage.tail.build(flightGeo, tailMat));
            root->add(wings.primaries.build(flightGeo, primMat));
            root->add(wings.secondaries.build(flightGeo, secMat));
            root->add(wings.tertials.build(flightGeo, secMat));
            root->add(wings.covertsGreater.build(contourGeo, covertMat));
            root->add(wings.covertsMedian.build(contourGeo, contourMat));
            root->add(wings.covertsLesser.build(contourGeo, contourMat));
            root->add(wings.arms.mesh(skinMat));
            std::cout << "feathers: body " << plumage.body.m.size() << ", neck " << plumage.neck.m.size() << ", head " << plumage.head.m.size()
                      << ", wing " << wings.primaries.m.size() + wings.secondaries.m.size() + wings.tertials.m.size() + wings.covertsGreater.m.size() + wings.covertsMedian.m.size() + wings.covertsLesser.m.size() << std::endl;
        }

        // the legs, from where the pedals are (the bike's frame is carried into the bird's through the world)
        void pose(const Bicycle& bike, float phase) {
            root->updateMatrixWorld(true);
            Matrix4 inv = *root->matrixWorld;
            inv.invert();
            const Matrix4& bikeWorld = *bike.root->matrixWorld;
            for (int i = 0; i < 2; ++i) {
                const int side = i == 0 ? 1 : -1;
                const Vector3 pedal = bike.pedalCentre(side, phase);
                const Vector3 target = Vector3(pedal.x - 0.012f, pedal.y + 0.0125f + 0.018f, pedal.z).applyMatrix4(bikeWorld).applyMatrix4(inv);
                const float heel = 62.f * PI / 180.f;
                const auto dir = [&](const Vector3& v) { return Vector3(v).transformDirection(bikeWorld).transformDirection(inv); };
                legs[i].solve(target, dir(Vector3(-std::cos(heel), std::sin(heel), 0.f)), dir(Vector3(1.f, 0.f, 0.f)), dir(Vector3(0.f, 1.f, 0.f)));
            }
        }
    };

}// namespace pb

#endif
