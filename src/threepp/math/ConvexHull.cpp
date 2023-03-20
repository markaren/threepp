
#include "threepp/math/ConvexHull.hpp"

#include "threepp/math/Line3.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Triangle.hpp"

#include <limits>

using namespace threepp;

namespace {

    Plane _plane;
    Triangle _triangle;
    Line3 _line3;
    Vector3 _closestPoint;

    const int Visible = 0;
    const int Deleted = 1;


}// namespace

ConvexHull& ConvexHull::setFromPoints(const std::vector<Vector3>& points) {

    // The algorithm needs at least four points.

    if (points.size() >= 4) {

        this->makeEmpty();

        for (const auto& point : points) {

            this->vertices.emplace_back(point);
        }

        this->compute();
    }

    return *this;
}

bool ConvexHull::containsPoint(const Vector3& point) {

    for (const auto& face : faces) {

        // compute signed distance and check on what half space the point lies

        if (face->distanceToPoint(point) > this->tolerance) return false;
    }

    return true;
}

ConvexHull& ConvexHull::makeEmpty() {

    this->faces.clear();
    this->vertices.clear();

    return *this;
}

ConvexHull& ConvexHull::addVertexToFace(VertexNode* vertex, Face* face) {

    vertex->face = face;

    if (face->outside == nullptr) {

        this->assigned.append(vertex);

    } else {

        this->assigned.insertBefore(face->outside, vertex);
    }

    face->outside = vertex;

    return *this;
}

ConvexHull& ConvexHull::removeVertexFromFace(VertexNode* vertex, Face* face) {

    if (vertex == face->outside) {

        // fix face.outside link

        if (vertex->next != nullptr && vertex->next->face == face) {

            // face has at least 2 outside vertices, move the 'outside' reference

            face->outside = vertex->next;

        } else {

            // vertex was the only outside vertex that face had

            face->outside = nullptr;
        }
    }

    this->assigned.remove(vertex);

    return *this;
}

VertexNode* ConvexHull::removeAllVerticesFromFace(Face* face) {

    if (face->outside != nullptr) {

        // reference to the first and last vertex of this face

        const auto start = face->outside;
        auto end = face->outside;

        while (end->next != nullptr && end->next->face == face) {

            end = end->next;
        }

        this->assigned.removeSubList(start, end);

        // fix references

        start->prev = end->next = nullptr;
        face->outside = nullptr;

        return start;
    }

    return nullptr;
}

ConvexHull& ConvexHull::deleteFaceVertices(Face* face, Face* absorbingFace) {

    const auto faceVertices = this->removeAllVerticesFromFace(face);

    if (faceVertices != nullptr) {

        if (absorbingFace == nullptr) {

            // mark the vertices to be reassigned to some other face

            this->unassigned.appendChain(faceVertices);


        } else {

            // if there's an absorbing face try to assign as many vertices as possible to it

            auto vertex = faceVertices;

            do {

                // we need to buffer the subsequent vertex at this point because the 'vertex.next' reference
                // will be changed by upcoming method calls

                const auto nextVertex = vertex->next;

                const auto distance = absorbingFace->distanceToPoint(vertex->point);

                // check if 'vertex' is able to see 'absorbingFace'

                if (distance > this->tolerance) {

                    this->addVertexToFace(vertex, absorbingFace);

                } else {

                    this->unassigned.append(vertex);
                }

                // now assign next vertex

                vertex = nextVertex;

            } while (vertex != nullptr);
        }
    }

    return *this;
}

ConvexHull& ConvexHull::resolveUnassignedPoints(std::vector<std::shared_ptr<Face>>& newFaces) {

    if (!this->unassigned.isEmpty()) {

        auto vertex = this->unassigned.first();

        do {

            // buffer 'next' reference, see .deleteFaceVertices()

            const auto nextVertex = vertex->next;

            auto maxDistance = this->tolerance;

            Face* maxFace = nullptr;

            for (auto& face : newFaces) {

                if (face->mark == Visible) {

                    const auto distance = face->distanceToPoint(vertex->point);

                    if (distance > maxDistance) {

                        maxDistance = distance;
                        maxFace = face.get();
                    }

                    if (maxDistance > 1000 * this->tolerance) break;
                }
            }

            // 'maxFace' can be null e.g. if there are identical vertices

            if (maxFace != nullptr) {

                this->addVertexToFace(vertex, maxFace);
            }

            vertex = nextVertex;

        } while (vertex != nullptr);
    }

    return *this;
}

std::pair<std::vector<VertexNode*>, std::vector<VertexNode*>> ConvexHull::computeExtremes() {

    Vector3 min;
    Vector3 max;

    std::vector<VertexNode*> minVertices;
    std::vector<VertexNode*> maxVertices;

    // initially assume that the first vertex is the min/max

    for (unsigned i = 0; i < 3; i++) {

        VertexNode* v = &this->vertices[0];

        minVertices.emplace_back(v);
        maxVertices.emplace_back(v);
    }

    min.copy(this->vertices[0].point);
    max.copy(this->vertices[0].point);

    // compute the min/max vertex on all six directions

    for (auto& vertex : this->vertices) {

        auto& point = vertex.point;

        // update the min coordinates

        for (unsigned j = 0; j < 3; j++) {

            if (point[j] < min[j]) {

                min[j] = point[j];
                minVertices[j] = &vertex;
            }
        }

        // update the max coordinates

        for (unsigned j = 0; j < 3; j++) {

            if (point[j] > max[j]) {

                max[j] = point[j];
                maxVertices[j] = &vertex;
            }
        }
    }

    // use min/max vectors to compute an optimal epsilon

    this->tolerance = 3 * std::numeric_limits<float>::epsilon() * (std::max(std::abs(min.x), std::abs(max.x)) + std::max(std::abs(min.y), std::abs(max.y)) + std::max(std::abs(min.z), std::abs(max.z)));

    return {minVertices, maxVertices};
}

ConvexHull& ConvexHull::computeInitialHull() {

    const auto extremes = this->computeExtremes();
    const auto& min = extremes.first;
    const auto& max = extremes.second;

    // 1. Find the two vertices 'v0' and 'v1' with the greatest 1d separation
    // (max.x - min.x)
    // (max.y - min.y)
    // (max.z - min.z)

    float maxDistance = 0;
    unsigned int index = 0;

    for (unsigned i = 0; i < 3; i++) {

        const auto distance = max[i]->point[i] - min[i]->point[i];

        if (distance > maxDistance) {

            maxDistance = distance;
            index = i;
        }
    }

    const auto v0 = min[index];
    const auto v1 = max[index];
    VertexNode* v2;
    VertexNode* v3;

    // 2. The next vertex 'v2' is the one farthest to the line formed by 'v0' and 'v1'

    maxDistance = 0;
    _line3.set(v0->point, v1->point);

    for (unsigned i = 0, l = this->vertices.size(); i < l; i++) {

        auto& vertex = vertices[i];

        if (&vertex != v0 && &vertex != v1) {

            _line3.closestPointToPoint(vertex.point, true, _closestPoint);

            const auto distance = _closestPoint.distanceToSquared(vertex.point);

            if (distance > maxDistance) {

                maxDistance = distance;
                v2 = &vertex;
            }
        }
    }

    // 3. The next vertex 'v3' is the one farthest to the plane 'v0', 'v1', 'v2'

    maxDistance = -1;
    _plane.setFromCoplanarPoints(v0->point, v1->point, v2->point);

    for (unsigned i = 0, l = this->vertices.size(); i < l; i++) {

        auto& vertex = vertices[i];

        if (&vertex != v0 && &vertex != v1 && &vertex != v2) {

            const auto distance = std::abs(_plane.distanceToPoint(vertex.point));

            if (distance > maxDistance) {

                maxDistance = distance;
                v3 = &vertex;
            }
        }
    }

    std::vector<std::shared_ptr<Face>> faces;

    if (_plane.distanceToPoint(v3->point) < 0) {

        // the face is not able to see the point so 'plane.normal' is pointing outside the tetrahedron

        faces.emplace_back(Face::create(v0, v1, v2));
        faces.emplace_back(Face::create(v3, v1, v0));
        faces.emplace_back(Face::create(v3, v2, v1));
        faces.emplace_back(Face::create(v3, v0, v2));

        // set the twin edge

        for (unsigned i = 0; i < 3; i++) {

            const auto j = (i + 1) % 3;

            // join face[ i ] i > 0, with the first face

            faces[i + 1]->getEdge(2)->setTwin(faces[0]->getEdge(j));

            // join face[ i ] with face[ i + 1 ], 1 <= i <= 3

            faces[i + 1]->getEdge(1)->setTwin(faces[j + 1]->getEdge(0));
        }

    } else {

        // the face is able to see the point so 'plane.normal' is pointing inside the tetrahedron

        faces.emplace_back(Face::create(v0, v2, v1));
        faces.emplace_back(Face::create(v3, v0, v1));
        faces.emplace_back(Face::create(v3, v1, v2));
        faces.emplace_back(Face::create(v3, v2, v0));

        // set the twin edge

        for (unsigned i = 0; i < 3; i++) {

            const auto j = (i + 1) % 3;

            // join face[ i ] i > 0, with the first face

            faces[i + 1]->getEdge(2)->setTwin(faces[0]->getEdge((3 - i) % 3));

            // join face[ i ] with face[ i + 1 ]

            faces[i + 1]->getEdge(0)->setTwin(faces[j + 1]->getEdge(1));
        }
    }

    // the initial hull is the tetrahedron

    for (unsigned i = 0; i < 4; i++) {

        this->faces.emplace_back(faces[i]);
    }

    // initial assignment of vertices to the faces of the tetrahedron

    for (auto& vertex : vertices) {

        if (&vertex != v0 && &vertex != v1 && &vertex != v2 && &vertex != v3) {

            maxDistance = this->tolerance;
            Face* maxFace = nullptr;

            for (unsigned j = 0; j < 4; j++) {

                const auto distance = this->faces[j]->distanceToPoint(vertex.point);

                if (distance > maxDistance) {

                    maxDistance = distance;
                    maxFace = this->faces[j].get();
                }
            }

            if (maxFace != nullptr) {

                this->addVertexToFace(&vertex, maxFace);
            }
        }
    }

    return *this;
}

ConvexHull& ConvexHull::reindexFaces() {

    std::vector<std::shared_ptr<Face>> activeFaces;

    for (auto& face : this->faces) {

        if (face->mark == Visible) {

            activeFaces.emplace_back(face);
        }
    }

    this->faces = activeFaces;

    return *this;
}

VertexNode* ConvexHull::nextVertexToAdd() {

    // if the 'assigned' list of vertices is empty, no vertices are left. return with 'undefined'

    if (!this->assigned.isEmpty()) {

        VertexNode* eyeVertex;
        float maxDistance = 0;

        // grap the first available face and start with the first visible vertex of that face

        const auto eyeFace = this->assigned.first()->face;
        auto vertex = eyeFace->outside;

        // now calculate the farthest vertex that face can see

        do {

            const auto distance = eyeFace->distanceToPoint(vertex->point);

            if (distance > maxDistance) {

                maxDistance = distance;
                eyeVertex = vertex;
            }

            vertex = vertex->next;

        } while (vertex != nullptr && vertex->face == eyeFace);

        return eyeVertex;
    }

    return nullptr;
}

ConvexHull& ConvexHull::computeHorizon(const Vector3& eyePoint, HalfEdge* crossEdge, Face* face, std::vector<std::shared_ptr<HalfEdge>>& horizon) {

    // moves face's vertices to the 'unassigned' vertex list

    this->deleteFaceVertices(face, nullptr);

    face->mark = Deleted;

    std::shared_ptr<HalfEdge> edge;

    if (crossEdge == nullptr) {

        edge = face->getEdge(0);
        crossEdge = face->getEdge(0).get();

    } else {

        // start from the next edge since 'crossEdge' was already analyzed
        // (actually 'crossEdge.twin' was the edge who called this method recursively)

        edge = crossEdge->next;
    }

    do {

        auto twinEdge = edge->twin;
        const auto oppositeFace = twinEdge->face;

        if (oppositeFace->mark == Visible) {

            if (oppositeFace->distanceToPoint(eyePoint) > this->tolerance) {

                // the opposite face can see the vertex, so proceed with next edge

                this->computeHorizon(eyePoint, twinEdge.get(), oppositeFace, horizon);

            } else {

                // the opposite face can't see the vertex, so this edge is part of the horizon

                horizon.emplace_back(edge);
            }
        }

        edge = edge->next;

    } while (edge.get() != crossEdge);

    return *this;
}

std::shared_ptr<HalfEdge> ConvexHull::addAdjoiningFace(VertexNode* eyeVertex, const std::shared_ptr<HalfEdge>& horizonEdge) {

    // all the half edges are created in ccw order thus the face is always pointing outside the hull

    auto face = Face::create(eyeVertex, horizonEdge->tail(), horizonEdge->head());

    this->faces.emplace_back(face);

    // join face.getEdge( - 1 ) with the horizon's opposite edge face.getEdge( - 1 ) = face.getEdge( 2 )

    face->getEdge(-1)->setTwin(horizonEdge->twin);

    return face->getEdge(0);// the half edge whose vertex is the eyeVertex
}

ConvexHull& ConvexHull::addNewFaces(VertexNode* eyeVertex, std::vector<std::shared_ptr<HalfEdge>>& horizon) {

    this->newFaces.clear();

    std::shared_ptr<HalfEdge> firstSideEdge = nullptr;
    std::shared_ptr<HalfEdge> previousSideEdge = nullptr;

    for (auto& horizonEdge : horizon) {

        // returns the right side edge

        const auto sideEdge = this->addAdjoiningFace(eyeVertex, horizonEdge);

        if (firstSideEdge == nullptr) {

            firstSideEdge = sideEdge;

        } else {

            // joins face.getEdge( 1 ) with previousFace.getEdge( 0 )

            sideEdge->next->setTwin(previousSideEdge);
        }

        this->newFaces.emplace_back(sideEdge->face);
        previousSideEdge = sideEdge;
    }

    // perform final join of new faces

    firstSideEdge->next->setTwin(previousSideEdge);

    return *this;
}

ConvexHull& ConvexHull::addVertexToHull(VertexNode* eyeVertex) {

    std::vector<std::shared_ptr<HalfEdge>> horizon;

    this->unassigned.clear();

    // remove 'eyeVertex' from 'eyeVertex.face' so that it can't be added to the 'unassigned' vertex list

    this->removeVertexFromFace(eyeVertex, eyeVertex->face);

    this->computeHorizon(eyeVertex->point, nullptr, eyeVertex->face, horizon);

    this->addNewFaces(eyeVertex, horizon);

    // reassign 'unassigned' vertices to the new faces

    this->resolveUnassignedPoints(this->newFaces);

    return *this;
}

ConvexHull& ConvexHull::cleanup() {

    this->assigned.clear();
    this->unassigned.clear();
    this->newFaces.clear();

    return *this;

}

ConvexHull& ConvexHull::compute() {

    VertexNode* vertex;

    this->computeInitialHull();

    // add all available vertices gradually to the hull

    while ((vertex = this->nextVertexToAdd()) != nullptr) {

        this->addVertexToHull(vertex);
    }

    this->reindexFaces();

    this->cleanup();

    return *this;
}


VertexNode* VertexList::first() const {

    return this->head;
}

VertexNode* VertexList::last() const {

    return this->tail;
}

VertexList& VertexList::clear() {

    this->head = this->tail = nullptr;

    return *this;
}

VertexList& VertexList::insertBefore(VertexNode* target, VertexNode* vertex) {

    vertex->prev = target->prev;
    vertex->next = target;

    if (vertex->prev == nullptr) {

        this->head = vertex;

    } else {

        vertex->prev->next = vertex;
    }

    target->prev = vertex;

    return *this;
}

VertexList& VertexList::insertAfter(VertexNode* target, VertexNode* vertex) {

    vertex->prev = target;
    vertex->next = target->next;

    if (vertex->next == nullptr) {

        this->tail = vertex;

    } else {

        vertex->next->prev = vertex;
    }

    target->next = vertex;

    return *this;
}

VertexList& VertexList::append(VertexNode* vertex) {

    if (this->head == nullptr) {

        this->head = vertex;

    } else {

        this->tail->next = vertex;
    }

    vertex->prev = this->tail;
    vertex->next = nullptr;// the tail has no subsequent vertex

    this->tail = vertex;

    return *this;
}

VertexList& VertexList::appendChain(VertexNode* vertex) {

    if (this->head == nullptr) {

        this->head = vertex;

    } else {

        this->tail->next = vertex;
    }

    vertex->prev = this->tail;

    // ensure that the 'tail' reference points to the last vertex of the chain

    while (vertex->next != nullptr) {

        vertex = vertex->next;
    }

    this->tail = vertex;

    return *this;
}

VertexList& VertexList::remove(VertexNode* vertex) {

    if (vertex->prev == nullptr) {

        this->head = vertex->next;

    } else {

        vertex->prev->next = vertex->next;
    }

    if (vertex->next == nullptr) {

        this->tail = vertex->prev;

    } else {

        vertex->next->prev = vertex->prev;
    }

    return *this;
}

VertexList& VertexList::removeSubList(VertexNode* a, VertexNode* b) {

    if (a->prev == nullptr) {

        this->head = b->next;

    } else {

        a->prev->next = b->next;
    }

    if (b->next == nullptr) {

        this->tail = a->prev;

    } else {

        b->next->prev = a->prev;
    }

    return *this;
}

bool VertexList::isEmpty() const {

    return this->head == nullptr;
}

Face::Face(): mark(Visible) {}

std::shared_ptr<Face> Face::create(VertexNode* a, VertexNode* b, VertexNode* c) {

    auto face = std::shared_ptr<Face>(new Face());

    const auto e0 = std::make_shared<HalfEdge>(a, face.get());
    const auto e1 = std::make_shared<HalfEdge>(b, face.get());
    const auto e2 = std::make_shared<HalfEdge>(c, face.get());

    // join edges

    e0->next = e2->prev = e1;
    e1->next = e0->prev = e2;
    e2->next = e1->prev = e0;

    // main half edge reference

    face->edge = e0;

    face->compute();

    return face;
}

std::shared_ptr<HalfEdge> Face::getEdge(unsigned int i) {

    while (i > 0) {

        edge = edge->next;
        i--;
    }

    while (i < 0) {

        edge = edge->prev;
        i++;
    }

    return edge;
}

void Face::compute() {
    const auto a = this->edge->tail();
    const auto b = this->edge->head();
    const auto c = this->edge->next->head();

    _triangle.set(a->point, b->point, c->point);

    _triangle.getNormal(this->normal);
    _triangle.getMidpoint(this->midpoint);
    this->area = _triangle.getArea();

    this->constant = this->normal.dot(this->midpoint);
}

float Face::distanceToPoint(const Vector3& point) const {

    return this->normal.dot(point) - this->constant;
}

HalfEdge::HalfEdge(VertexNode* vertex, Face* face): vertex(vertex), face(face) {}

VertexNode* HalfEdge::head() const {

    return this->vertex;
}
VertexNode* HalfEdge::tail() const {

    return this->prev ? this->prev->vertex : nullptr;
}
float HalfEdge::length() const {

    const auto head = this->head();
    const auto tail = this->tail();

    if (tail != nullptr) {

        return tail->point.distanceTo(head->point);
    }

    return -1;
}
float HalfEdge::lengthSquared() const {

    const auto head = this->head();
    const auto tail = this->tail();

    if (tail != nullptr) {

        return tail->point.distanceToSquared(head->point);
    }

    return -1;
}

HalfEdge& HalfEdge::setTwin(const std::shared_ptr<HalfEdge>& edge) {

    this->twin = edge;
    edge->twin = shared_from_this();

    return *this;
}
