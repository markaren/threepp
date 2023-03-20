
#include "threepp/math/ConvexHull.hpp"

#include "threepp/math/Vector3.hpp"
#include "threepp/core/Face3.hpp"
#include "threepp/math/Triangle.hpp"

using namespace threepp;

namespace {

    Triangle _triangle;

}

ConvexHull& ConvexHull::setFromPoints(const std::vector<Vector3>& points) {

    // The algorithm needs at least four points.

    if ( points.size() >= 4 ) {

        this->makeEmpty();

        for ( unsigned i = 0, l = points.size(); i < l; i ++ ) {

            this->vertices.emplace_back( new VertexNode( points[ i ] ) );

        }

        this->compute();

    }

    return *this;

}

ConvexHull& ConvexHull::makeEmpty() {

    this->faces.clear();
    this->vertices.clear();

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

    if ( vertex->prev == nullptr ) {

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

    if ( vertex->next == nullptr ) {

        this->tail = vertex;

    } else {

        vertex->next->prev = vertex;

    }

    target->next = vertex;

    return *this;

}

VertexList& VertexList::append(VertexNode* vertex) {

    if ( this->head == nullptr ) {

        this->head = vertex;

    } else {

        this->tail->next = vertex;

    }

    vertex->prev = this->tail;
    vertex->next = nullptr; // the tail has no subsequent vertex

    this->tail = vertex;

    return *this;

}

VertexList& VertexList::appendChain(VertexNode* vertex) {

    if ( this->head == nullptr ) {

        this->head = vertex;

    } else {

        this->tail->next = vertex;

    }

    vertex->prev = this->tail;

    // ensure that the 'tail' reference points to the last vertex of the chain

    while ( vertex->next != nullptr ) {

        vertex = vertex->next;

    }

    this->tail = vertex;

    return *this;

}

VertexList& VertexList::remove(VertexNode* vertex) {

    if ( vertex->prev == nullptr ) {

        this->head = vertex->next;

    } else {

        vertex->prev->next = vertex->next;

    }

    if ( vertex->next == nullptr ) {

        this->tail = vertex->prev;

    } else {

        vertex->next->prev = vertex->prev;

    }

    return *this;

}

VertexList& VertexList::removeSubList(VertexNode* a, VertexNode* b) {

    if ( a->prev == nullptr ) {

        this->head = b->next;

    } else {

        a->prev->next = b->next;

    }

    if ( b->next == nullptr ) {

        this->tail = a->prev;

    } else {

        b->next->prev = a->prev;

    }

    return *this;

}

bool VertexList::isEmpty() const {

    return this->head == nullptr;

}

std::shared_ptr<Face> Face::create(VertexNode* a, VertexNode* b, VertexNode* c) {

    auto face = std::shared_ptr<Face>(new Face());

    const auto e0 = std::make_shared<HalfEdge>( a, face );
    const auto e1 = std::make_shared<HalfEdge>( b, face );
    const auto e2 = std::make_shared<HalfEdge>( c, face );

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

    while ( i > 0 ) {

        edge = edge->next;
        i --;

    }

    while ( i < 0 ) {

        edge = edge->prev;
        i ++;

    }

    return edge;

}

void Face::compute() {
    const auto a = this->edge->tail();
    const auto b = this->edge->head();
    const auto c = this->edge->next->head();

    _triangle.set( a->point, b->point, c->point );

    _triangle.getNormal( this->normal );
    _triangle.getMidpoint( this->midpoint );
    this->area = _triangle.getArea();

    this->constant = this->normal.dot( this->midpoint );
}

float Face::distanceToPoint(const Vector3& point) const {

    return this->normal.dot( point ) - this->constant;

}

HalfEdge::HalfEdge(VertexNode* vertex, std::shared_ptr<Face> face): vertex(vertex), face(std::move(face)) {}

VertexNode* HalfEdge::head() const {

    return this->vertex;

}
VertexNode* HalfEdge::tail() const {

    return this->prev ? this->prev->vertex : nullptr;

}
float HalfEdge::length() const {

    const auto head = this->head();
    const auto tail = this->tail();

    if ( tail != nullptr ) {

        return tail->point.distanceTo( head->point );

    }

    return - 1;

}
float HalfEdge::lengthSquared() const {

    const auto head = this->head();
    const auto tail = this->tail();

    if ( tail != nullptr ) {

        return tail->point.distanceToSquared( head->point );

    }

    return - 1;

}

HalfEdge& HalfEdge::setTwin(const std::shared_ptr<HalfEdge>& edge) {

    this->twin = edge.get();
    edge->twin = this;

    return *this;

}
