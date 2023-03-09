//

#ifndef THREEPP_EARCUT__HPP
#define THREEPP_EARCUT__HPP

#include <vector>
#include <optional>
#include <memory>
#include <algorithm>

namespace threepp::earcut {


    struct Node {


        // vertex index in coordinates array
        unsigned int i;

        // vertex coordinates
        float x;
        float y;

        // previous and next vertex nodes in a polygon ring
        std::shared_ptr<Node> prev;
        std::shared_ptr<Node> next;

        // z-order curve value
        std::optional<float> z;

        // previous and next nodes in z-order
        std::shared_ptr<Node> prevZ;
        std::shared_ptr<Node> nextZ;

        // indicates whether this is a steiner point
        bool steiner = false;

        Node(unsigned int i, float x, float y): i(i), x(x), y(y) {}

    };


    // eliminate colinear or duplicate points
    std::shared_ptr<Node> filterPoints( const std::shared_ptr<Node>& start, std::shared_ptr<Node>& end ) {

        if ( ! start ) return start;
        if ( ! end ) end = start;

        auto p = start;
        bool again;
        do {

            again = false;

            if ( ! p->steiner && ( equals( p, p->next ) || area( p->prev, p, p->next ) == 0 ) ) {

                removeNode( p );
                p = end = p->prev;
                if ( p == p->next ) break;
                again = true;

            } else {

                p = p->next;

            }

        } while ( again || p != end );

        return end;

    }

    // main ear slicing loop which triangulates a polygon (given as a linked list)
    function earcutLinked( ear, triangles, dim, minX, minY, invSize, pass ) {

        if ( ! ear ) return;

        // interlink polygon nodes in z-order
        if ( ! pass && invSize ) indexCurve( ear, minX, minY, invSize );

        let stop = ear,
            prev, next;

        // iterate through ears, slicing them one by one
        while ( ear.prev !== ear.next ) {

            prev = ear.prev;
            next = ear.next;

            if ( invSize ? isEarHashed( ear, minX, minY, invSize ) : isEar( ear ) ) {

                // cut off the triangle
                triangles.push( prev.i / dim );
                triangles.push( ear.i / dim );
                triangles.push( next.i / dim );

                removeNode( ear );

                // skipping the next vertex leads to less sliver triangles
                ear = next.next;
                stop = next.next;

                continue;

            }

            ear = next;

            // if we looped through the whole remaining polygon and can't find any more ears
            if ( ear == stop ) {

                // try filtering points and slicing again
                if ( ! pass ) {

                    earcutLinked( filterPoints( ear ), triangles, dim, minX, minY, invSize, 1 );

                    // if this didn't work, try curing all small self-intersections locally

                } else if ( pass == 1 ) {

                    ear = cureLocalIntersections( filterPoints( ear ), triangles, dim );
                    earcutLinked( ear, triangles, dim, minX, minY, invSize, 2 );

                    // as a last resort, try splitting the remaining polygon into two

                } else if ( pass == 2 ) {

                    splitEarcut( ear, triangles, dim, minX, minY, invSize );

                }

                break;

            }

        }

    }

    // check whether a polygon node forms a valid ear with adjacent nodes
    bool isEar( const std::shared_ptr<Node>& ear ) {

        const auto a = ear->prev,
              b = ear,
              c = ear->next;

        if ( area( a, b, c ) >= 0 ) return false; // reflex, can't be an ear

        // now make sure we don't have other points inside the potential ear
        auto p = ear->next->next;

        while ( p != ear->prev ) {

            if ( pointInTriangle( a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y ) &&
                area( p->prev, p, p->next ) >= 0 ) return false;
            p = p.next;

        }

        return true;

    }

    function isEarHashed( ear, minX, minY, invSize ) {

        const a = ear.prev,
              b = ear,
              c = ear.next;

        if ( area( a, b, c ) >= 0 ) return false; // reflex, can't be an ear

        // triangle bbox; min & max are calculated like this for speed
        const minTX = a.x < b.x ? ( a.x < c.x ? a.x : c.x ) : ( b.x < c.x ? b.x : c.x ),
              minTY = a.y < b.y ? ( a.y < c.y ? a.y : c.y ) : ( b.y < c.y ? b.y : c.y ),
              maxTX = a.x > b.x ? ( a.x > c.x ? a.x : c.x ) : ( b.x > c.x ? b.x : c.x ),
              maxTY = a.y > b.y ? ( a.y > c.y ? a.y : c.y ) : ( b.y > c.y ? b.y : c.y );

        // z-order range for the current triangle bbox;
        const minZ = zOrder( minTX, minTY, minX, minY, invSize ),
              maxZ = zOrder( maxTX, maxTY, minX, minY, invSize );

        let p = ear.prevZ,
            n = ear.nextZ;

        // look for points inside the triangle in both directions
        while ( p && p.z >= minZ && n && n.z <= maxZ ) {

            if ( p !== ear.prev && p !== ear.next &&
                                         pointInTriangle( a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y ) &&
                                         area( p.prev, p, p.next ) >= 0 ) return false;
            p = p.prevZ;

            if ( n !== ear.prev && n !== ear.next &&
                                         pointInTriangle( a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y ) &&
                                         area( n.prev, n, n.next ) >= 0 ) return false;
            n = n.nextZ;

        }

        // look for remaining points in decreasing z-order
        while ( p && p.z >= minZ ) {

            if ( p !== ear.prev && p !== ear.next &&
                                         pointInTriangle( a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y ) &&
                                         area( p.prev, p, p.next ) >= 0 ) return false;
            p = p.prevZ;

        }

        // look for remaining points in increasing z-order
        while ( n && n.z <= maxZ ) {

            if ( n !== ear.prev && n !== ear.next &&
                                         pointInTriangle( a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y ) &&
                                         area( n.prev, n, n.next ) >= 0 ) return false;
            n = n.nextZ;

        }

        return true;

    }

    // go through all polygon nodes and cure small local self-intersections
    function cureLocalIntersections( start, triangles, dim ) {

        let p = start;
        do {

            const a = p.prev,
                  b = p.next.next;

            if ( ! equals( a, b ) && intersects( a, p, p.next, b ) && locallyInside( a, b ) && locallyInside( b, a ) ) {

                triangles.push( a.i / dim );
                triangles.push( p.i / dim );
                triangles.push( b.i / dim );

                // remove two nodes involved
                removeNode( p );
                removeNode( p.next );

                p = start = b;

            }

            p = p.next;

        } while ( p !== start );

        return filterPoints( p );

    }

    // try splitting polygon into two and triangulate them independently
    function splitEarcut( start, triangles, dim, minX, minY, invSize ) {

        // look for a valid diagonal that divides the polygon into two
        let a = start;
        do {

            let b = a.next.next;
            while ( b !== a.prev ) {

                if ( a.i !== b.i && isValidDiagonal( a, b ) ) {

                    // split the polygon in two by the diagonal
                    let c = splitPolygon( a, b );

                    // filter colinear points around the cuts
                    a = filterPoints( a, a.next );
                    c = filterPoints( c, c.next );

                    // run earcut on each half
                    earcutLinked( a, triangles, dim, minX, minY, invSize );
                    earcutLinked( c, triangles, dim, minX, minY, invSize );
                    return;

                }

                b = b.next;

            }

            a = a.next;

        } while ( a !== start );

    }

    // link every hole into the outer loop, producing a single-ring polygon without holes
    function eliminateHoles( data, holeIndices, outerNode, dim ) {

        const queue = [];
        let i, len, start, end, list;

        for ( i = 0, len = holeIndices.length; i < len; i ++ ) {

            start = holeIndices[ i ] * dim;
            end = i < len - 1 ? holeIndices[ i + 1 ] * dim : data.length;
            list = linkedList( data, start, end, dim, false );
            if ( list === list.next ) list.steiner = true;
            queue.push( getLeftmost( list ) );

        }

        queue.sort( compareX );

        // process holes from left to right
        for ( i = 0; i < queue.length; i ++ ) {

            eliminateHole( queue[ i ], outerNode );
            outerNode = filterPoints( outerNode, outerNode.next );

        }

        return outerNode;

    }

    function compareX( a, b ) {

        return a.x - b.x;

    }

    // find a bridge between vertices that connects hole with an outer ring and and link it
    function eliminateHole( hole, outerNode ) {

        outerNode = findHoleBridge( hole, outerNode );
        if ( outerNode ) {

            const b = splitPolygon( outerNode, hole );

            // filter collinear points around the cuts
            filterPoints( outerNode, outerNode.next );
            filterPoints( b, b.next );

        }

    }

    // David Eberly's algorithm for finding a bridge between hole and outer polygon
    function findHoleBridge( hole, outerNode ) {

        let p = outerNode;
        const hx = hole.x;
        const hy = hole.y;
        let qx = - Infinity, m;

        // find a segment intersected by a ray from the hole's leftmost point to the left;
        // segment's endpoint with lesser x will be potential connection point
        do {

            if ( hy <= p.y && hy >= p.next.y && p.next.y !== p.y ) {

                const x = p.x + ( hy - p.y ) * ( p.next.x - p.x ) / ( p.next.y - p.y );
                if ( x <= hx && x > qx ) {

                    qx = x;
                    if ( x === hx ) {

                        if ( hy === p.y ) return p;
                        if ( hy === p.next.y ) return p.next;

                    }

                    m = p.x < p.next.x ? p : p.next;

                }

            }

            p = p.next;

        } while ( p !== outerNode );

        if ( ! m ) return null;

        if ( hx === qx ) return m; // hole touches outer segment; pick leftmost endpoint

        // look for points inside the triangle of hole point, segment intersection and endpoint;
        // if there are no points found, we have a valid connection;
        // otherwise choose the point of the minimum angle with the ray as connection point

        const stop = m,
              mx = m.x,
              my = m.y;
        let tanMin = Infinity, tan;

        p = m;

        do {

            if ( hx >= p.x && p.x >= mx && hx !== p.x &&
                                                pointInTriangle( hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p.x, p.y ) ) {

                tan = Math.abs( hy - p.y ) / ( hx - p.x ); // tangential

                if ( locallyInside( p, hole ) && ( tan < tanMin || ( tan === tanMin && ( p.x > m.x || ( p.x === m.x && sectorContainsSector( m, p ) ) ) ) ) ) {

                    m = p;
                    tanMin = tan;

                }

            }

            p = p.next;

        } while ( p !== stop );

        return m;

    }

    // whether sector in vertex m contains sector in vertex p in the same coordinates
    function sectorContainsSector( m, p ) {

        return area( m.prev, m, p.prev ) < 0 && area( p.next, m, m.next ) < 0;

    }

    // interlink polygon nodes in z-order
    function indexCurve( start, minX, minY, invSize ) {

        let p = start;
        do {

            if ( p.z === null ) p.z = zOrder( p.x, p.y, minX, minY, invSize );
            p.prevZ = p.prev;
            p.nextZ = p.next;
            p = p.next;

        } while ( p !== start );

        p.prevZ.nextZ = null;
        p.prevZ = null;

        sortLinked( p );

    }

    // Simon Tatham's linked list merge sort algorithm
    // http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
    std::shared_ptr<Node> sortLinked( std::shared_ptr<Node>& list ) {

        int i;
        std::shared_ptr<Node> p, q, e, tail;
        int numMerges, pSize, qSize,
                 inSize = 1;

        do {

            p = list;
            list = nullptr;
            tail = nullptr;
            numMerges = 0;

            while ( p ) {

                numMerges ++;
                q = p;
                pSize = 0;
                for ( i = 0; i < inSize; i ++ ) {

                    pSize ++;
                    q = q->nextZ;
                    if ( ! q ) break;

                }

                qSize = inSize;

                while ( pSize > 0 || ( qSize > 0 && q ) ) {

                    if ( pSize != 0 && ( qSize == 0 || ! q || p->z <= q->z ) ) {

                        e = p;
                        p = p->nextZ;
                        pSize --;

                    } else {

                        e = q;
                        q = q->nextZ;
                        qSize --;

                    }

                    if ( tail ) tail->nextZ = e;
                    else list = e;

                    e->prevZ = tail;
                    tail = e;

                }

                p = q;

            }

            tail->nextZ = nullptr;
            inSize *= 2;

        } while ( numMerges > 1 );

        return list;

    }

    // z-order of a point given coords and inverse of the longer side of data bbox
    float zOrder( float _x, float _y, float minX, float minY, float invSize ) {

        // coords are transformed into non-negative 15-bit integer range
        unsigned int x = 32767 * ( _x - minX ) * invSize;
        unsigned int y = 32767 * ( _y - minY ) * invSize;

        x = ( x | ( x << 8 ) ) & 0x00FF00FF;
        x = ( x | ( x << 4 ) ) & 0x0F0F0F0F;
        x = ( x | ( x << 2 ) ) & 0x33333333;
        x = ( x | ( x << 1 ) ) & 0x55555555;

        y = ( y | ( y << 8 ) ) & 0x00FF00FF;
        y = ( y | ( y << 4 ) ) & 0x0F0F0F0F;
        y = ( y | ( y << 2 ) ) & 0x33333333;
        y = ( y | ( y << 1 ) ) & 0x55555555;

        return static_cast<float>(x | ( y << 1 ));

    }

    // find the leftmost node of a polygon ring
    std::shared_ptr<Node> getLeftmost( const std::shared_ptr<Node>& start ) {

        auto p = start,
            leftmost = start;
        do {

            if ( p->x < leftmost->x || ( p->x == leftmost->x && p->y < leftmost->y ) ) leftmost = p;
            p = p->next;

        } while ( p != start );

        return leftmost;

    }

    // check if a point lies within a convex triangle
    bool pointInTriangle( float ax, float ay, float bx, float by, float cx, float cy, float px, float py ) {

        return ( cx - px ) * ( ay - py ) - ( ax - px ) * ( cy - py ) >= 0 &&
               ( ax - px ) * ( by - py ) - ( bx - px ) * ( ay - py ) >= 0 &&
               ( bx - px ) * ( cy - py ) - ( cx - px ) * ( by - py ) >= 0;

    }

    // check if a diagonal between two polygon nodes is valid (lies in polygon interior)
    bool isValidDiagonal( const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b ) {

        return a->next->i != b->i && a->prev->i != b->i && ! intersectsPolygon( a, b ) && // dones't intersect other edges
                                                  ( locallyInside( a, b ) && locallyInside( b, a ) && middleInside( a, b ) && // locally visible
                                                           ( area( a->prev, a, b->prev ) || area( a, b->prev, b ) ) || // does not create opposite-facing sectors
                                                   equals( a, b ) && area( a->prev, a, a->next ) > 0 && area( b->prev, b, b->next ) > 0 ); // special zero-length case

    }

    // signed area of a triangle
    float area( const std::shared_ptr<Node>& p, const std::shared_ptr<Node>& q, const std::shared_ptr<Node>& r ) {

        return ( q->y - p->y ) * ( r->x - q->x ) - ( q->x - p->x ) * ( r->y - q->y );

    }

    // check if two points are equal
    bool equals( Node& p1, Node& p2 ) {

        return p1.x == p2.x && p1.y == p2.y;

    }

    // check if two segments intersect
    bool intersects( const std::shared_ptr<Node>& p1, const std::shared_ptr<Node>& q1, const std::shared_ptr<Node>& p2, const std::shared_ptr<Node>& q2 ) {

        const auto  o1 = sign( area( p1, q1, p2 ) );
        const auto o2 = sign( area( p1, q1, q2 ) );
        const auto o3 = sign( area( p2, q2, p1 ) );
        const auto o4 = sign( area( p2, q2, q1 ) );

        if ( o1 != o2 && o3 != o4 ) return true; // general case

        if ( o1 == 0 && onSegment( p1, p2, q1 ) ) return true; // p1, q1 and p2 are collinear and p2 lies on p1q1
        if ( o2 == 0 && onSegment( p1, q2, q1 ) ) return true; // p1, q1 and q2 are collinear and q2 lies on p1q1
        if ( o3 == 0 && onSegment( p2, p1, q2 ) ) return true; // p2, q2 and p1 are collinear and p1 lies on p2q2
        if ( o4 == 0 && onSegment( p2, q1, q2 ) ) return true; // p2, q2 and q1 are collinear and q1 lies on p2q2

        return false;

    }

    // for collinear points p, q, r, check if point q lies on segment pr
    bool onSegment( const std::shared_ptr<Node>& p, const std::shared_ptr<Node>& q, const std::shared_ptr<Node>& r ) {

        return q->x <= std::max( p->x, r->x ) && q->x >= std::min( p->x, r->x ) && q->y <= std::max( p->y, r->y ) && q->y >= std::min( p->y, r->y );

    }

    int sign( float num ) {

        return num > 0 ? 1 : num < 0 ? - 1 : 0;

    }

    // check if a polygon diagonal intersects any polygon segments
    bool intersectsPolygon( const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b ) {

        auto p = a;
        do {

            if ( p->i != a->i && p->next->i != a->i && p->i != b->i && p->next->i != b->i &&
                                                                                intersects( p, p->next, a, b ) ) return true;
            p = p->next;

        } while ( p != a );

        return false;

    }

    // check if a polygon diagonal is locally inside the polygon
    bool locallyInside( const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b ) {

        return area( a->prev, a, a->next ) < 0 ?
                                           area( a, b, a->next ) >= 0 && area( a, a->prev, b ) >= 0 :
                                           area( a, b, a->prev ) < 0 || area( a, a->next, b ) < 0;

    }

    // check if the middle point of a polygon diagonal is inside the polygon
    bool middleInside( std::shared_ptr<Node>& a, std::shared_ptr<Node>& b ) {

        auto p = a;
            bool inside = false;
        const auto px = ( a->x + b->x ) / 2,
              py = ( a->y + b->y ) / 2;
        do {

            if ( ( ( p->y > py ) != ( p->next->y > py ) ) && p->next->y != p->y &&
                                                                  ( px < ( p->next->x - p->x ) * ( py - p->y ) / ( p->next->y - p->y ) + p->x ) )
                inside = ! inside;
            p = p->next;

        } while ( p != a );

        return inside;

    }

    // link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits polygon into two;
    // if one belongs to the outer ring and another to a hole, it merges it into a single ring
    std::shared_ptr<Node> splitPolygon( const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b ) {

        auto a2 = std::make_shared<Node>( a->i, a->x, a->y ),
              b2 = std::make_shared<Node>( b->i, b->x, b->y ),
              an = a->next,
              bp = b->prev;

        a->next = b;
        b->prev = a;

        a2->next = an;
        an->prev = a2;

        b2->next = a2;
        a2->prev = b2;

        bp->next = b2;
        b2->prev = bp;

        return b2;

    }

    // create a node and optionally link it with previous one (in a circular doubly linked list)
    std::shared_ptr<Node> insertNode( unsigned int i, float x, float y, const std::shared_ptr<Node>& last ) {

        auto p = std::make_shared<Node>( i, x, y );

        if ( ! last ) {

            p->prev = p;
            p->next = p;

        } else {

            p->next = last->next;
            p->prev = last;
            last->next->prev = p;
            last->next = p;

        }

        return p;

    }

    void removeNode( Node* p ) {

        p->next->prev = p->prev;
        p->prev->next = p->next;

        if ( p->prevZ ) p->prevZ->nextZ = p->nextZ;
        if ( p->nextZ ) p->nextZ->prevZ = p->prevZ;

    }


    float signedArea( const std::vector<float>& data, unsigned int start, unsigned int end, unsigned int dim ) {

        float sum = 0;
        for ( unsigned i = start, j = end - dim; i < end; i += dim ) {

            sum += ( data[ j ] - data[ i ] ) * ( data[ i + 1 ] + data[ j + 1 ] );
            j = i;

        }

        return sum;

    }

}

#endif//THREEPP_EARCUT__HPP
