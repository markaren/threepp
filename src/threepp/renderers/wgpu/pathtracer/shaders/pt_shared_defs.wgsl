
const TRI_PAGE_W:  i32 = 8192;
const TRI_PAGE_H:  i32 = 8;
const MAX_LEAF_TRIS: i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}
