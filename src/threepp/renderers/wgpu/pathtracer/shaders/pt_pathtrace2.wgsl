
fn pathTrace(ray_in: Ray,
             primaryHit: Hit,  // pre-computed by rt_primary_main; skip BVH for i==0
             seed: ptr<function, u32>,
             pixel: vec2<i32>,
             maxBounces:     i32,
             primaryMeshIdx: ptr<function, u32>,
             primaryNormal:  ptr<function, vec3<f32>>,
             primaryDepth:   ptr<function, f32>,
             primaryAlbedo:  ptr<function, vec3<f32>>,
             primaryRough:   ptr<function, f32>,
             primaryMatIdx:  ptr<function, i32>,
             primaryTriIdx:  ptr<function, i32>,
             touchedMoved:   ptr<function, bool>) -> SplitRadiance {
    *touchedMoved = false;
    let result = primaryShade(primaryHit, ray_in, seed, pixel, maxBounces,
                              primaryMeshIdx, primaryNormal, primaryDepth,
                              primaryAlbedo, primaryRough, primaryMatIdx, primaryTriIdx);
    if (!result.pathAlive) {
        return SplitRadiance(result.diffRad, result.specRad);
    }
    let bouncesResult = runBounces(result, seed, pixel, *primaryMeshIdx, touchedMoved);
    return SplitRadiance(result.diffRad + bouncesResult.diff,
                         result.specRad + bouncesResult.spec);
}
