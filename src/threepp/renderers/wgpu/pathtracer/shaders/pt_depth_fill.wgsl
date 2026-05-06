
struct DepthFillUniforms {
    projView:   mat4x4<f32>,
    camOri:     vec4<f32>,
    camFwd:     vec4<f32>,
    camRgt:     vec4<f32>,
    camUp:      vec4<f32>,
    iRes:       vec4<f32>,
    tanHalfFov: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: DepthFillUniforms;
@group(0) @binding(1) var gBuf: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(vid & 1u) * 4.0 - 1.0;
    let y = f32((vid >> 1u) & 1u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) fpos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let px  = vec2<i32>(i32(fpos.x), i32(fpos.y));
    let t   = textureLoad(gBuf, px, 0).w;
    if (t <= 0.0) { return 1.0; }
    let ndc    = vec2<f32>((fpos.x / u.iRes.x) * 2.0 - 1.0,
                            1.0 - (fpos.y / u.iRes.y) * 2.0);
    let aspect = u.iRes.x / u.iRes.y;
    let rayDir = normalize(u.camFwd.xyz
                         + u.camRgt.xyz * (ndc.x * u.tanHalfFov.x * aspect)
                         + u.camUp.xyz  * (ndc.y * u.tanHalfFov.x));
    let worldPos = u.camOri.xyz + t * rayDir;
    let clip     = u.projView * vec4<f32>(worldPos, 1.0);
    if (clip.w <= 0.0) { return 1.0; }
    return clamp(clip.z / clip.w, 0.0, 1.0);
}
