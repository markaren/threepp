
struct UpscaleUniforms {
    prevCamOri: vec4<f32>,
    prevCamFwd: vec4<f32>,
    prevCamRgt: vec4<f32>,
    prevCamUp:  vec4<f32>,
    curCamOri:  vec4<f32>,
    curCamFwd:  vec4<f32>,
    curCamRgt:  vec4<f32>,
    curCamUp:   vec4<f32>,
    iRes:       vec4<f32>,   // [0]=fullW [1]=fullH [2]=pixelScale [3]=0
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,
};
@group(0) @binding(0) var<uniform> up: UpscaleUniforms;
@group(0) @binding(1) var denoisedDiff: texture_2d<f32>;
@group(0) @binding(2) var denoisedSpec: texture_2d<f32>;
@group(0) @binding(3) var gBufCurLow: texture_2d<f32>;  // normal.xyz + rayDist.w
@group(0) @binding(4) var historyIn:  texture_2d<f32>;  // previous full-res output
@group(0) @binding(5) var historyOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn upscale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fullPx  = vec2<i32>(i32(gid.x), i32(gid.y));
    let fullW   = up.iRes.x;
    let fullH   = up.iRes.y;
    if (f32(fullPx.x) >= fullW || f32(fullPx.y) >= fullH) { return; }

    let pixScale   = up.iRes.z;
    let lowResSize = vec2<i32>(textureDimensions(denoisedDiff));

    // Map full-res pixel to low-res accum pixel
    let lowResPx = clamp(
        vec2<i32>(vec2<f32>(fullPx) * pixScale),
        vec2<i32>(0),
        lowResSize - vec2<i32>(1)
    );

    // Current denoised color at low-res pixel
    let curColor = textureLoad(denoisedDiff, lowResPx, 0).xyz
                 + textureLoad(denoisedSpec, lowResPx, 0).xyz;

    // G-buffer: normal.xyz + ray distance.w
    let gBuf     = textureLoad(gBufCurLow, lowResPx, 0);
    let curDepth = gBuf.w;

    // Sky pixel — no temporal history possible
    if (curDepth < 1e-6) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, -1.0));
        return;
    }

    // Reconstruct world position from low-res pixel-center ray + ray distance
    let aspect  = fullW / fullH;
    let tanHfov = up.tanHalfFov.x;
    let lowNdc  = vec2<f32>(
        (f32(lowResPx.x) + 0.5) / f32(lowResSize.x) * 2.0 - 1.0,
        1.0 - (f32(lowResPx.y) + 0.5) / f32(lowResSize.y) * 2.0
    );
    let rayDir  = normalize(up.curCamFwd.xyz
                          + up.curCamRgt.xyz * (lowNdc.x * tanHfov * aspect)
                          + up.curCamUp.xyz  * (lowNdc.y * tanHfov));
    let worldPos = up.curCamOri.xyz + rayDir * curDepth;

    // Reproject world position to previous frame's full-res screen
    let relP  = worldPos - up.prevCamOri.xyz;
    let prevZ = dot(relP, up.prevCamFwd.xyz);
    if (prevZ <= 0.001) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, 1.0));
        return;
    }

    let prevNdcX = dot(relP, up.prevCamRgt.xyz) / (prevZ * tanHfov * aspect);
    let prevNdcY = dot(relP, up.prevCamUp.xyz)  / (prevZ * tanHfov);
    let prevU    = (prevNdcX + 1.0) * 0.5 * fullW - 0.5;
    let prevV    = (1.0 - prevNdcY) * 0.5 * fullH - 0.5;
    let prevFullPx = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));

    if (prevFullPx.x < 0 || prevFullPx.y < 0 ||
        prevFullPx.x >= i32(fullW) || prevFullPx.y >= i32(fullH)) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, 1.0));
        return;
    }

    // Fetch full-res history (.w = histLen; -1 = sky sentinel; 0 = fresh/reset)
    let histSamp  = textureLoad(historyIn, prevFullPx, 0);
    let histColor = histSamp.xyz;
    let histLen   = histSamp.w;

    // Disocclusion: only reject sky-sentinel history (depth cross-frame comparison
    // is unreliable for moving cameras — let short max-history handle ghosting).
    var result     = curColor;
    var newHistLen = 1.0;
    if (histLen >= 0.0) {
        newHistLen = min(histLen + 1.0, 32.0);
        let alpha  = max(1.0 / 16.0, 1.0 / newHistLen);
        result = mix(histColor, curColor, alpha);
    }

    textureStore(historyOut, fullPx, vec4<f32>(result, newHistLen));
}
