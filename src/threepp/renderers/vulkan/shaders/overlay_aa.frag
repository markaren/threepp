#version 460

// Masked edge anti-aliasing for the hybrid raster overlay (vector content:
// SVG/UI fills, lines, wireframes, points). The overlay rasterizes post-TAA
// straight onto the 1-sample swapchain, so nothing antialiases its edges —
// and MSAA'ing the pass would 4x the depth prepass + overlay raster. Instead
// the overlay writes a 1-byte coverage mask (attachment 1) as it draws, and
// this fullscreen pass runs an FXAA-style edge blend ONLY on pixels the
// overlay touched (dilated by 1 px so both sides of a silhouette edge are
// treated). Everything else discards, leaving the TAA-resolved scene
// untouched. srcTex is a copy of the post-overlay swapchain (a render pass
// can't sample the attachment it writes), display-referred sRGB — exactly
// the space FXAA's luma heuristics were designed for.

layout(binding = 0) uniform sampler2D srcTex; // post-overlay swapchain copy
layout(binding = 1) uniform sampler2D maskTex;// R8 overlay coverage

layout(location = 0) out vec4 outColor;

float lumaOf(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    const vec2 rcp = 1.0 / vec2(textureSize(srcTex, 0));
    const vec2 uv  = gl_FragCoord.xy * rcp;

    // Coverage test, dilated one pixel (CLAMP_TO_EDGE sampler handles the
    // borders). An overlay silhouette edge has coverage on one side only;
    // the outside pixel needs blending too.
    float m = textureLod(maskTex, uv, 0.0).r;
    m += textureLod(maskTex, uv + vec2( rcp.x, 0.0), 0.0).r;
    m += textureLod(maskTex, uv + vec2(-rcp.x, 0.0), 0.0).r;
    m += textureLod(maskTex, uv + vec2(0.0,  rcp.y), 0.0).r;
    m += textureLod(maskTex, uv + vec2(0.0, -rcp.y), 0.0).r;
    if (m <= 0.0) discard;

    // FXAA 3.11 (quality preset, condensed). Standard Lottes algorithm:
    // local-contrast gate, edge orientation from Sobel-ish luma taps,
    // end-of-edge search along the edge, sub-pixel blend fallback.
    const vec3 rgbM = textureLod(srcTex, uv, 0.0).rgb;
    const float lM  = lumaOf(rgbM);
    const float lN  = lumaOf(textureLod(srcTex, uv + vec2(0.0, -rcp.y), 0.0).rgb);
    const float lS  = lumaOf(textureLod(srcTex, uv + vec2(0.0,  rcp.y), 0.0).rgb);
    const float lE  = lumaOf(textureLod(srcTex, uv + vec2( rcp.x, 0.0), 0.0).rgb);
    const float lW  = lumaOf(textureLod(srcTex, uv + vec2(-rcp.x, 0.0), 0.0).rgb);

    const float lMin = min(lM, min(min(lN, lS), min(lE, lW)));
    const float lMax = max(lM, max(max(lN, lS), max(lE, lW)));
    const float range = lMax - lMin;
    // Below-threshold contrast: nothing to smooth.
    if (range < max(0.0312, lMax * 0.125)) discard;

    const float lNW = lumaOf(textureLod(srcTex, uv + vec2(-rcp.x, -rcp.y), 0.0).rgb);
    const float lNE = lumaOf(textureLod(srcTex, uv + vec2( rcp.x, -rcp.y), 0.0).rgb);
    const float lSW = lumaOf(textureLod(srcTex, uv + vec2(-rcp.x,  rcp.y), 0.0).rgb);
    const float lSE = lumaOf(textureLod(srcTex, uv + vec2( rcp.x,  rcp.y), 0.0).rgb);

    // Horizontal vs vertical edge.
    const float edgeH = abs((lNW + lNE) - 2.0 * lN) + 2.0 * abs((lW + lE) - 2.0 * lM) + abs((lSW + lSE) - 2.0 * lS);
    const float edgeV = abs((lNW + lSW) - 2.0 * lW) + 2.0 * abs((lN + lS) - 2.0 * lM) + abs((lNE + lSE) - 2.0 * lE);
    const bool horzSpan = edgeH >= edgeV;

    // Pick the higher-contrast side of the edge.
    const float l1 = horzSpan ? lN : lW;
    const float l2 = horzSpan ? lS : lE;
    const float grad1 = l1 - lM;
    const float grad2 = l2 - lM;
    const bool is1Steepest = abs(grad1) >= abs(grad2);
    const float gradScaled = 0.25 * max(abs(grad1), abs(grad2));

    float stepLen = horzSpan ? rcp.y : rcp.x;
    float lAvg;// average luma on the edge boundary
    if (is1Steepest) {
        stepLen = -stepLen;
        lAvg = 0.5 * (l1 + lM);
    } else {
        lAvg = 0.5 * (l2 + lM);
    }

    // Move to the boundary between the two rows/columns.
    vec2 posB = uv;
    if (horzSpan) posB.y += stepLen * 0.5;
    else          posB.x += stepLen * 0.5;

    // March along the edge in both directions until luma leaves the edge.
    const vec2 offDir = horzSpan ? vec2(rcp.x, 0.0) : vec2(0.0, rcp.y);
    vec2 posP = posB, posN = posB;
    float lEndP = 0.0, lEndN = 0.0;
    bool doneP = false, doneN = false;
    // 10 steps with growing stride ≈ 32 px reach — plenty for UI edges.
    const float kSteps[10] = float[](1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0);
    for (int i = 0; i < 10; ++i) {
        if (!doneP) {
            posP += offDir * kSteps[i];
            lEndP = lumaOf(textureLod(srcTex, posP, 0.0).rgb) - lAvg;
            doneP = abs(lEndP) >= gradScaled;
        }
        if (!doneN) {
            posN -= offDir * kSteps[i];
            lEndN = lumaOf(textureLod(srcTex, posN, 0.0).rgb) - lAvg;
            doneN = abs(lEndN) >= gradScaled;
        }
        if (doneP && doneN) break;
    }

    const float distP = horzSpan ? (posP.x - uv.x) : (posP.y - uv.y);
    const float distN = horzSpan ? (uv.x - posN.x) : (uv.y - posN.y);
    const bool isCloserP = distP < distN;
    const float distMin = min(distP, distN);
    const float spanLen = distP + distN;

    // Only shift when this pixel's side of the edge matches the end point's
    // luma direction (avoids bleeding across T-junctions).
    const float lEnd = isCloserP ? lEndP : lEndN;
    const bool goodSpan = ((lM - lAvg) < 0.0) != (lEnd < 0.0);
    float pixelOffset = goodSpan ? (0.5 - distMin / max(spanLen, 1e-6)) : 0.0;

    // Sub-pixel aliasing fallback (thin features shorter than one pixel).
    const float lAvgAll = (2.0 * (lN + lS + lE + lW) + lNW + lNE + lSW + lSE) / 12.0;
    float subPix = clamp(abs(lAvgAll - lM) / max(range, 1e-6), 0.0, 1.0);
    subPix = smoothstep(0.0, 1.0, subPix);
    subPix = subPix * subPix * 0.75;
    pixelOffset = max(pixelOffset, subPix);

    vec2 finalUv = uv;
    if (horzSpan) finalUv.y += pixelOffset * stepLen;
    else          finalUv.x += pixelOffset * stepLen;
    outColor = vec4(textureLod(srcTex, finalUv, 0.0).rgb, 1.0);
}
