#version 450

// Water fragment shader -- standalone Vulkan GLSL 450 for naga translation.
// Matches WgpuRenderer bind group layout for custom ShaderMaterial pipelines.

// Binding 0: TransformUniforms (same as vertex)
layout(std140, set=0, binding=0) uniform TransformUniforms {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 normalCol0;
    vec4 normalCol1;
    vec4 normalCol2;
    vec4 cameraPosition;
};

// Binding 1: LightData placeholder
layout(std140, set=0, binding=1) uniform LightPlaceholder {
    vec4 _lightPad;
};

// Binding 2: CustomUniforms (must match vertex shader layout exactly)
// See water.vert for the full layout rationale (alphabetical, 16-byte-per-slot).
layout(std140, set=0, binding=2) uniform CustomUniforms {
    float alpha;            // offset 0
    float _pad0[3];
    float distortionScale;  // offset 16
    float _pad1[3];
    vec3  eye;              // offset 32
    float _pad2;
    float size;             // offset 48
    float _pad3[3];
    vec3  sunColor;         // offset 64
    float _pad4;
    vec3  sunDirection;     // offset 80
    float _pad5;
    mat4  textureMatrix;    // offset 96
    float time;             // offset 160
    float _pad6[3];
    vec3  waterColor;       // offset 176
    float _pad7;
};

// Texture bindings: naga translates Vulkan combined image+sampler (sampler2D)
// into separate texture_2d and sampler bindings in WGSL. Each sampler2D
// declared here at binding N produces WGSL bindings at N (texture) and N+1
// (sampler). This matches WgpuPipelines which assigns:
//   mirrorSampler: texture@3, sampler@4
//   normalSampler: texture@5, sampler@6
layout(set=0, binding=3) uniform sampler2D mirrorSampler;
layout(set=0, binding=5) uniform sampler2D normalSampler;

// Inputs from vertex stage
layout(location=0) in vec4 mirrorCoord;
layout(location=1) in vec4 worldPosition;

// Output
layout(location=0) out vec4 fragColor;

// Sample four offset noise textures from the normal map
vec4 getNoise(vec2 uv) {
    vec2 uv0 = (uv / 103.0) + vec2(time / 17.0, time / 29.0);
    vec2 uv1 = uv / 107.0 - vec2(time / -19.0, time / 31.0);
    vec2 uv2 = uv / vec2(8907.0, 9803.0) + vec2(time / 101.0, time / 97.0);
    vec2 uv3 = uv / vec2(1091.0, 1027.0) - vec2(time / 109.0, time / -113.0);

    vec4 noise = texture(normalSampler, uv0) +
                 texture(normalSampler, uv1) +
                 texture(normalSampler, uv2) +
                 texture(normalSampler, uv3);

    return noise * 0.5 - 1.0;
}

// Compute sun contribution (specular + diffuse)
void sunLight(vec3 surfaceNormal, vec3 eyeDirection,
              float shiny, float spec, float diffuse,
              inout vec3 diffuseColor, inout vec3 specularColor) {
    vec3 reflection = normalize(reflect(-sunDirection, surfaceNormal));
    float direction = max(0.0, dot(eyeDirection, reflection));
    specularColor += pow(direction, shiny) * sunColor * spec;
    diffuseColor  += max(dot(sunDirection, surfaceNormal), 0.0) * sunColor * diffuse;
}

void main() {
    vec4 noise = getNoise(worldPosition.xz * size);
    vec3 surfaceNormal = normalize(noise.xzy * vec3(1.5, 1.0, 1.5));

    vec3 diffuseLight  = vec3(0.0);
    vec3 specularLight = vec3(0.0);

    vec3 worldToEye = eye - worldPosition.xyz;
    vec3 eyeDirection = normalize(worldToEye);

    sunLight(surfaceNormal, eyeDirection, 100.0, 2.0, 0.5, diffuseLight, specularLight);

    float dist = length(worldToEye);

    vec2 distortion = surfaceNormal.xz * (0.001 + 1.0 / dist) * distortionScale;
    vec3 reflectionSample = vec3(texture(mirrorSampler,
                                         mirrorCoord.xy / mirrorCoord.w + distortion));

    float theta = max(dot(eyeDirection, surfaceNormal), 0.0);

    // Fresnel (Schlick approximation)
    float rf0 = 0.3;
    float reflectance = rf0 + (1.0 - rf0) * pow(1.0 - theta, 5.0);

    vec3 scatter = max(0.0, dot(surfaceNormal, eyeDirection)) * waterColor;

    vec3 albedo = mix(
        (sunColor * diffuseLight * 0.3 + scatter),
        (vec3(0.1) + reflectionSample * 0.9 + reflectionSample * specularLight),
        reflectance
    );

    fragColor = vec4(albedo, alpha);
}
