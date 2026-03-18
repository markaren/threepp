#version 450

// Water vertex shader -- standalone Vulkan GLSL 450 for naga translation.
// Matches DawnRenderer bind group layout for custom ShaderMaterial pipelines.

// Binding 0: TransformUniforms (matches DawnRenderer's 256-byte transform UBO)
layout(std140, set=0, binding=0) uniform TransformUniforms {
    mat4 modelMatrix;       // floats  0-15
    mat4 viewMatrix;        // floats 16-31
    mat4 projectionMatrix;  // floats 32-47
    vec4 normalCol0;        // floats 48-51 (normal matrix column 0 + pad)
    vec4 normalCol1;        // floats 52-55 (normal matrix column 1 + pad)
    vec4 normalCol2;        // floats 56-59 (normal matrix column 2 + pad)
    vec4 cameraPosition;    // floats 60-63 (.xyz = camera world pos)
};

// Binding 1: LightData placeholder (required by BGL even if unused in vertex)
layout(std140, set=0, binding=1) uniform LightPlaceholder {
    vec4 _lightPad;
};

// Binding 2: CustomUniforms
// The CPU-side packer in DawnRenderer sorts uniform names alphabetically and
// places each field in a 16-byte slot (float/int/vec3 each occupy 16 bytes;
// mat4 occupies 64 bytes). This layout matches that scheme exactly:
//
//   alpha           float   offset  0  (4 bytes + 12 pad = 16)
//   distortionScale float   offset 16  (4 bytes + 12 pad = 16)
//   eye             vec3    offset 32  (12 bytes + 4 pad = 16)
//   size            float   offset 48  (4 bytes + 12 pad = 16)
//   sunColor        vec3    offset 64  (12 bytes + 4 pad = 16)
//   sunDirection    vec3    offset 80  (12 bytes + 4 pad = 16)
//   textureMatrix   mat4    offset 96  (64 bytes)
//   time            float   offset 160 (4 bytes + 12 pad = 16)
//   waterColor      vec3    offset 176 (12 bytes + 4 pad = 16)
//
// The vertex stage only reads textureMatrix and time.
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

// Vertex attributes
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;

// Outputs to fragment stage
layout(location=0) out vec4 mirrorCoord;
layout(location=1) out vec4 worldPosition;

void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    worldPosition = worldPos;
    mirrorCoord = textureMatrix * worldPos;

    vec4 mvPosition = viewMatrix * worldPos;
    gl_Position = projectionMatrix * mvPosition;
}
