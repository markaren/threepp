
varying vec3 vWorldDirection;

// A parallel projection has ONE view direction for every pixel, where a
// perspective one has a direction per pixel. The box below only fills the
// screen under perspective because the eye sits inside it and the divide
// expands it to any size; under an orthographic camera there is no divide, so
// it projects at its literal size and the environment shows up as a small box
// in the middle of the viewport. GLBackground scales the box to cover the
// frustum and passes the camera's world forward here instead. .w is 1 for an
// orthographic camera, 0 otherwise, so the perspective path is byte-identical
// to before. Matches the Vulkan backend's camRayDir (see camera_ray.glsl).
uniform vec4 orthoDirection;

#include <common>

void main() {

	vWorldDirection = orthoDirection.w > 0.5 ? orthoDirection.xyz : transformDirection( position, modelMatrix );

	#include <begin_vertex>
	#include <project_vertex>

	gl_Position.z = gl_Position.w; // set z to camera.far

}

