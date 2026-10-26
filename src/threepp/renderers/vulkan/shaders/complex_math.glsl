// Complex arithmetic on vec2 (x = real, y = imaginary), for the ocean FFT
// passes: the spectrum's time evolution (dynamic_spectrum.comp) and the
// inverse-FFT butterflies (ifft_horizontal.comp, ifft_vertical.comp).

#ifndef COMPLEX_MATH_GLSL
#define COMPLEX_MATH_GLSL

vec2 complexMul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

#endif// COMPLEX_MATH_GLSL
