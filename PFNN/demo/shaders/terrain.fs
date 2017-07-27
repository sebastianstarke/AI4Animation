#version 120

#define NSAMPLES 32

uniform vec3 light_dir;
uniform mat4 light_view;
uniform mat4 light_proj;
uniform float light_bias = 0.00;

uniform sampler2D shadows;

uniform vec3 foot0;
uniform vec3 foot1;
uniform vec3 foot2;
uniform vec3 foot3;
uniform vec3 hip;

varying vec3 fNormal;
varying vec3 fPosition;
varying float fAO;

float segment_distance(vec3 s0, vec3 s1, vec3 p) {
  float l = dot(s0 - s1, s0 - s1);
  if (l == 0.0) { return length(p - s0); }
  float t = clamp(dot(p - s0, s1 - s0) / l, 0, 1);
  vec3 projection = s0 + t * (s1 - s0);
  return length(p - projection);
}

vec3 rand(vec3 seed){
  return 2.0*fract(sin(dot(seed, vec3(12.9898, 78.233, 21.317))) * vec3(43758.5453, 21383.21227, 20431.20563))-1.0;
}

float shadow_amount(vec3 position) {

  vec3 sample_sphere[32] = vec3[32](
    vec3(-0.68, -0.21, -0.00), vec3(-0.79,  0.40, -0.20), vec3(-0.10, -0.81, -0.35),
    vec3(-0.11, -0.24, -0.33), vec3(-0.39, -0.37, -0.65), vec3(-0.72,  0.22,  0.13),
    vec3(-0.42, -0.24, -0.57), vec3( 0.03,  0.09, -0.47), vec3(-0.30,  0.23,  0.36),
    vec3(-0.14, -0.48,  0.55), vec3(-0.33, -0.67,  0.40), vec3(-0.11,  0.06,  0.13),
    vec3( 0.26, -0.17, -0.43), vec3(-0.51, -0.34, -0.66), vec3( 0.13,  0.08,  0.26),
    vec3( 0.69, -0.35, -0.32), vec3( 0.52,  0.35,  0.58), vec3(-0.03, -0.75, -0.39),
    vec3( 0.07,  0.32,  0.21), vec3( 0.12,  0.25,  0.42), vec3( 0.80, -0.30, -0.09),
    vec3(-0.02,  0.68,  0.23), vec3(-0.72,  0.15, -0.63), vec3(-0.30,  0.69,  0.20),
    vec3( 0.44,  0.25,  0.52), vec3(-0.36, -0.09, -0.42), vec3( 0.69, -0.26, -0.04),
    vec3( 0.38, -0.46,  0.36), vec3(-0.25,  0.40, -0.32), vec3(-0.25,  0.85,  0.07),
    vec3(-0.27,  0.37, -0.59), vec3(-0.63, -0.52, -0.28));

  vec4 light_pos = light_proj * light_view * vec4(position, 1.0);
  light_pos = light_pos / light_pos.w;
  float pixel_depth = light_pos.z / 2 + 0.5;
  vec2  pixel_coords = vec2(light_pos.x, light_pos.y) / 2.0 + 0.5;
  
  vec2 seed = normalize(rand(position).xy);
  
  float cover = 0.0;
  for (int i = 0; i < NSAMPLES; i++) {
    cover += max(sign(texture2D(shadows, pixel_coords + (0.001 * reflect(seed, normalize(sample_sphere[i].xy)))).r - pixel_depth + light_bias), 0.0);
  }
  cover = cover / NSAMPLES;
  
  return cover / 2 + 0.5;
}

vec3 to_gamma(vec3 color) {
  vec3 ret;
  ret.r = pow(color.r, 2.2);
  ret.g = pow(color.g, 2.2);
  ret.b = pow(color.b, 2.2);
	return ret;
}
  
  
void main() {
  
  float d0 = segment_distance(foot0, foot1, fPosition);
  float d1 = segment_distance(foot2, foot3, fPosition);
  float a0 = clamp((d0 - 0.0) / 10.0 + 0.25, 0, 1);
  float a1 = clamp((d1 - 0.0) / 10.0 + 0.25, 0, 1);
  float a2 = clamp(max(length(hip.xz - fPosition.xz) + 200.0, 0.0) / 265.0, 0, 1);
  
  vec3 color_square = floor(mod(fPosition / 100.0, 1.0) + 0.5);
  float color = 0.85 * clamp(mod(color_square.x + color_square.z, 2.0) + 0.95, 0, 1);
  
  float diffuse = shadow_amount(fPosition) * 0.3 * max(dot(-light_dir, fNormal), 0.0);
  float ambient = 0.5 * fAO * a0 * a1 * a2 + 0.2;

  gl_FragColor.rgb = to_gamma(color * vec3(diffuse + ambient));
  //gl_FragColor.rgb = vec3(fAO);
  gl_FragColor.a = 1.0;
} 