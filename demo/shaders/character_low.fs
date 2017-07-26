#version 120

uniform vec3 light_dir;
uniform mat4 light_view;
uniform mat4 light_proj;
uniform float light_bias = 0.0;
uniform vec3 color = vec3(1.0, 0.9, 0.6);

uniform sampler2D shadows;

varying vec3 fNormal;
varying vec3 fPosition;
varying float fAO;

float shadow_amount(vec3 position) {

  vec4 light_pos = light_proj * light_view * vec4(position, 1.0);
  light_pos = light_pos / light_pos.w;
  float pixel_depth = light_pos.z / 2 + 0.5;
  vec2  pixel_coords = vec2(light_pos.x, light_pos.y) / 2.0 + 0.5;
  
  float cover = max(sign(texture2D(shadows, pixel_coords).r - pixel_depth + light_bias), 0.0);
  
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

  float diffuse = shadow_amount(fPosition) * 0.3 * max(dot(-light_dir, fNormal), 0.0);
  float ambient = 0.5 * clamp(fAO, 0.0, 1) + 0.2;

  gl_FragColor.rgb = to_gamma(color * vec3(diffuse + ambient));
  gl_FragColor.a = 1.0;
} 