#version 120

uniform vec3 light_dir;
uniform mat4 light_view;
uniform mat4 light_proj;
uniform float light_bias = 0.00;

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
  
  vec3 color_square = floor(mod(fPosition / 100.0, 1.0) + 0.5);
  float color = 0.85 * clamp(mod(color_square.x + color_square.z, 2.0) + 0.95, 0, 1);
  
  float diffuse = shadow_amount(fPosition) * 0.3 * max(dot(-light_dir, fNormal), 0.0);
  float ambient = 0.5 * fAO + 0.2;

  gl_FragColor.rgb = to_gamma(color * vec3(diffuse + ambient));
  //gl_FragColor.rgb = vec3(fAO);
  gl_FragColor.a = 1.0;
} 