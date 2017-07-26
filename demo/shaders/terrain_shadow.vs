#version 120

uniform mat4 light_view;
uniform mat4 light_proj;

attribute vec3 vPosition;

void main() {
  gl_Position = light_proj * light_view * vec4(vPosition, 1);
}