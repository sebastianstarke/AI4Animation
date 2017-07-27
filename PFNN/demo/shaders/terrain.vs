#version 120

uniform mat4 view;
uniform mat4 proj;

attribute vec3 vPosition;
attribute vec3 vNormal;
attribute float vAO;

varying vec3 fNormal;
varying vec3 fPosition;
varying float fAO;

void main() {
  fNormal = vNormal;
  fPosition = vPosition;
  fAO = vAO;
  gl_Position = proj * view * vec4(vPosition, 1);
}