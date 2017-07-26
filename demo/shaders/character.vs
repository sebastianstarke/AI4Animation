#version 120

uniform mat4 view;
uniform mat4 proj;

uniform mat4 joints[31];

attribute vec3 vPosition;
attribute vec3 vNormal;
attribute float vAO;
attribute vec4 vWeightVal;
attribute vec4 vWeightIds;

varying vec3 fNormal;
varying float fAO;
varying vec3 fPosition;

void main() {

  vec4 tpos = vec4(0);
  tpos += vWeightVal.x * joints[int(vWeightIds.x)] * vec4(vPosition, 1);
  tpos += vWeightVal.y * joints[int(vWeightIds.y)] * vec4(vPosition, 1);
  tpos += vWeightVal.z * joints[int(vWeightIds.z)] * vec4(vPosition, 1);
  tpos += vWeightVal.w * joints[int(vWeightIds.w)] * vec4(vPosition, 1);
  fPosition = tpos.xyz / tpos.w;
  
  fNormal = vec3(0);
  fNormal += vWeightVal.x * mat3(joints[int(vWeightIds.x)]) * vNormal;
  fNormal += vWeightVal.y * mat3(joints[int(vWeightIds.y)]) * vNormal;
  fNormal += vWeightVal.z * mat3(joints[int(vWeightIds.z)]) * vNormal;
  fNormal += vWeightVal.w * mat3(joints[int(vWeightIds.w)]) * vNormal;

  /*
  fPosition = vec3(0);
  fPosition += (vWeightVal.x * joints[int(vWeightIds.x)] * vec4(vPosition, 1)).xyz;
  fPosition += (vWeightVal.y * joints[int(vWeightIds.y)] * vec4(vPosition, 1)).xyz;
  fPosition += (vWeightVal.z * joints[int(vWeightIds.z)] * vec4(vPosition, 1)).xyz;
  fPosition += (vWeightVal.w * joints[int(vWeightIds.w)] * vec4(vPosition, 1)).xyz;

  fNormal = vec3(0);
  fNormal += vWeightVal.x * mat3(joints[int(vWeightIds.x)]) * vNormal;
  fNormal += vWeightVal.y * mat3(joints[int(vWeightIds.y)]) * vNormal;
  fNormal += vWeightVal.z * mat3(joints[int(vWeightIds.z)]) * vNormal;
  fNormal += vWeightVal.w * mat3(joints[int(vWeightIds.w)]) * vNormal;
  */
  
  fAO = vAO;
  
  gl_Position = proj * view * vec4(fPosition, 1);
}