#version 120

uniform mat4 light_view;
uniform mat4 light_proj;

uniform mat4 joints[31];

attribute vec3 vPosition;
attribute vec4 vWeightVal;
attribute vec4 vWeightIds;

void main() {

  vec4 tpos = vec4(0);
  tpos += vWeightVal.x * joints[int(vWeightIds.x)] * vec4(vPosition, 1);
  tpos += vWeightVal.y * joints[int(vWeightIds.y)] * vec4(vPosition, 1);
  tpos += vWeightVal.z * joints[int(vWeightIds.z)] * vec4(vPosition, 1);
  tpos += vWeightVal.w * joints[int(vWeightIds.w)] * vec4(vPosition, 1);
  
  gl_Position = light_proj * light_view * vec4(tpos.xyz / tpos.w, 1);
}