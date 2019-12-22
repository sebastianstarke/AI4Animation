#ifndef __CUBOID_LINE_COMMON_INCLUDED__

#define __CUBOID_LINE_COMMON_INCLUDED__

#include "UnityCG.cginc"

struct appdata
{
    float4 vertex : POSITION;
};

struct v2g
{
    float4 vertex : SV_POSITION;
};

struct g2f
{
    float4 vertex : SV_POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

half4 _Color;
half _Thickness;

v2g vert (appdata v)
{
    v2g o;
    o.vertex = mul(unity_ObjectToWorld, v.vertex);
    return o;
}

g2f vertex_output(in g2f o, float4 pos, float3 wnrm, float2 uv)
{
    o.vertex = UnityWorldToClipPos(pos);
    o.normal = wnrm;
    o.uv = uv;
    return o;
}

void add_face(inout TriangleStream<g2f> OUT, float4 p[4], float3 wnrm)
{
    g2f o = vertex_output(o, p[0], wnrm, float2(1.0f, 0.0f));
    OUT.Append(o);

    o = vertex_output(o, p[1], wnrm, float2(1.0f, 1.0f));
    OUT.Append(o);

    o = vertex_output(o, p[2], wnrm, float2(0.0f, 0.0f));
    OUT.Append(o);

    o = vertex_output(o, p[3], wnrm, float2(0.0f, 1.0f));
    OUT.Append(o);

    OUT.RestartStrip();
}

[maxvertexcount(24)]
void geom(in line v2g IN[2], inout TriangleStream<g2f> OUT)
{
    float3 pos = (IN[0].vertex.xyz + IN[1].vertex.xyz) * 0.5;
    float3 tangent = (IN[1].vertex.xyz - pos);
    float3 forward = normalize(tangent) * (length(tangent) + _Thickness);
    float3 nforward = normalize(forward);
    float3 ntmp = cross(nforward, float3(0, 1, 0));
    float3 up = (cross(ntmp, nforward));
    float3 nup = normalize(up);
    float3 right = (cross(nforward, nup));
    float3 nright = normalize(right);

    up = nup * _Thickness;
    right = nright * _Thickness;

    float4 v[4];

    // forward
    v[0] = float4(pos + forward + right - up, 1.0f);
    v[1] = float4(pos + forward + right + up, 1.0f);
    v[2] = float4(pos + forward - right - up, 1.0f);
    v[3] = float4(pos + forward - right + up, 1.0f);
    add_face(OUT, v, nforward);

    // back
    v[0] = float4(pos - forward - right - up, 1.0f);
    v[1] = float4(pos - forward - right + up, 1.0f);
    v[2] = float4(pos - forward + right - up, 1.0f);
    v[3] = float4(pos - forward + right + up, 1.0f);
    add_face(OUT, v, -nforward);

    // up
    v[0] = float4(pos - forward + right + up, 1.0f);
    v[1] = float4(pos - forward - right + up, 1.0f);
    v[2] = float4(pos + forward + right + up, 1.0f);
    v[3] = float4(pos + forward - right + up, 1.0f);
    add_face(OUT, v, nup);

    // down
    v[0] = float4(pos + forward + right - up, 1.0f);
    v[1] = float4(pos + forward - right - up, 1.0f);
    v[2] = float4(pos - forward + right - up, 1.0f);
    v[3] = float4(pos - forward - right - up, 1.0f);
    add_face(OUT, v, -nup);

    // left
    v[0] = float4(pos + forward - right - up, 1.0f);
    v[1] = float4(pos + forward - right + up, 1.0f);
    v[2] = float4(pos - forward - right - up, 1.0f);
    v[3] = float4(pos - forward - right + up, 1.0f);
    add_face(OUT, v, -nright);

    // right
    v[0] = float4(pos - forward + right + up, 1.0f);
    v[1] = float4(pos + forward + right + up, 1.0f);
    v[2] = float4(pos - forward + right - up, 1.0f);
    v[3] = float4(pos + forward + right - up, 1.0f);
    add_face(OUT, v, nright);
}

#endif

