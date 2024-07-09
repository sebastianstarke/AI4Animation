Shader "Scene/Wall"
{

  Properties
  {
    _Color ("Main Color", Color) = (1,1,1,1)
    _MainTex ("Base (RGB)", 2D) = "white" {}
    _NormalTex ("Normal Map", 2D) = "bump" {}
    _Tiling ("Tiling", float) = 1.0
  }

  SubShader
  {
    Tags { "RenderType"="Opaque" "Queue"="Geometry" }
    LOD 200

    Stencil
    {
      Ref 1
      Comp NotEqual
      Pass Keep
    }

    CGPROGRAM
    #pragma surface surf Standard noshadow
    #pragma target 3.0

    sampler2D _MainTex;
    sampler2D _NormalTex;
    fixed4 _Color;
    float _Tiling;

    struct Input
    {
      float2 uv_MainTex;
      float2 uv_NormalTex;
    };

    void surf (Input IN, inout SurfaceOutputStandard o)
    {
      fixed4 c = tex2D(_MainTex, IN.uv_MainTex * _Tiling) * _Color;
      o.Albedo = c.rgb;
      o.Alpha = c.a;
      o.Normal = UnpackNormal (tex2D(_NormalTex, IN.uv_MainTex * _Tiling));
    }

    ENDCG
  }

  Fallback "VertexLit"
}
