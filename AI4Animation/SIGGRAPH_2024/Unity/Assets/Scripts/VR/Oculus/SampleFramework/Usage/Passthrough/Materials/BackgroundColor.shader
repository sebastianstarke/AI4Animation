Shader "MixedReality/BackgroundColor" {
  Properties{
	  _Color("Color", Color) = (1, 1, 1, 1)
  } SubShader {
    Tags{"RenderType" =
             "Opaque"} LOD 100 Cull Front ZTest Always ZWrite On Blend SrcAlpha OneMinusSrcAlpha

        Pass {
      CGPROGRAM
#pragma vertex vert
#pragma fragment frag
// make fog work
#pragma multi_compile_fog

#include "UnityCG.cginc"

      struct appdata {
        float4 vertex : POSITION;
        float2 uv : TEXCOORD0;
      };

      struct v2f {
        float4 vertex : SV_POSITION;
        float2 uv : TEXCOORD0;
      };

      float4 _Color;

      v2f vert(appdata v) {
        v2f o;
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = v.uv;
        return o;
      }

      fixed4 frag(v2f i) : SV_Target {
        float gradient = (i.uv.y - 0.5) * 0.5;
        float4 col =
            float4(_Color.r + gradient, _Color.g + gradient, _Color.b + gradient, _Color.a);
        return col;
      }
      ENDCG
    }
  }
}
