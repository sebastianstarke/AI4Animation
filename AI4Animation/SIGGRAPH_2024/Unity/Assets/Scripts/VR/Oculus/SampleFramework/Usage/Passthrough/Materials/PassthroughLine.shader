Shader "MixedReality/PassthroughLine" {
  Properties{
	  _LineLength("Line Length", float) = 1

  } SubShader {
    Tags{"RenderType" = "Transparent"} LOD 100

        Pass {
      ZWrite Off
	  BlendOp RevSub, Min
	  Blend Zero One, One One

              CGPROGRAM
      // Upgrade NOTE: excluded shader from DX11; has structs without semantics (struct v2f members
      // center)
      //#pragma exclude_renderers d3d11
#pragma vertex vert
#pragma fragment frag

#include "UnityCG.cginc"

          struct appdata {
        float4 vertex : POSITION;
        float2 uv : TEXCOORD0;
        float3 normal : NORMAL;
      };

      struct v2f {
        float2 uv : TEXCOORD0;
        float4 vertex : SV_POSITION;
      };

      v2f vert(appdata v) {
        v2f o;
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = v.uv;
        return o;
      }

      float _LineLength;

      fixed4 frag(v2f i) : SV_Target {
        float UVfadeRange = 0.2;
        fixed widthAlpha = 1 - smoothstep(0.5 - UVfadeRange, 0.5, abs(i.uv.y - 0.5));
        fixed lengthAlpha = 1 - smoothstep(0.5 - (UVfadeRange / _LineLength), 0.5, abs(i.uv.x - 0.5));
        return float4(0, 0, 0, 1-(widthAlpha * lengthAlpha));
      }
      ENDCG
    }
  }
}
