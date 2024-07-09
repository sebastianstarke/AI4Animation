Shader "MixedReality/SelectivePassthroughSoft" {
  Properties{
	  _MainTex("Texture", 2D) = "white" {}
	  _SoftFade("Soft Fade Distance", Float) = 0.05
	 _Inflation("Inflation", float) = 0
  }
	  SubShader {
    Tags{"RenderType" = "Transparent"} LOD 100

        Pass {
			ZWrite Off
			BlendOp RevSub, Min 
			Blend Zero One, One OneMinusSrcAlpha

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
        float2 projPos : TEXCOORD1;
        float depth : DEPTH;
      };

      sampler2D _MainTex;
      UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
      float4 _MainTex_ST;
      float _SoftFade;
      float _Inflation;

      v2f vert(appdata v) {
        v2f o;
        o.vertex = UnityObjectToClipPos(v.vertex + v.normal * _Inflation);
        float4 origin = mul(unity_ObjectToWorld, float4(0.0, 0.0, 0.0, 1.0));
        o.uv = TRANSFORM_TEX(v.uv, _MainTex);
        o.depth = -UnityObjectToViewPos(v.vertex).z * _ProjectionParams.w;
        o.projPos = (o.vertex.xy / o.vertex.w) * 0.5 + 0.5;
        return o;
      }

      fixed4 frag(v2f i) : SV_Target {
        // get a linear & normalized (0-1) depth value of the scene
        float frameDepth = Linear01Depth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.projPos.xy));
        // convert the normalized value to world units, with depth of 0 at this mesh
        frameDepth = (frameDepth - i.depth) * _ProjectionParams.z;
        // remap it to a fade distance
        frameDepth = saturate(frameDepth / _SoftFade);
        fixed4 col = tex2D(_MainTex, i.uv);
        return float4(0, 0, 0, 1-(col.r * frameDepth));
      }
      ENDCG
    }
  }
}
