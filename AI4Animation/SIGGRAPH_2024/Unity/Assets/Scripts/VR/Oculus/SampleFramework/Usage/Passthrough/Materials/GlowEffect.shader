// Oculus VR, LLC Proprietary and Confidential.
// clang-format off
Shader "MixedReality/GlowEffect"
{
  Properties
  {
    _GlowColor ("Glow Color", Color) = (1.0, 1.0, 1.0)
	_Pow ("Pow", Range (0.2,10)) = 2
    _Intensity ("Intensity", Range (0,10)) = 1
    }
	  SubShader {
    Tags {"Queue" = "Transparent"}

    Pass {
      ZWrite Off
			BlendOp RevSub
            Blend Zero One, One OneMinusSrcAlpha
	  Cull Front

      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag
      #pragma multi_compile_instancing

      #include "UnityCG.cginc"
      struct vertexInput {
        float4 vertex : POSITION;
        float3 normal: NORMAL;
		float4 texcoord : TEXCOORD0;
		float4 vertCol : COLOR;
        UNITY_VERTEX_INPUT_INSTANCE_ID
      };

      struct v2f {
		float2 uv : TEXCOORD0;
        float4 vertex : SV_POSITION;
        half3 worldNormal : TEXCOORD3;
        float3 viewDir: TEXCOORD2;
        half4 localPos : TEXCOORD4;
		float4 color : TEXCOORD1;

        float eye : EYE;
        UNITY_VERTEX_INPUT_INSTANCE_ID
        UNITY_VERTEX_OUTPUT_STEREO
      };

      float4 _GlowColor;
	  float _Pow;
	  float _Intensity;
      float4x4 _TrackingSpaceTransform;

      v2f vert(vertexInput v) {
        v2f o;
        UNITY_SETUP_INSTANCE_ID(v);
        UNITY_TRANSFER_INSTANCE_ID(v, o);
        UNITY_INITIALIZE_OUTPUT(v2f, o);
        UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
        o.vertex = UnityObjectToClipPos(v.vertex);

        o.worldNormal = UnityObjectToWorldNormal(v.normal);
        o.viewDir = WorldSpaceViewDir(v.vertex);
        o.localPos = v.vertex;
		o.uv = v.texcoord;
		o.color = v.vertCol;

        return o;
      }

      float4 frag(v2f i) : SV_Target {
        UNITY_SETUP_INSTANCE_ID(i);
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

        float fresnelNdot = (dot (normalize (i.viewDir), normalize (-i.worldNormal)));
		fresnelNdot = pow(fresnelNdot,_Pow);
		float4 color = _GlowColor;
		color.rgb += _GlowColor.rgb * fresnelNdot * _Intensity;
		color.rgb = 0;
		color.a *= fresnelNdot;

        return color;
      }
      ENDCG
    }
  }
}
