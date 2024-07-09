Shader "Unlit/Multiview Stereo"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_SrcRectLeft("SrcRectLeft", Vector) = (0,0,1,1)
		_SrcRectRight("SrcRectRight", Vector) = (0,0,1,1)
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float2 pos : TEXCOORD1;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;

			float4 _SrcRectLeft;
			float4 _SrcRectRight;

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);

				float4 srcRect = lerp(_SrcRectLeft, _SrcRectRight, unity_StereoEyeIndex);

				o.pos = TRANSFORM_TEX(v.uv, _MainTex);
				o.uv = (o.pos * srcRect.zw) + srcRect.xy;
				return o;
			}

			fixed4 frag (v2f i) : SV_Target
			{
				if (i.pos.x < 0.0 || i.pos.y < 0.0 || i.pos.x > 1.0 || i.pos.y > 1.0)
				{
					return float4(0,0,0,0);
				}

				// sample the texture
				fixed4 col = tex2D(_MainTex, i.uv);
				return col;
			}
			ENDCG
		}
	}
	Fallback "Unlit/Texture"
}
