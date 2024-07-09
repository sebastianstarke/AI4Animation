Shader "Oculus/OVRMRCameraFrame"
{
	Properties
	{
		_Color("Main Color", Color) = (1,1,1,1)
		_MainTex("Main Texture", 2D) = "white" {}
		_Visible("Visible", Range(0.0,1.0)) = 1.0
		_ChromaAlphaCutoff("ChromaAlphaCutoff", Range(0.0,1.0)) = 0.01
		_ChromaToleranceA("ChromaToleranceA", Range(0.0,50.0)) = 20.0
		_ChromaToleranceB("ChromaToleranceB", Range(0.0,50.0)) = 15.0
		_ChromaShadows("ChromaShadows", Range(0.0,1.0)) = 0.02
	}
	SubShader
	{
		Tags { "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
		Blend SrcAlpha OneMinusSrcAlpha
		AlphaTest Greater .01
		Fog{ Mode Off }
		LOD 100
		Cull Off

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"
			#include "OVRMRChromaKey.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 texcoord : TEXCOORD0;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float2 texcoord : TEXCOORD0;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			sampler2D _MaskTex;

			float4 _TextureDimension;		// (w, h, 1/w, 1/h)

			fixed4 _Color;
			fixed  _Visible;
			float4 _FlipParams;		// (flip_h, flip_v, 0, 0)

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.vertex *= _Visible;
				o.texcoord = TRANSFORM_TEX(float2(v.texcoord.x, 1.0-v.texcoord.y), _MainTex);
				return o;
			}

			fixed GetMask(float2 uv)
			{
				return tex2D(_MaskTex, uv).r;
			}

			fixed4 GetCameraColor(float2 colorUV)
			{
				fixed4 c = tex2D(_MainTex, colorUV) * _Color;
				return c;
			}

			fixed4 frag (v2f i) : SV_Target
			{
				float2 colorUV = i.texcoord;
				if (_FlipParams.x > 0.0)
				{
					colorUV.x = 1.0 - colorUV.x;
				}
				if (_FlipParams.y > 0.0)
				{
					colorUV.y = 1.0 - colorUV.y;
				}
				float mask = GetMask(float2(colorUV.x, 1.0 - colorUV.y));
				if (mask == 0.0)
				{
					discard;
				}
				float4 col = GetColorAfterChromaKey(colorUV, _TextureDimension.zw, 1.0);
				if (col.a < 0.0)
				{
					discard;
				}
				return col;
			}
			ENDCG
		}
	}
}
