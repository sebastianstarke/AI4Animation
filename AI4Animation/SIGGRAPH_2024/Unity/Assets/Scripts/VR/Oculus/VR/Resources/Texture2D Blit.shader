Shader "Oculus/Texture2D Blit" {
    Properties{
        _MainTex("Base (RGB) Trans (A)", 2D) = "white" {}
        _linearToSrgb("Perform linear-to-gamma conversion", Int) = 0
        _premultiply("Pre-multiply alpha", Int) = 0
		_flip("Y-Flip", Int) = 0
    }
    SubShader{
        Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }

        Pass{
            ZWrite Off
            ColorMask RGBA

			CGPROGRAM
				#pragma vertex vert
				#pragma fragment frag

				#include "UnityCG.cginc"

				struct appdata_t
				{
					float4 vertex : POSITION;
					float2 texcoord : TEXCOORD0;
				};

				struct v2f
				{
					float4 vertex : SV_POSITION;
					half2 texcoord : TEXCOORD0;
				};

				sampler2D _MainTex;
				float4 _MainTex_ST;
				int _linearToSrgb;
				int _premultiply;
				int _flip;

				v2f vert (appdata_t v)
				{
					v2f o;
					o.vertex = UnityObjectToClipPos(v.vertex);
					o.texcoord = TRANSFORM_TEX(v.texcoord, _MainTex);
					return o;
				}

				fixed4 frag (v2f i) : COLOR
				{
#if SHADER_API_D3D11
					if (_flip)
					{
							i.texcoord.y = 1.0f - i.texcoord.y;
					}
#endif
					fixed4 col = tex2D(_MainTex, i.texcoord);
					if (_linearToSrgb)
					{
						float3 S1 = sqrt(col.rgb);
						float3 S2 = sqrt(S1);
						float3 S3 = sqrt(S2);
						col.rgb = 0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.0225411470 * col.rgb;
					}

					if (_premultiply)
						col.rgb *= col.a;

					return col;
				}
			ENDCG
		}
    }
}
