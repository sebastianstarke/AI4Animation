Shader "UI/Prerendered"
{
	Properties
	{
		_MainTex("Texture", 2D) = "white" {} 
		_Color("Color", Color) = (1,1,1,1)
	}
	SubShader
	{
		Tags {"Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent"}

		Blend One OneMinusSrcAlpha, Zero Zero
		Cull Off
		ZWrite Off

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile _ ALPHA_SQUARED

			#include "UnityCG.cginc"

			struct appdata_t {
				float4 vertex : POSITION;
				float2 texcoord : TEXCOORD0;
			};

			struct v2f {
				float4 vertex : SV_POSITION;
				half2 texcoord : TEXCOORD0;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			fixed4 _Color;

			v2f vert(appdata_t v) {
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.texcoord = TRANSFORM_TEX(v.texcoord, _MainTex);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target {
				fixed4 col = tex2D(_MainTex, i.texcoord);

				#if ALPHA_SQUARED
				// prerended UI will have a = Alpha * SrcAlpha, so we need to sqrt
				// to get the original alpha value
				col.a = sqrt(col.a);
				#endif
				
				// It should be noted that with Gamma lighting on PC,
				// the blend will result in not correct colors of transparent
				// portions of the overlay

				col *= _Color;
				return col;
			}
			ENDCG
		}
	}
}
