Shader "Oculus/OVRVignette"
{
	Properties 
	{
		_Color("Color", Color) = (0,0,0,0)
		[Enum(UnityEngine.Rendering.BlendMode)]_BlendSrc ("Blend Source", Float) = 1
		[Enum(UnityEngine.Rendering.BlendMode)]_BlendDst ("Blend Destination", Float) = 0
		_ZWrite ("Z Write", Float) = 0
	}
	SubShader
	{
		Tags { "IgnoreProjector" = "True" }

		Pass
		{
			Blend [_BlendSrc] [_BlendDst]
			ZTest Always
			ZWrite [_ZWrite]
			Cull Off

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile _ QUADRATIC_FALLOFF
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				half4 color : COLOR;
			};

			float4 _ScaleAndOffset0[2];
			float4 _ScaleAndOffset1[2];
			float4 _Color;

			v2f vert (appdata v)
			{
				v2f o;

				float4 scaleAndOffset = lerp(_ScaleAndOffset0[unity_StereoEyeIndex], _ScaleAndOffset1[unity_StereoEyeIndex], v.uv.x);

				o.vertex = float4(scaleAndOffset.zw + v.vertex.xy * scaleAndOffset.xy, _ProjectionParams.y, 1);

				o.color.rgb = _Color.rgb;
				o.color.a = v.uv.y;
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
#if QUADRATIC_FALLOFF
				i.color.a *= i.color.a;
#endif				
				return i.color;
			}
			ENDCG
		}
	}
}
