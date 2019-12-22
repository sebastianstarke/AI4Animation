Shader "CuboidLine/Color"
{

	Properties
	{
        _Color ("Color", Color) = (1, 1, 1, 1)
        _Intensity ("Intensity", Float) = 1.0
        _Thickness ("Thickness", Float) = 0.1
	}

	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100
		Cull Off

		Pass
		{
			CGPROGRAM

			#include "CuboidLine.cginc"

			#pragma vertex vert
			#pragma geometry geom
			#pragma fragment frag

            half _Intensity;

			float4 frag (g2f i) : SV_Target
			{
				return _Color * _Intensity;
			}

			ENDCG
		}
	}
}
