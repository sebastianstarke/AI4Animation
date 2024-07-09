//
// Magic Mirror Pro - Recursive Edition
// (c) 2018 Digital Ruby, LLC
// Source code may be used for personal or commercial projects.
// Source code may NOT be redistributed or sold.
// 

Shader "MagicMirror/MirrorBump"
{
	Properties
	{
		_MainTex("Detail Texture", 2D) = "clear" {}
		_BumpMap("Bump Map Texture", 2D) = "bumpmap" {}
		_RecursionLimitTex ("Recursion Limit Texture", 2D) = "grey" {}
		_Color ("Detail Tint Color", Color) = (1,1,1,1)
		_SpecColor ("Specular Color", Color) = (1,1,1,1)
		_SpecularArea ("Specular Area", Range (0, 0.99)) = 0.1
		_SpecularIntensity ("Specular Intensity", Range (0, 10)) = 0.75
		_ReflectionColor ("Reflection Tint Color", Color) = (1,1,1,1)

		[HideInInspector] _ReflectionTex("Emissive Texture", 2D) = "black" {}
		[HideInInspector] _ReflectionTex2("Emissive Texture 2 (for stereo)", 2D) = "black" {}
	}
	SubShader
	{ 
		Tags { "RenderQueue"="Geometry" "RenderType"="Opaque" }
		LOD 300
     
		CGPROGRAM

		#pragma target 3.0
		#pragma surface surf BlinnPhong fullforwardshadows vertex:vert
		#pragma fragmentoption ARB_precision_hint_fastest
		#pragma glsl_no_auto_normalization
		#pragma multi_compile_instancing
		#pragma multi_compile __ MIRROR_RECURSION_LIMIT

		#include "UnityCG.cginc"
  
		fixed4 _Color;
		fixed4 _ReflectionColor;
		half _SpecularArea;
		half _SpecularIntensity;
		sampler2D _MainTex;
		sampler2D _BumpMap;
		sampler2D _RecursionLimitTex;
		sampler2D _ReflectionTex;
		sampler2D _ReflectionTex2;
  
		struct Input
		{
			float2 uv_MainTex;
			float2 uv_BumpMap;
			INTERNAL_DATA
			float4 screenPosNoStereo;
		};

		void vert(inout appdata_full v, out Input o)
		{
			UNITY_INITIALIZE_OUTPUT(Input, o);
			o.screenPosNoStereo = ComputeNonStereoScreenPos(UnityObjectToClipPos(v.vertex));

#if defined(UNITY_SINGLE_PASS_STEREO)

			o.screenPosNoStereo.z = unity_StereoEyeIndex;

#else

			// When not using single pass stereo rendering, eye index must be determined by testing the
			// sign of the horizontal skew of the projection matrix.
			o.screenPosNoStereo.z = (unity_CameraProjection[0][2] > 0);

#endif

		}

		void surf (Input IN, inout SurfaceOutput o)
		{
			fixed4 detail = tex2D(_MainTex, IN.uv_MainTex);
			fixed4 refl;

#if defined(MIRROR_RECURSION_LIMIT)

			refl = fixed4(0.0, 0.0, 0.0, 0.0);
			fixed4 tex = tex2D(_RecursionLimitTex, IN.uv_MainTex);
			o.Albedo = (detail.rgb * detail.a) + (tex.rgb * (1.0 - detail.a));

#else

			float2 screenUV = IN.screenPosNoStereo.xy / max(0.001, IN.screenPosNoStereo.w);
			refl = lerp(tex2D(_ReflectionTex, screenUV), tex2D(_ReflectionTex2, screenUV), IN.screenPosNoStereo.z);
			o.Albedo = detail.rgb * _Color.rgb;

#endif
			
			o.Alpha = 1;
			o.Specular = 1.0f - _SpecularArea;
			o.Gloss = _SpecularIntensity;
			o.Emission = refl.rgb * _ReflectionColor.rgb;
			o.Normal = UnpackNormal(tex2D(_BumpMap, IN.uv_BumpMap));
		}

		ENDCG
	}
 
	FallBack "Reflective/Specular"
}
