// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'

Shader "FX/Water4" {
Properties {
	_ReflectionTex ("Internal reflection", 2D) = "white" {}
	
	_MainTex ("Fallback texture", 2D) = "black" {}
	_ShoreTex ("Shore & Foam texture ", 2D) = "black" {}
	_BumpMap ("Normals ", 2D) = "bump" {}
	
	_DistortParams ("Distortions (Bump waves, Reflection, Fresnel power, Fresnel bias)", Vector) = (1.0 ,1.0, 2.0, 1.15)
	_InvFadeParemeter ("Auto blend parameter (Edge, Shore, Distance scale)", Vector) = (0.15 ,0.15, 0.5, 1.0)
	
	_AnimationTiling ("Animation Tiling (Displacement)", Vector) = (2.2 ,2.2, -1.1, -1.1)
	_AnimationDirection ("Animation Direction (displacement)", Vector) = (1.0 ,1.0, 1.0, 1.0)

	_BumpTiling ("Bump Tiling", Vector) = (1.0 ,1.0, -2.0, 3.0)
	_BumpDirection ("Bump Direction & Speed", Vector) = (1.0 ,1.0, -1.0, 1.0)
	
	_FresnelScale ("FresnelScale", Range (0.15, 4.0)) = 0.75

	_BaseColor ("Base color", COLOR)  = ( .54, .95, .99, 0.5)
	_ReflectionColor ("Reflection color", COLOR)  = ( .54, .95, .99, 0.5)
	_SpecularColor ("Specular color", COLOR)  = ( .72, .72, .72, 1)
	
	_WorldLightDir ("Specular light direction", Vector) = (0.0, 0.1, -0.5, 0.0)
	_Shininess ("Shininess", Range (2.0, 500.0)) = 200.0
	
	_Foam ("Foam (intensity, cutoff)", Vector) = (0.1, 0.375, 0.0, 0.0)
	
	_GerstnerIntensity("Per vertex displacement", Float) = 1.0
	_GAmplitude ("Wave Amplitude", Vector) = (0.3 ,0.35, 0.25, 0.25)
	_GFrequency ("Wave Frequency", Vector) = (1.3, 1.35, 1.25, 1.25)
	_GSteepness ("Wave Steepness", Vector) = (1.0, 1.0, 1.0, 1.0)
	_GSpeed ("Wave Speed", Vector) = (1.2, 1.375, 1.1, 1.5)
	_GDirectionAB ("Wave Direction", Vector) = (0.3 ,0.85, 0.85, 0.25)
	_GDirectionCD ("Wave Direction", Vector) = (0.1 ,0.9, 0.5, 0.5)
}


CGINCLUDE

	#include "UnityCG.cginc"
	#include "WaterInclude.cginc"

	struct appdata
	{
		float4 vertex : POSITION;
		float3 normal : NORMAL;
	};

	// interpolator structs
	
	struct v2f
	{
		float4 pos : SV_POSITION;
		float4 normalInterpolator : TEXCOORD0;
		float4 viewInterpolator : TEXCOORD1;
		float4 bumpCoords : TEXCOORD2;
		float4 screenPos : TEXCOORD3;
		float4 grabPassPos : TEXCOORD4;
		UNITY_FOG_COORDS(5)
	};

	struct v2f_noGrab
	{
		float4 pos : SV_POSITION;
		float4 normalInterpolator : TEXCOORD0;
		float3 viewInterpolator : TEXCOORD1;
		float4 bumpCoords : TEXCOORD2;
		float4 screenPos : TEXCOORD3;
		UNITY_FOG_COORDS(4)
	};
	
	struct v2f_simple
	{
		float4 pos : SV_POSITION;
		float4 viewInterpolator : TEXCOORD0;
		float4 bumpCoords : TEXCOORD1;
		UNITY_FOG_COORDS(2)
	};

	// textures
	sampler2D _BumpMap;
	sampler2D _ReflectionTex;
	sampler2D _RefractionTex;
	sampler2D _ShoreTex;
	sampler2D_float _CameraDepthTexture;

	// colors in use
	uniform float4 _RefrColorDepth;
	uniform float4 _SpecularColor;
	uniform float4 _BaseColor;
	uniform float4 _ReflectionColor;
	
	// edge & shore fading
	uniform float4 _InvFadeParemeter;

	// specularity
	uniform float _Shininess;
	uniform float4 _WorldLightDir;

	// fresnel, vertex & bump displacements & strength
	uniform float4 _DistortParams;
	uniform float _FresnelScale;
	uniform float4 _BumpTiling;
	uniform float4 _BumpDirection;

	uniform float4 _GAmplitude;
	uniform float4 _GFrequency;
	uniform float4 _GSteepness;
	uniform float4 _GSpeed;
	uniform float4 _GDirectionAB;
	uniform float4 _GDirectionCD;
	
	// foam
	uniform float4 _Foam;
	
	// shortcuts
	#define PER_PIXEL_DISPLACE _DistortParams.x
	#define REALTIME_DISTORTION _DistortParams.y
	#define FRESNEL_POWER _DistortParams.z
	#define VERTEX_WORLD_NORMAL i.normalInterpolator.xyz
	#define FRESNEL_BIAS _DistortParams.w
	#define NORMAL_DISPLACEMENT_PER_VERTEX _InvFadeParemeter.z
	
	//
	// HQ VERSION
	//
	
	v2f vert(appdata_full v)
	{
		v2f o;
		
		half3 worldSpaceVertex = mul(unity_ObjectToWorld,(v.vertex)).xyz;
		half3 vtxForAni = (worldSpaceVertex).xzz;

		half3 nrml;
		half3 offsets;
		Gerstner (
			offsets, nrml, v.vertex.xyz, vtxForAni,						// offsets, nrml will be written
			_GAmplitude,												// amplitude
			_GFrequency,												// frequency
			_GSteepness,												// steepness
			_GSpeed,													// speed
			_GDirectionAB,												// direction # 1, 2
			_GDirectionCD												// direction # 3, 4
		);
		
		v.vertex.xyz += offsets;
		
		// one can also use worldSpaceVertex.xz here (speed!), albeit it'll end up a little skewed
		half2 tileableUv = mul(unity_ObjectToWorld,(v.vertex)).xz;
		
		o.bumpCoords.xyzw = (tileableUv.xyxy + _Time.xxxx * _BumpDirection.xyzw) * _BumpTiling.xyzw;

		o.viewInterpolator.xyz = worldSpaceVertex - _WorldSpaceCameraPos;

		o.pos = UnityObjectToClipPos(v.vertex);

		ComputeScreenAndGrabPassPos(o.pos, o.screenPos, o.grabPassPos);
		
		o.normalInterpolator.xyz = nrml;
		
		o.viewInterpolator.w = saturate(offsets.y);
		o.normalInterpolator.w = 1;//GetDistanceFadeout(o.screenPos.w, DISTANCE_SCALE);
		
		UNITY_TRANSFER_FOG(o,o.pos);
		return o;
	}

	half4 frag( v2f i ) : SV_Target
	{
		half3 worldNormal = PerPixelNormal(_BumpMap, i.bumpCoords, VERTEX_WORLD_NORMAL, PER_PIXEL_DISPLACE);
		half3 viewVector = normalize(i.viewInterpolator.xyz);

		half4 distortOffset = half4(worldNormal.xz * REALTIME_DISTORTION * 10.0, 0, 0);
		half4 screenWithOffset = i.screenPos + distortOffset;
		half4 grabWithOffset = i.grabPassPos + distortOffset;
		
		half4 rtRefractionsNoDistort = tex2Dproj(_RefractionTex, UNITY_PROJ_COORD(i.grabPassPos));
		half refrFix = SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(grabWithOffset));
		half4 rtRefractions = tex2Dproj(_RefractionTex, UNITY_PROJ_COORD(grabWithOffset));
		
		#ifdef WATER_REFLECTIVE
			half4 rtReflections = tex2Dproj(_ReflectionTex, UNITY_PROJ_COORD(screenWithOffset));
		#endif

		#ifdef WATER_EDGEBLEND_ON
		if (LinearEyeDepth(refrFix) < i.screenPos.z)
			rtRefractions = rtRefractionsNoDistort;
		#endif
		
		half3 reflectVector = normalize(reflect(viewVector, worldNormal));
		half3 h = normalize ((_WorldLightDir.xyz) + viewVector.xyz);
		float nh = max (0, dot (worldNormal, -h));
		float spec = max(0.0,pow (nh, _Shininess));
		
		half4 edgeBlendFactors = half4(1.0, 0.0, 0.0, 0.0);
		
		#ifdef WATER_EDGEBLEND_ON
			float depth = SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.screenPos));
			depth = LinearEyeDepth(depth);
			edgeBlendFactors = saturate(_InvFadeParemeter * (depth-i.screenPos.w));
			edgeBlendFactors.y = 1.0-edgeBlendFactors.y;
		#endif
		
		// shading for fresnel term
		worldNormal.xz *= _FresnelScale;
		half refl2Refr = Fresnel(viewVector, worldNormal, FRESNEL_BIAS, FRESNEL_POWER);
		
		// base, depth & reflection colors
		half4 baseColor = ExtinctColor (_BaseColor, i.viewInterpolator.w * _InvFadeParemeter.w);
		#ifdef WATER_REFLECTIVE
			half4 reflectionColor = lerp (rtReflections,_ReflectionColor,_ReflectionColor.a);
		#else
			half4 reflectionColor = _ReflectionColor;
		#endif
		
		baseColor = lerp (lerp (rtRefractions, baseColor, baseColor.a), reflectionColor, refl2Refr);
		baseColor = baseColor + spec * _SpecularColor;
		
		// handle foam
		half4 foam = Foam(_ShoreTex, i.bumpCoords * 2.0);
		baseColor.rgb += foam.rgb * _Foam.x * (edgeBlendFactors.y + saturate(i.viewInterpolator.w - _Foam.y));
		
		baseColor.a = edgeBlendFactors.x;
		UNITY_APPLY_FOG(i.fogCoord, baseColor);
		return baseColor;
	}
	
	//
	// MQ VERSION
	//
	
	v2f_noGrab vert300(appdata_full v)
	{
		v2f_noGrab o;
		
		half3 worldSpaceVertex = mul(unity_ObjectToWorld,(v.vertex)).xyz;
		half3 vtxForAni = (worldSpaceVertex).xzz;

		half3 nrml;
		half3 offsets;
		Gerstner (
			offsets, nrml, v.vertex.xyz, vtxForAni,						// offsets, nrml will be written
			_GAmplitude,												// amplitude
			_GFrequency,												// frequency
			_GSteepness,												// steepness
			_GSpeed,													// speed
			_GDirectionAB,												// direction # 1, 2
			_GDirectionCD												// direction # 3, 4
		);
		
		v.vertex.xyz += offsets;
		
		// one can also use worldSpaceVertex.xz here (speed!), albeit it'll end up a little skewed
		half2 tileableUv = mul(unity_ObjectToWorld,v.vertex).xz;
		o.bumpCoords.xyzw = (tileableUv.xyxy + _Time.xxxx * _BumpDirection.xyzw) * _BumpTiling.xyzw;

		o.viewInterpolator.xyz = worldSpaceVertex - _WorldSpaceCameraPos;

		o.pos = UnityObjectToClipPos(v.vertex);

		o.screenPos = ComputeNonStereoScreenPos(o.pos);
		
		o.normalInterpolator.xyz = nrml;
		o.normalInterpolator.w = 1;//GetDistanceFadeout(o.screenPos.w, DISTANCE_SCALE);
		
		UNITY_TRANSFER_FOG(o,o.pos);
		return o;
	}

	half4 frag300( v2f_noGrab i ) : SV_Target
	{
		half3 worldNormal = PerPixelNormal(_BumpMap, i.bumpCoords, normalize(VERTEX_WORLD_NORMAL), PER_PIXEL_DISPLACE);

		half3 viewVector = normalize(i.viewInterpolator.xyz);

		half4 distortOffset = half4(worldNormal.xz * REALTIME_DISTORTION * 10.0, 0, 0);
		half4 screenWithOffset = i.screenPos + distortOffset;
		
		#ifdef WATER_REFLECTIVE
			half4 rtReflections = tex2Dproj(_ReflectionTex, UNITY_PROJ_COORD(screenWithOffset));
		#endif
		
		half3 reflectVector = normalize(reflect(viewVector, worldNormal));
		half3 h = normalize (_WorldLightDir.xyz + viewVector.xyz);
		float nh = max (0, dot (worldNormal, -h));
		float spec = max(0.0,pow (nh, _Shininess));
		
		half4 edgeBlendFactors = half4(1.0, 0.0, 0.0, 0.0);
		
		#ifdef WATER_EDGEBLEND_ON
			half depth = SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.screenPos));
			depth = LinearEyeDepth(depth);
			edgeBlendFactors = saturate(_InvFadeParemeter * (depth-i.screenPos.z));
			edgeBlendFactors.y = 1.0-edgeBlendFactors.y;
		#endif
		
		worldNormal.xz *= _FresnelScale;
		half refl2Refr = Fresnel(viewVector, worldNormal, FRESNEL_BIAS, FRESNEL_POWER);
		
		half4 baseColor = _BaseColor;
		#ifdef WATER_REFLECTIVE
			baseColor = lerp (baseColor, lerp (rtReflections,_ReflectionColor,_ReflectionColor.a), saturate(refl2Refr * 2.0));
		#else
			baseColor = lerp (baseColor, _ReflectionColor, saturate(refl2Refr * 2.0));
		#endif
		
		baseColor = baseColor + spec * _SpecularColor;
		
		baseColor.a = edgeBlendFactors.x * saturate(0.5 + refl2Refr * 1.0);
		UNITY_APPLY_FOG(i.fogCoord, baseColor);
		return baseColor;
	}
	
	//
	// LQ VERSION
	//
	
	v2f_simple vert200(appdata_full v)
	{
		v2f_simple o;
		
		half3 worldSpaceVertex = mul(unity_ObjectToWorld, v.vertex).xyz;
		half2 tileableUv = worldSpaceVertex.xz;

		o.bumpCoords.xyzw = (tileableUv.xyxy + _Time.xxxx * _BumpDirection.xyzw) * _BumpTiling.xyzw;

		o.viewInterpolator.xyz = worldSpaceVertex-_WorldSpaceCameraPos;
		
		o.pos = UnityObjectToClipPos( v.vertex);
		
		o.viewInterpolator.w = 1;//GetDistanceFadeout(ComputeNonStereoScreenPos(o.pos).w, DISTANCE_SCALE);
		
		UNITY_TRANSFER_FOG(o,o.pos);
		return o;

	}

	half4 frag200( v2f_simple i ) : SV_Target
	{
		half3 worldNormal = PerPixelNormal(_BumpMap, i.bumpCoords, half3(0,1,0), PER_PIXEL_DISPLACE);
		half3 viewVector = normalize(i.viewInterpolator.xyz);

		half3 reflectVector = normalize(reflect(viewVector, worldNormal));
		half3 h = normalize ((_WorldLightDir.xyz) + viewVector.xyz);
		float nh = max (0, dot (worldNormal, -h));
		float spec = max(0.0,pow (nh, _Shininess));

		worldNormal.xz *= _FresnelScale;
		half refl2Refr = Fresnel(viewVector, worldNormal, FRESNEL_BIAS, FRESNEL_POWER);

		half4 baseColor = _BaseColor;
		baseColor = lerp(baseColor, _ReflectionColor, saturate(refl2Refr * 2.0));
		baseColor.a = saturate(2.0 * refl2Refr + 0.5);

		baseColor.rgb += spec * _SpecularColor.rgb;
		UNITY_APPLY_FOG(i.fogCoord, baseColor);
		return baseColor;
	}
	
ENDCG

Subshader
{
	Tags {"RenderType"="Transparent" "Queue"="Transparent"}
	
	Lod 500
	ColorMask RGB
	
	GrabPass { "_RefractionTex" }
	
	Pass {
			Blend SrcAlpha OneMinusSrcAlpha
			ZTest LEqual
			ZWrite Off
			Cull Off
		
			CGPROGRAM
		
			#pragma target 3.0
		
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fog
		
			#pragma multi_compile WATER_VERTEX_DISPLACEMENT_ON WATER_VERTEX_DISPLACEMENT_OFF
			#pragma multi_compile WATER_EDGEBLEND_ON WATER_EDGEBLEND_OFF
			#pragma multi_compile WATER_REFLECTIVE WATER_SIMPLE
		
			ENDCG
	}
}

Subshader
{
	Tags {"RenderType"="Transparent" "Queue"="Transparent"}
	
	Lod 300
	ColorMask RGB
	
	Pass {
			Blend SrcAlpha OneMinusSrcAlpha
			ZTest LEqual
			ZWrite Off
			Cull Off
		
			CGPROGRAM
		
			#pragma target 3.0
		
			#pragma vertex vert300
			#pragma fragment frag300
			#pragma multi_compile_fog
		
			#pragma multi_compile WATER_VERTEX_DISPLACEMENT_ON WATER_VERTEX_DISPLACEMENT_OFF
			#pragma multi_compile WATER_EDGEBLEND_ON WATER_EDGEBLEND_OFF
			#pragma multi_compile WATER_REFLECTIVE WATER_SIMPLE
		
			ENDCG
	}
}

Subshader
{
	Tags {"RenderType"="Transparent" "Queue"="Transparent"}
	
	Lod 200
	ColorMask RGB
	
	Pass {
			Blend SrcAlpha OneMinusSrcAlpha
			ZTest LEqual
			ZWrite Off
			Cull Off
		
			CGPROGRAM
		
			#pragma vertex vert200
			#pragma fragment frag200
			#pragma multi_compile_fog
		
			ENDCG
	}
}

Fallback "Transparent/Diffuse"
}
