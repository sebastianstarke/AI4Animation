// Custom shader to draw our toy cubes and balls with an outline around them.
Shader "Custom/ToyCubeOutline" 
{
	Properties 
	{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo", 2D) = "white" {}
		[PerRendererData] _OutlineColor ("Outline Color", Color) = (0,0,0,1)
		_OutlineWidth ("Outline width", Range (.002, 0.03)) = .005
		
		[HideInInspector] _Mode ("__mode", Float) = 0.0
		[HideInInspector] _SrcBlend ("__src", Float) = 1.0
		[HideInInspector] _DstBlend ("__dst", Float) = 0.0
		[HideInInspector] _ZWrite ("__zw", Float) = 1.0
	}

	
	CGINCLUDE
	#include "UnityCG.cginc"
	
	struct appdata 
	{
		float4 vertex : POSITION;
		float3 normal : NORMAL;

		UNITY_VERTEX_INPUT_INSTANCE_ID
	};

	struct v2f 
	{
		float4 pos : SV_POSITION;
		fixed4 color : COLOR;

		UNITY_VERTEX_OUTPUT_STEREO
	};
	
	uniform float _OutlineWidth;
	uniform float4 _OutlineColor;
	uniform float4x4 _ObjectToWorldFixed;
	
	// Pushes the verts out a little from the object center.
	// Lets us give an outline to objects that all have normals facing away from the center.
	// If we can't assume that, we need to tweak the math of this shader.
	v2f vert(appdata v) 
	{
		v2f o;

		UNITY_SETUP_INSTANCE_ID(v);
		UNITY_INITIALIZE_OUTPUT(v2f, o);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		// MTF TODO 
		// 1. Fix batching so that it actually occurs.
		// 2. See if batching causes problems,
		// if it does fix this line by adding that component that sets it.
		//float4 objectCenterWorld = mul(_ObjectToWorldFixed, float4(0.0, 0.0, 0.0, 1.0));
		float4 objectCenterWorld = mul(unity_ObjectToWorld, float4(0.0, 0.0, 0.0, 1.0));
		float4 vertWorld = mul(unity_ObjectToWorld, v.vertex);

		float3 offsetDir = vertWorld.xyz - objectCenterWorld.xyz;
		offsetDir = normalize(offsetDir) * _OutlineWidth;

		o.pos = UnityWorldToClipPos(vertWorld+offsetDir);

		o.color = _OutlineColor;
		return o;
	}
	ENDCG

	SubShader 
	{
		Tags { "Queue" = "Transparent" }
		Pass 
		{
			Name "OUTLINE"
			// To allow the cube to render entirely on top of the outline.
			ZWrite Off
			Blend SrcAlpha OneMinusSrcAlpha

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			fixed4 frag(v2f i) : SV_Target
			{
				// Just draw the _OutlineColor from the vert pass above.
				return i.color;
			}
			ENDCG
		}
		// Standard forward render.
		UsePass "Standard/FORWARD"
	}
	
	Fallback Off
}