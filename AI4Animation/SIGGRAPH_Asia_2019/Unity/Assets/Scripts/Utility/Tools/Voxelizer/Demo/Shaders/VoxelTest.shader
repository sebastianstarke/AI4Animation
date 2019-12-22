Shader"Voxelizer/Demo/VoxelTest" {

	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0

        _Range ("Range", Vector) = (-10, 10, 0, 0)
        _Threshold ("Threshold", Range(0.0, 1.0)) = 0.0
	}

	SubShader {
		Tags { "RenderType"="Opaque" }
		LOD 200

		CGPROGRAM

		#pragma surface surf Standard fullforwardshadows vertex:vert

		#pragma target 3.0

		sampler2D _MainTex;

		struct Input {
			float2 uv_MainTex;
		};

		half _Glossiness;
		half _Metallic;
		fixed4 _Color;

        float2 _Range;
        half _Threshold;

		UNITY_INSTANCING_BUFFER_START(Props)
		UNITY_INSTANCING_BUFFER_END(Props)

		void vert (inout appdata_full v) {
            float3 world = mul(unity_ObjectToWorld, float4(v.tangent.xyz, 1)).xyz;
            float t = step(world.z, lerp(_Range.x, _Range.y, _Threshold));
			v.vertex.xyz = lerp(v.vertex.xyz, v.tangent.xyz, t);
		}

		void surf (Input IN, inout SurfaceOutputStandard o) {
			fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
			o.Albedo = c.rgb;
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
			o.Alpha = c.a;
		}

		ENDCG
	}

	FallBack "Diffuse"
}
