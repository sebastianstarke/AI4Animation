Shader "Oculus/OVRMRCameraFrameLit" {
	Properties{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}
		_DepthTex("Depth (cm)", 2D) = "black" {}
		_InconfidenceTex("Inconfidence (0-100)", 2D) = "black" {}
		_Visible("Visible", Range(0.0,1.0)) = 1.0
	}
	SubShader{
		Tags{ "Queue" = "Transparent" "RenderType" = "Transparent" }
		LOD 200

		CGPROGRAM
		#pragma surface surf Lambert alpha:fade
		#pragma target 3.0

		#include "OVRMRChromaKey.cginc"

		#define TEST_ENVIRONMENT			0

		sampler2D _MainTex;
		sampler2D _DepthTex;
		sampler2D _MaskTex;

		float4 _TextureDimension;		// (w, h, 1/w, 1/h)
		float4 _TextureWorldSize;	  // (width_in_meter, height_in_meter, 0, 0)

		float _SmoothFactor;
		float _DepthVariationClamp;

		float _CullingDistance;

		struct Input {
		#if TEST_ENVIRONMENT
			float2 uv_MainTex;
		#endif
			float4 screenPos;
		};

		fixed4 _Color;
		fixed  _Visible;
		float4 _FlipParams;		// (flip_h, flip_v, 0, 0)

		fixed GetMask(float2 uv)
		{
			return tex2D(_MaskTex, uv).r;
		}

		fixed4 GetCameraColor(float2 colorUV)
		{
			fixed4 c = tex2D(_MainTex, colorUV) * _Color;
			return c;
		}

		float GetDepth(float2 uv)
		{
			float depth = tex2D(_DepthTex, uv).x * 1.0 / 100;
			return depth;
		}

		float3 GetNormal(float2 uv)
		{
			float dz_x = GetDepth(uv + float2(_TextureDimension.z, 0)) - GetDepth(uv - float2(_TextureDimension.z, 0));
			float dz_y = GetDepth(uv + float2(0, _TextureDimension.w)) - GetDepth(uv - float2(0, _TextureDimension.w));
			dz_x = clamp(dz_x, -_DepthVariationClamp, _DepthVariationClamp);
			dz_y = clamp(dz_y, -_DepthVariationClamp, _DepthVariationClamp);
			//float dist = 0.01;
			//float3 normal = cross(float3(dist, 0, dz_x), float3(0, dist, dz_y));
			float3 normal = cross(float3(_TextureWorldSize.x * _TextureDimension.z * 2.0 * _SmoothFactor, 0, dz_x), float3(0, _TextureWorldSize.y * _TextureDimension.w * 2.0 * _SmoothFactor, dz_y));
			normal = normalize(normal);
			return normal;
		}

		void surf(Input IN, inout SurfaceOutput o) {
	#if TEST_ENVIRONMENT
			float2 colorUV = float2(IN.uv_MainTex.x, IN.uv_MainTex.y);
	#else
			float2 screenUV = IN.screenPos.xy / IN.screenPos.w;
			float2 colorUV = float2(screenUV.x, 1.0 - screenUV.y);
	#endif
			if (_FlipParams.x > 0.0)
			{
				colorUV.x = 1.0 - colorUV.x;
			}
			if (_FlipParams.y > 0.0)
			{
				colorUV.y = 1.0 - colorUV.y;
			}
			float mask = GetMask(colorUV);
			if (mask == 0.0)
			{
				discard;
			}
			float4 col = GetColorAfterChromaKey(colorUV, _TextureDimension.zw, 1.0);
			if (col.a < 0.0)
			{
				discard;
			}
			float depth = GetDepth(colorUV);
			if (depth > _CullingDistance)
			{
				discard;
			}
			float3 normal = GetNormal(colorUV);
			o.Albedo = col.rgb;
			o.Normal = normal;
			o.Alpha = col.a *_Visible;
		}
		ENDCG
	}
	FallBack "Alpha-Diffuse"
}
