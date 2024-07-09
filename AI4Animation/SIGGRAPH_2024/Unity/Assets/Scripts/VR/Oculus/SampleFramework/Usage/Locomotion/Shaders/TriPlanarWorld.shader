/************************************************************************************

Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

See SampleFramework license.txt for license terms.  Unless required by applicable law
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific
language governing permissions and limitations under the license.

************************************************************************************/

Shader "Custom/TriPlanarWorld" {
	Properties {
		_Color("Color", Color) = (1,1,1,1)
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
	}
	SubShader {
		Tags { "RenderType"="Opaque" }
		LOD 200

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows vertex:vert

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0

		sampler2D _MainTex;

		struct Input {
			float2 uv_MainTex;
			float3 worldPos;
			float3 worldNormal;
			float4 vertColor : COLOR;
		};

		half _Glossiness;
		half _Metallic;
		fixed4 _Color;

		void vert(inout appdata_full v, out Input o) {
			UNITY_INITIALIZE_OUTPUT(Input, o);
			o.vertColor = v.color;
			o.vertColor = float4(1, 0, 1, 1);
		}

		void surf (Input IN, inout SurfaceOutputStandard o) {
			float3 absNormal = abs(IN.worldNormal);

			// exponentially scale the absNormal so that it strongly biases to a single cardinal axis.
			absNormal = normalize(pow(absNormal, 5));

			fixed4 x = tex2D(_MainTex, IN.worldPos.yz);
			fixed4 y = tex2D(_MainTex, IN.worldPos.xz);
			fixed4 z = tex2D(_MainTex, IN.worldPos.xy);

			fixed4 c = x * absNormal.x + y * absNormal.y + z * absNormal.z;
			c *= _Color * IN.vertColor;
			o.Albedo = c.rgb;
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
			o.Alpha = c.a;
			//o.Albedo = abs(IN.worldNormal);
		}
		ENDCG
	}
	FallBack "Diffuse"
}
