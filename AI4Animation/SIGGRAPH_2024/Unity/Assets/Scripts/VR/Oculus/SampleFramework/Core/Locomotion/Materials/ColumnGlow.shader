/************************************************************************************

Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

See SampleFramework license.txt for license terms.  Unless required by applicable law
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific
language governing permissions and limitations under the license.

************************************************************************************/

Shader "Custom/ColumnGlow"
{
	Properties
	{
		_TintColor ("Tint Color", Color) = (0.5,0.5,0.5,0.5)
		_Thickness("Thickness", Range(0, 1)) = 0.5
		_FadeStart("Fade Start", Range(0, 1)) = 0.5
		_FadeEnd("Fade End", Range(-1, 1)) = 0.5
		_Intensity("Intensity", Range(0, 1)) = 0.5

	}
	SubShader
	{
		Tags { "RenderType"="Transparent" "IgnoreProjector"="True"  "Queue"="Transparent" }
		Blend SrcAlpha One
		Cull Off Lighting Off ZWrite On

		LOD 0

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			// make fog work


			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float4 normal : NORMAL;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float3 normal : NORMAL;
				float3 origPosition : POSITION1;
				float3 eyeDir : DIRECTION;
			};

			fixed4 _TintColor;
			float _Thickness;
			float _FadeStart;
			float _FadeEnd;
			float _Intensity;



			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.normal = normalize(mul(UNITY_MATRIX_IT_MV,v.normal).xyz);
				o.origPosition = v.vertex;
				o.eyeDir = -normalize(mul(UNITY_MATRIX_MV, v.vertex).xyz);
				//o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				return o;
			}

			fixed4 frag (v2f i) : SV_Target
			{
				float d = 	dot(i.normal,i.eyeDir);
				float p = smoothstep (0,_Thickness,dot(i.normal,i.eyeDir)) * smoothstep(_FadeStart,_FadeEnd,i.origPosition.y);
				return float4((p * _TintColor * _Intensity).xyz,p) ;
			}
			ENDCG
		}
	}
}
