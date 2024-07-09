/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

Shader "Unlit/OVROverlaySrcRectEditor"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_SrcRectLeft ("SrcRectLeft", Vector) = (0,0,1,1)
		_SrcRectRight("SrcRectRight", Vector) = (0,0,1,1)
		_BackgroundColor("Background Color", Color) = (0.225, 0.225, 0.225, 1)
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float4 leftDragX : TEXCOORD1;
				float4 leftDragY : TEXCOORD2;
				float4 rightDragX : TEXCOORD3;
				float4 rightDragY : TEXCOORD4;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;

			float4 _SrcRectLeft;
			float4 _SrcRectRight;

			fixed4 _BackgroundColor;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				// Add padding
				o.uv = (o.uv - 0.5) * (256.0 + 8.0) / (256.0) + 0.5;

				// left
				o.leftDragX.x = _SrcRectLeft.x;
				o.leftDragY.x = _SrcRectLeft.y + _SrcRectLeft.w * 0.5;
				// right
				o.leftDragX.y = _SrcRectLeft.x + _SrcRectLeft.z;
				o.leftDragY.y = _SrcRectLeft.y + _SrcRectLeft.w * 0.5;
				// top
				o.leftDragX.z = _SrcRectLeft.x + _SrcRectLeft.z * 0.5;
				o.leftDragY.z = _SrcRectLeft.y;
				// bottom
				o.leftDragX.w = _SrcRectLeft.x + _SrcRectLeft.z * 0.5;
				o.leftDragY.w = _SrcRectLeft.y + _SrcRectLeft.w;
				// right
				o.rightDragX.x = _SrcRectRight.x;
				o.rightDragY.x = _SrcRectRight.y + _SrcRectRight.w * 0.5;
				// right
				o.rightDragX.y = _SrcRectRight.x + _SrcRectRight.z;
				o.rightDragY.y = _SrcRectRight.y + _SrcRectRight.w * 0.5;
				// top
				o.rightDragX.z = _SrcRectRight.x + _SrcRectRight.z * 0.5;
				o.rightDragY.z = _SrcRectRight.y;
				// bottom
				o.rightDragX.w = _SrcRectRight.x + _SrcRectRight.z * 0.5;
				o.rightDragY.w = _SrcRectRight.y + _SrcRectRight.w;

				return o;
			}

			float onDrag(float2 uv, float x, float y)
			{
				const float pixelSize = 6;
				return abs(uv.x - x) < ((pixelSize / 2) / 256.0) && abs(uv.y - y) < ((pixelSize / 2) / 128.0);
			}

			float onLine(float2 uv, float4 rect)
			{
				return
					(abs(uv.x - rect.x) < (1 / 256.0) && uv.y >= rect.y && uv.y <= rect.y + rect.w) ||
					(abs(uv.x - rect.x - rect.z) < (1 / 256.0) && uv.y >= rect.y && uv.y <= rect.y + rect.w) ||
					(abs(uv.y - rect.y) < (1 / 128.0) && uv.x >= rect.x && uv.x <= rect.x + rect.z) ||
					(abs(uv.y - rect.y - rect.w) < (1 / 128.0) && uv.x >= rect.x && uv.x <= rect.x + rect.z);
			}

			float checkerboard(float2 uv)
			{
				float x = floor(uv.x * (16));
				float y = floor(uv.y * 8);

				return 2 * ((x + y) / 2.0 - floor((x + y) / 2.0));
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				// sample the texture
				fixed4 col = tex2D(_MainTex, i.uv);

				col.rgb = lerp(0.41 - 0.13 * checkerboard(i.uv), col.rgb, col.a);

				if (i.uv.x < 0 || i.uv.x > 1 || i.uv.y < 0 || i.uv.y > 1)
				{
					col = _BackgroundColor;
				}

				float2 uv = i.uv.xy;

				// now draw clipping objects
				float left = onLine(uv, _SrcRectLeft) ||
					onDrag(uv, i.leftDragX.x, i.leftDragY.x) ||
					onDrag(uv, i.leftDragX.y, i.leftDragY.y) ||
					onDrag(uv, i.leftDragX.z, i.leftDragY.z) ||
					onDrag(uv, i.leftDragX.w, i.leftDragY.w);

				float right = onLine(uv, _SrcRectRight) ||
					onDrag(uv, i.rightDragX.x, i.rightDragY.x) ||
					onDrag(uv, i.rightDragX.y, i.rightDragY.y) ||
					onDrag(uv, i.rightDragX.z, i.rightDragY.z) ||
					onDrag(uv, i.rightDragX.w, i.rightDragY.w);

				return lerp(col, fixed4(left, right, 0, 1), left || right);
			}
			ENDCG
		}
	}
}
