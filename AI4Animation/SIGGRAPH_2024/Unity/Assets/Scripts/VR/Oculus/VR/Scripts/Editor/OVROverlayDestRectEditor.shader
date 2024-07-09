/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

Shader "Unlit/OVROverlayDestRectEditor"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_SrcRectLeft("SrcRectLeft", Vector) = (0,0,1,1)
		_SrcRectRight("SrcRectRight", Vector) = (0,0,1,1)
		_DestRectLeft ("DestRectLeft", Vector) = (0,0,1,1)
		_DestRectRight("DestRectRight", Vector) = (0,0,1,1)
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
			float4 _DestRectLeft;
			float4 _DestRectRight;

			fixed4 _BackgroundColor;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				// Add padding
				o.uv = (o.uv - 0.5) * (256.0 + 8.0) / (256.0) + 0.5;

				// left
				o.leftDragX.x = _DestRectLeft.x;
				o.leftDragY.x = _DestRectLeft.y + _DestRectLeft.w * 0.5;
				// right
				o.leftDragX.y = _DestRectLeft.x + _DestRectLeft.z;
				o.leftDragY.y = _DestRectLeft.y + _DestRectLeft.w * 0.5;
				// top
				o.leftDragX.z = _DestRectLeft.x + _DestRectLeft.z * 0.5;
				o.leftDragY.z = _DestRectLeft.y;
				// bottom
				o.leftDragX.w = _DestRectLeft.x + _DestRectLeft.z * 0.5;
				o.leftDragY.w = _DestRectLeft.y + _DestRectLeft.w;
				// right
				o.rightDragX.x = _DestRectRight.x;
				o.rightDragY.x = _DestRectRight.y + _DestRectRight.w * 0.5;
				// right
				o.rightDragX.y = _DestRectRight.x + _DestRectRight.z;
				o.rightDragY.y = _DestRectRight.y + _DestRectRight.w * 0.5;
				// top
				o.rightDragX.z = _DestRectRight.x + _DestRectRight.z * 0.5;
				o.rightDragY.z = _DestRectRight.y;
				// bottom
				o.rightDragX.w = _DestRectRight.x + _DestRectRight.z * 0.5;
				o.rightDragY.w = _DestRectRight.y + _DestRectRight.w;

				return o;
			}

			float onDrag(float2 uv, float x, float y)
			{
				const float pixelSize = 6;
				return abs(uv.x - x) < ((pixelSize / 2) / 128.0) && abs(uv.y - y) < ((pixelSize / 2) / 128.0);
			}

			float onLine(float2 uv, float4 rect)
			{
				return
					(abs(uv.x - rect.x) < (1 / 128.0) && uv.y >= rect.y && uv.y <= rect.y + rect.w) ||
					(abs(uv.x - rect.x - rect.z) < (1 / 128.0) && uv.y >= rect.y && uv.y <= rect.y + rect.w) ||
					(abs(uv.y - rect.y) < (1 / 128.0) && uv.x >= rect.x && uv.x <= rect.x + rect.z) ||
					(abs(uv.y - rect.y - rect.w) < (1 / 128.0) && uv.x >= rect.x && uv.x <= rect.x + rect.z);
			}

			float checkerboard(float2 uv)
			{
				float x = floor(uv.x * (16 + 2));
				float y = floor(uv.y * 8);

				return 2 * ((x + y) / 2.0 - floor((x + y) / 2.0));
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				float isLeftEye = i.uv < 0.5;
				float2 leftUV = float2(i.uv.x * (256.0 + 32.0) / 128.0, i.uv.y);
				float2 rightUV = float2(1 - ((1 - i.uv.x) * (256.0 + 32.0) / 128.0), i.uv.y);

				float2 uv = i.uv;
				float2 textureUV = i.uv;
				if (isLeftEye)
				{
					uv = (leftUV - _DestRectLeft.xy) / _DestRectLeft.zw;
					textureUV = uv * _SrcRectLeft.zw + _SrcRectLeft.xy;
				}
				else
				{
					uv = (rightUV - _DestRectRight.xy) / _DestRectRight.zw;
					textureUV = uv * _SrcRectRight.zw + _SrcRectRight.xy;
				}

				// sample the texture
				fixed4 col = tex2D(_MainTex, float2(textureUV.x, 1 - textureUV.y));
				
				if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
				{
					col.a = 0;
				}

				col.rgb = lerp(0.41 - 0.13 * checkerboard(i.uv), col.rgb, col.a);

				if (i.uv.x < 0 || i.uv.x > 1 || i.uv.y < 0 || i.uv.y > 1 || abs(i.uv.x - 0.5) < (14 / 256.0))
				{
					col = _BackgroundColor;
				}

				// now draw clipping objects
				float left = isLeftEye && (onLine(leftUV, _DestRectLeft) ||
					onDrag(leftUV, i.leftDragX.x, i.leftDragY.x) ||
					onDrag(leftUV, i.leftDragX.y, i.leftDragY.y) ||
					onDrag(leftUV, i.leftDragX.z, i.leftDragY.z) ||
					onDrag(leftUV, i.leftDragX.w, i.leftDragY.w));

				float right = (!isLeftEye) && (onLine(rightUV, _DestRectRight) ||
					onDrag(rightUV, i.rightDragX.x, i.rightDragY.x) ||
					onDrag(rightUV, i.rightDragX.y, i.rightDragY.y) ||
					onDrag(rightUV, i.rightDragX.z, i.rightDragY.z) ||
					onDrag(rightUV, i.rightDragX.w, i.rightDragY.w));

				return lerp(col, fixed4(left, right, 0, 1), left || right);
			}
			ENDCG
		}
	}
}
