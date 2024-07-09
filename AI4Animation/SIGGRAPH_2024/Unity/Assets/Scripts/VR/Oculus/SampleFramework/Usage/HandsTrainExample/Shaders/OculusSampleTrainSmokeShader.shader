/************************************************************************************

Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

See SampleFramework license.txt for license terms.  Unless required by applicable law
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific
language governing permissions and limitations under the license.

************************************************************************************/

Shader "Oculus Sample/Train Smoke"
{
  Properties
  {
    _MainTex ("Texture", 2D) = "white" {}
    _TintColor ("Tint Color", Color) = (0.5, 0.5, 0.5, 0.5)
  }
    SubShader
    {
      Tags{"Queue" = "Transparent" "RenderType" = "Transparent"
        "IgnoreProjector" = "True" "PreviewType" = "Plane"} 
      LOD 100
      Blend SrcAlpha OneMinusSrcAlpha
      Cull Off
      Lighting Off
      ZWrite Off

      Pass
      {
        CGPROGRAM
        #pragma vertex vert
        #pragma fragment frag
       // In the future, we can play with particle instancing, see:
       // see https://docs.unity3d.com/Manual/PartSysInstancing.html
       // for now we cannot, because our minimum supported Unity version doesn't support it
        #include "UnityCG.cginc"

        struct vertIn
        {
          float2 uv : TEXCOORD0;
          float4 vertex : POSITION;
          fixed4 color : COLOR;
          UNITY_VERTEX_INPUT_INSTANCE_ID
        };

        struct vertOut
        {
          float2 uv : TEXCOORD0;
          float4 clipPos : SV_POSITION;
          fixed4 color : COLOR;
          UNITY_VERTEX_OUTPUT_STEREO
        };

        sampler2D _MainTex;
        float4 _MainTex_ST;
        fixed4 _TintColor;

        vertOut vert (vertIn vIn)
        {
          vertOut vOut;
          UNITY_SETUP_INSTANCE_ID(vIn);
          UNITY_INITIALIZE_OUTPUT(vertOut, vOut);
          UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(vOut);

          vOut.clipPos = UnityObjectToClipPos(vIn.vertex);
          vOut.uv = TRANSFORM_TEX(vIn.uv, _MainTex);
          vOut.color = vIn.color * _TintColor;
          return vOut;
        }

        fixed4 frag (vertOut vOut) : SV_Target
        {
          fixed4 fragColor = 2.0f * vOut.color * tex2D(_MainTex, vOut.uv);
          // don't amplify opacity
          fragColor.a = saturate(fragColor.a);
          return fragColor;
        }

        ENDCG
      }
  }
}
