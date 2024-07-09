/************************************************************************************

Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

See SampleFramework license.txt for license terms.  Unless required by applicable law
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific
language governing permissions and limitations under the license.

************************************************************************************/

Shader "Oculus Sample/Xing Light"
{
    Properties
    {
      _MainTex("Base (RGB) Alpha (A)", 2D) = "white" {}
      _Color("Tint Color", Color) = (0.5, 0.5, 0.5, 0.5)
    }
    SubShader
    {
      Tags{"Queue" = "Transparent" "RenderType" =  "Transparent"
        "IgnoreProjector" = "True" "PreviewType" = "Plane"}
      LOD 100
      Blend SrcAlpha OneMinusSrcAlpha
      Cull Back
      Lighting Off
      ZWrite Off

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
          float4 vertex : SV_POSITION;
          float2 uv : TEXCOORD0;
        };

        fixed4 _Color;
        sampler2D _MainTex;
        float4 _MainTex_ST;

        v2f vert (appdata v)
        {
          v2f o;
          o.vertex = UnityObjectToClipPos(v.vertex);
          o.uv = TRANSFORM_TEX(v.uv, _MainTex);
          return o;
        }

        fixed4 frag (v2f i) : SV_Target
        {
          return _Color * tex2D(_MainTex, i.uv);
        }
        ENDCG
      }
    }
}
