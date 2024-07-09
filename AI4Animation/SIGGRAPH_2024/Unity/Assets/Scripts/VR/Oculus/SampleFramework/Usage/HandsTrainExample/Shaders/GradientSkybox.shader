/************************************************************************************

Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.  

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

Shader "Oculus Sample/Procedural Gradient Skybox"
{
  Properties
  {
    _TopColor ("Top Color", Color) = (1, 1, 1, 0)
    _HorizonColor ("Horizon Color", Color) = (1, 1, 1, 0)
    _BottomColor ("Bottom Color", Color) = (1, 1, 1, 0)
    _TopExponent ("Top Exponent", Float) = 0.5
    _BottomExponent ("Bottom Exponent", Float) = 0.5
    _AmplFactor ("Amplification", Float) = 1.0
  }
  SubShader
  {
    Tags{"RenderType" ="Background" "Queue" = "Background"}
    ZWrite Off Cull Off 
    Fog { Mode Off }
    LOD 100

    Pass
    {
      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag
    
      #include "UnityCG.cginc"

      struct vertIn
      {
        float4 vertex : POSITION;
        float3 uv : TEXCOORD0;
      };

      struct vertOut
      {
        float4 vertex : SV_POSITION;
        float3 uv: TEXCOORD0;
      };
      
      vertOut vert (vertIn v)
      {
        vertOut o;
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = v.uv;
        return o;
      }

      half _TopExponent;
      half _BottomExponent;
      fixed4 _TopColor;
      fixed4 _HorizonColor;
      fixed4 _BottomColor;
      half _AmplFactor;
      
      fixed4 frag (vertOut i) : SV_Target
      {
        float interpUv = normalize (i.uv).y;
        // top goes from 0->1 going down toward horizon
        float topLerp = 1.0f - pow (min (1.0f, 1.0f - interpUv), _TopExponent);
        // bottom goes from 0->1 going up toward horizon
        float bottomLerp = 1.0f - pow (min (1.0f, 1.0f + interpUv), _BottomExponent);
        // last lerp param is horizon. all must add up to 1.0
        float horizonLerp = 1.0f - topLerp - bottomLerp;
        return (_TopColor * topLerp + _HorizonColor * horizonLerp + _BottomColor * bottomLerp) *
          _AmplFactor;
      }

      ENDCG
    }
  }
}
