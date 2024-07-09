Shader "Scene/StylizedOcclusions" {
  Properties {
    _Color("Color", Color) = (1,1,1,1)
  }
  SubShader {
    Tags{"RenderType" = "Solid"}
    LOD 100
    Cull off
    ZWrite Off
    ZTest Always
    Pass {
      CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma multi_compile_fog
#include "UnityCG.cginc"

      struct appdata {
        float4 vertex : POSITION;
      };

      struct v2f {
        float4 vertex : SV_POSITION;
      };
      float4 _Color;

      v2f vert(appdata v) {
        v2f o;
        o.vertex = UnityObjectToClipPos(v.vertex);
        return o;
      }

      fixed4 frag(v2f i) : SV_Target {
        return _Color;
      }
      ENDCG
    }
  }
}
