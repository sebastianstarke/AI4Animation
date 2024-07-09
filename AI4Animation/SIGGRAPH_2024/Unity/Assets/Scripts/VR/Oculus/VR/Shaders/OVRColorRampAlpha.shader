Shader "OVRColorRampAlpa" {
	Properties {
		_Color ("Main Color", Color ) = (1,1,1,1)
		_MainTex ("Diffuse (RGB) AlphaMask (A)", 2D) = "white" {}
		_ColorRamp ("Color Ramp (A)", 2D) = "white" {}
		_ColorRampOffset ("Color Ramp Offset", Range(0,1)) = 0.0
	}
	
Category {
	Tags {"Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent"}
	Blend SrcAlpha OneMinusSrcAlpha
	AlphaTest Greater .01
	ColorMask RGB
	Cull Off Lighting Off ZWrite Off Fog { Color (0,0,0,0) }
	
    SubShader {
        Pass {
            CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag
			
			#include "UnityCG.cginc"

			sampler2D _MainTex;
			sampler2D _ColorRamp;
			float _ColorRampOffset;
			fixed4 _Color;
			
			struct appdata_t {
				float4 vertex : POSITION;
				fixed4 color : COLOR;
				float2 texcoord : TEXCOORD0;
			};
			
			struct v2f {
				float4 vertex : SV_POSITION;
				fixed4 color : COLOR;
				float2 texcoord : TEXCOORD0;
			};
			
			float4 _MainTex_ST;
			
            v2f vert(appdata_t v) {
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.color = v.color;
				o.texcoord = TRANSFORM_TEX(v.texcoord,_MainTex);
				return o;
            }

			fixed4 frag(v2f i) : SV_Target {
				float4 texel = tex2D(_MainTex, i.texcoord);
				float2 colorIndex = float2( texel.x, _ColorRampOffset );
				float4 outColor = tex2D(_ColorRamp, colorIndex) * _Color;
				outColor.a = texel.a;
				return outColor;
			}
            ENDCG
        }
    }
}
} // Category