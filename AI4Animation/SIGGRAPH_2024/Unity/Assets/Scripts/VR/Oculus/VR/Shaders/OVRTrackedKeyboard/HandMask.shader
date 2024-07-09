Shader "Unlit/HandMask"
{
    Properties
    {
		_Color("Color", Color) = (1, 1, 1, 1)
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType"="Transparent" }
        LOD 200
		Blend One Zero, One Zero

        CGPROGRAM
		#pragma surface surf NoLighting keepalpha

		#pragma target 3.0

		sampler2D _MainTex;

		fixed4 LightingNoLighting(SurfaceOutput s, fixed3 lightDir, fixed atten) {
          return fixed4(s.Albedo, s.Alpha);
        }

        struct Input {
            float2 uv_MainTex;
        };
        fixed4 _Color;

        void surf(Input IN, inout SurfaceOutput o) {
          o.Albedo = 0;
          float alpha = tex2D(_MainTex, IN.uv_MainTex);
          o.Alpha = 0;
        }
        ENDCG
    }
}
