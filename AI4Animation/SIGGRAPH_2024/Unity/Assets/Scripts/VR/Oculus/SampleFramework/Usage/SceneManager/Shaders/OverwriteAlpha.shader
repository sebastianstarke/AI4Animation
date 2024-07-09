// clang-format off
Shader "Unlit/OverwriteAlpha"
{
    Properties
    {
      _Alpha("Alpha", Range (0.0, 1.0)) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Off
        ZTest Always
        ZWrite Off
        // we want to keep the color buffer as is, but override the alpha
        Blend Zero One, One Zero 

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                // Push the z to the far plane
                //o.vertex.z = 1.0 - UNITY_NEAR_CLIP_VALUE;
                return o;
            }

            float _Alpha;

            fixed4 frag (v2f i) : SV_Target
            {
                // apply fog
                return float4(0,0,0, _Alpha);
            }
            ENDCG
        }
    }
}
