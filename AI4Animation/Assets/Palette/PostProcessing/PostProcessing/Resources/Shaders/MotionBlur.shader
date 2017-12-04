Shader "Hidden/Post FX/Motion Blur"
{
    CGINCLUDE

        #pragma target 3.0

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0) Velocity texture setup
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertDefault
                #pragma fragment FragVelocitySetup

            ENDCG
        }

        // (1) TileMax filter (4 pixels width with normalization)
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertDefault
                #pragma fragment FragTileMax4

            ENDCG
        }

        //  (2) TileMax filter (2 pixels width)
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertDefault
                #pragma fragment FragTileMax2

            ENDCG
        }

        // (3) TileMax filter (variable width)
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertDefault
                #pragma fragment FragTileMaxV

            ENDCG
        }

        // (4) NeighborMax filter
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertDefault
                #pragma fragment FragNeighborMax

            ENDCG
        }

        // (5) Reconstruction filter
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertMultitex
                #pragma fragment FragReconstruction

            ENDCG
        }

        // (6) Reconstruction filter (unrolled)
        Pass
        {
            CGPROGRAM

                #define UNROLL_LOOP_COUNT 2
                #include "MotionBlur.cginc"
                #pragma vertex VertMultitex
                #pragma fragment FragReconstruction

            ENDCG
        }

        // (7) Frame compression
        Pass
        {
            CGPROGRAM

                #pragma multi_compile __ UNITY_COLORSPACE_GAMMA
                #include "MotionBlur.cginc"
                #pragma vertex VertFrameCompress
                #pragma fragment FragFrameCompress

            ENDCG
        }

        // (8) Frame blending
        Pass
        {
            CGPROGRAM

                #pragma multi_compile __ UNITY_COLORSPACE_GAMMA
                #include "MotionBlur.cginc"
                #pragma vertex VertMultitex
                #pragma fragment FragFrameBlending

            ENDCG
        }

        // (9) Frame blending (without chroma subsampling)
        Pass
        {
            CGPROGRAM

                #include "MotionBlur.cginc"
                #pragma vertex VertMultitex
                #pragma fragment FragFrameBlendingRaw

            ENDCG
        }
    }
}
