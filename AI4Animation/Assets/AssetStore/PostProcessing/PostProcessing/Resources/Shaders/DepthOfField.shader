Shader "Hidden/Post FX/Depth Of Field"
{
    Properties
    {
        _MainTex ("", 2D) = "black"
    }

    CGINCLUDE
        #pragma exclude_renderers d3d11_9x
        #pragma target 3.0
    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0) Downsampling, prefiltering & CoC
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragPrefilter
                #include "DepthOfField.cginc"
            ENDCG
        }

        // (1-4) Bokeh filter with disk-shaped kernels
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_SMALL
                #include "DepthOfField.cginc"
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_MEDIUM
                #include "DepthOfField.cginc"
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_LARGE
                #include "DepthOfField.cginc"
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_VERYLARGE
                #include "DepthOfField.cginc"
            ENDCG
        }

        // (5) CoC antialiasing
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragAntialiasCoC
                #include "DepthOfField.cginc"
            ENDCG
        }

        // (6) CoC history clearing
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragClearCoCHistory
                #include "DepthOfField.cginc"
            ENDCG
        }
    }

    FallBack Off
}
