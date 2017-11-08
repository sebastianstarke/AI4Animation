// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

#ifndef __DEPTH_OF_FIELD__
#define __DEPTH_OF_FIELD__

#include "UnityCG.cginc"
#include "Common.cginc"
#include "DiskKernels.cginc"

#define PREFILTER_LUMA_WEIGHT 1

sampler2D_float _CameraDepthTexture;
sampler2D_float _HistoryCoC;

// Camera parameters
float _Distance;
float _LensCoeff;  // f^2 / (N * (S1 - f) * film_width * 2)
float _MaxCoC;
float _RcpMaxCoC;
float _RcpAspect;

struct VaryingsDOF
{
    float4 pos : SV_POSITION;
    half2 uv : TEXCOORD0;
    half2 uvAlt : TEXCOORD1;
};

// Common vertex shader with single pass stereo rendering support
VaryingsDOF VertDOF(AttributesDefault v)
{
    half2 uvAlt = v.texcoord;
#if UNITY_UV_STARTS_AT_TOP
    if (_MainTex_TexelSize.y < 0.0) uvAlt.y = 1.0 - uvAlt.y;
#endif

    VaryingsDOF o;
#if defined(UNITY_SINGLE_PASS_STEREO)
    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv = UnityStereoScreenSpaceUVAdjust(v.texcoord, _MainTex_ST);
    o.uvAlt = UnityStereoScreenSpaceUVAdjust(uvAlt, _MainTex_ST);
#else
    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv = v.texcoord;
    o.uvAlt = uvAlt;
#endif

    return o;
}

// Downsampling, prefiltering and CoC calculation
half4 FragPrefilter(VaryingsDOF i) : SV_Target
{
    float3 duv = _MainTex_TexelSize.xyx * float3(0.5, 0.5, -0.5);

    // Sample source colors.
    half3 c0 = tex2D(_MainTex, i.uv - duv.xy).rgb;
    half3 c1 = tex2D(_MainTex, i.uv - duv.zy).rgb;
    half3 c2 = tex2D(_MainTex, i.uv + duv.zy).rgb;
    half3 c3 = tex2D(_MainTex, i.uv + duv.xy).rgb;

    // Sample linear depths.
    float d0 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt - duv.xy));
    float d1 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt - duv.zy));
    float d2 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt + duv.zy));
    float d3 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt + duv.xy));
    float4 depths = float4(d0, d1, d2, d3);

    // Calculate the radiuses of CoCs at these sample points.
    float4 cocs = (depths - _Distance) * _LensCoeff / depths;
    cocs = clamp(cocs, -_MaxCoC, _MaxCoC);

    // Premultiply CoC to reduce background bleeding.
    float4 weights = saturate(abs(cocs) * _RcpMaxCoC);

#if defined(PREFILTER_LUMA_WEIGHT)
    // Apply luma weights to reduce flickering.
    // References:
    //   http://gpuopen.com/optimized-reversible-tonemapper-for-resolve/
    //   http://graphicrants.blogspot.fr/2013/12/tone-mapping.html
    weights.x *= 1.0 / (Max3(c0) + 1.0);
    weights.y *= 1.0 / (Max3(c1) + 1.0);
    weights.z *= 1.0 / (Max3(c2) + 1.0);
    weights.w *= 1.0 / (Max3(c3) + 1.0);
#endif

    // Weighted average of the color samples
    half3 avg = c0 * weights.x + c1 * weights.y + c2 * weights.z + c3 * weights.w;
    avg /= dot(weights, 1.0);

    // Output CoC = average of CoCs
    half coc = dot(cocs, 0.25);

#if defined(UNITY_COLORSPACE_GAMMA)
    avg = GammaToLinearSpace(avg);
#endif

    return half4(avg, coc);
}

// Very simple temporal antialiasing on CoC to reduce jitter (mostly visible on the front plane)
struct Output
{
    half4 base : SV_Target0;
    half4 history : SV_Target1;
};

Output FragAntialiasCoC(VaryingsDOF i)
{
    half4 base = tex2D(_MainTex, i.uv);
    half hCoC = tex2D(_HistoryCoC, i.uv).r;
    half CoC = base.a;
    half nCoC = (hCoC + CoC) / 2.0; // TODO: Smarter CoC AA

    Output output;
    output.base = half4(base.rgb, nCoC);
    output.history = nCoC.xxxx;
    return output;
}

// CoC history clearing
half4 FragClearCoCHistory(VaryingsDOF i) : SV_Target
{
    return tex2D(_MainTex, i.uv).aaaa;
}

// Bokeh filter with disk-shaped kernels
half4 FragBlur(VaryingsDOF i) : SV_Target
{
    half4 samp0 = tex2D(_MainTex, i.uv);

    half4 bgAcc = 0.0; // Background: far field bokeh
    half4 fgAcc = 0.0; // Foreground: near field bokeh

    UNITY_LOOP for (int si = 0; si < kSampleCount; si++)
    {
        float2 disp = kDiskKernel[si] * _MaxCoC;
        float dist = length(disp);

        float2 duv = float2(disp.x * _RcpAspect, disp.y);
        half4 samp = tex2D(_MainTex, i.uv + duv);

        // BG: Compare CoC of the current sample and the center sample
        // and select smaller one.
        half bgCoC = max(min(samp0.a, samp.a), 0.0);

        // Compare the CoC to the sample distance.
        // Add a small margin to smooth out.
        half bgWeight = saturate((bgCoC - dist + 0.005) / 0.01);
        half fgWeight = saturate((-samp.a - dist + 0.005) / 0.01);

        // Accumulation
        bgAcc += half4(samp.rgb, 1.0) * bgWeight;
        fgAcc += half4(samp.rgb, 1.0) * fgWeight;
    }

    // Get the weighted average.
    bgAcc.rgb /= bgAcc.a + (bgAcc.a == 0.0); // zero-div guard
    fgAcc.rgb /= fgAcc.a + (fgAcc.a == 0.0);

    // BG: Calculate the alpha value only based on the center CoC.
    // This is a rather aggressive approximation but provides stable results.
    bgAcc.a = smoothstep(_MainTex_TexelSize.y, _MainTex_TexelSize.y * 2.0, samp0.a);

    // FG: Normalize the total of the weights.
    fgAcc.a *= UNITY_PI / kSampleCount;

    // Alpha premultiplying
    half3 rgb = 0.0;
    rgb = lerp(rgb, bgAcc.rgb, saturate(bgAcc.a));
    rgb = lerp(rgb, fgAcc.rgb, saturate(fgAcc.a));

    // Combined alpha value
    half alpha = (1.0 - saturate(bgAcc.a)) * (1.0 - saturate(fgAcc.a));

    return half4(rgb, alpha);
}

#endif // __DEPTH_OF_FIELD__
