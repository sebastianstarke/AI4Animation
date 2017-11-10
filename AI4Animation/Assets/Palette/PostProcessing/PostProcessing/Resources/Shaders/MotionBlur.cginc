// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

#ifndef __MOTION_BLUR__
#define __MOTION_BLUR__

#include "UnityCG.cginc"
#include "Common.cginc"

// Camera depth texture
sampler2D_float _CameraDepthTexture;

// Camera motion vectors texture
sampler2D_half _CameraMotionVectorsTexture;
float4 _CameraMotionVectorsTexture_TexelSize;

// Packed velocity texture (2/10/10/10)
sampler2D_half _VelocityTex;
float2 _VelocityTex_TexelSize;

// NeighborMax texture
sampler2D_half _NeighborMaxTex;
float2 _NeighborMaxTex_TexelSize;

// Velocity scale factor
float _VelocityScale;

// TileMax filter parameters
int _TileMaxLoop;
float2 _TileMaxOffs;

// Maximum blur radius (in pixels)
half _MaxBlurRadius;

// Filter parameters/coefficients
int _LoopCount;

// History buffer for frame blending
sampler2D _History1LumaTex;
sampler2D _History2LumaTex;
sampler2D _History3LumaTex;
sampler2D _History4LumaTex;

sampler2D _History1ChromaTex;
sampler2D _History2ChromaTex;
sampler2D _History3ChromaTex;
sampler2D _History4ChromaTex;

half _History1Weight;
half _History2Weight;
half _History3Weight;
half _History4Weight;

struct VaryingsMultitex
{
    float4 pos : SV_POSITION;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
};

VaryingsMultitex VertMultitex(AttributesDefault v)
{
    VaryingsMultitex o;
    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv0 = v.texcoord.xy;
    o.uv1 = v.texcoord.xy;

#if UNITY_UV_STARTS_AT_TOP
    if (_MainTex_TexelSize.y < 0.0)
        o.uv1.y = 1.0 - v.texcoord.y;
#endif

    return o;
}

// -----------------------------------------------------------------------------
// Prefilter

// Velocity texture setup
half4 FragVelocitySetup(VaryingsDefault i) : SV_Target
{
    // Sample the motion vector.
    float2 v = tex2D(_CameraMotionVectorsTexture, i.uv).rg;

    // Apply the exposure time.
    v *= _VelocityScale;

    // Halve the vector and convert it to the pixel space.
    v = v * 0.5 * _CameraMotionVectorsTexture_TexelSize.zw;

    // Clamp the vector with the maximum blur radius.
    float lv = length(v);
    v *= min(lv, _MaxBlurRadius) / max(lv, 1e-6);

    // Sample the depth of the pixel.
    float d = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv.xy);
    half z01 = LinearizeDepth(d);

    // Pack into 10/10/10/2 format.
    return half4((v / _MaxBlurRadius + 1.0) / 2.0, z01, 0.0);
}

// TileMax filter (4 pixels width with normalization)
half4 FragTileMax4(VaryingsDefault i) : SV_Target
{
    float4 d1 = _MainTex_TexelSize.xyxy * float4( 0.5, 0.5,  1.5, 1.5);
    float4 d2 = _MainTex_TexelSize.xyxy * float4(-0.5, 0.5, -1.5, 1.5);

    half2 v01 = tex2D(_MainTex, i.uv - d1.zw).rg; // -1.5, -1.5
    half2 v02 = tex2D(_MainTex, i.uv - d1.xw).rg; // -0.5, -1.5
    half2 v03 = tex2D(_MainTex, i.uv - d2.xw).rg; // +0.5, -1.5
    half2 v04 = tex2D(_MainTex, i.uv - d2.zw).rg; // +1.5, -1.5

    half2 v05 = tex2D(_MainTex, i.uv - d1.zy).rg; // -1.5, -0.5
    half2 v06 = tex2D(_MainTex, i.uv - d1.xy).rg; // -0.5, -0.5
    half2 v07 = tex2D(_MainTex, i.uv - d2.xy).rg; // +0.5, -0.5
    half2 v08 = tex2D(_MainTex, i.uv - d2.zy).rg; // +1.5, -0.5

    half2 v09 = tex2D(_MainTex, i.uv + d2.zy).rg; // -1.5, +0.5
    half2 v10 = tex2D(_MainTex, i.uv + d2.xy).rg; // -0.5, +0.5
    half2 v11 = tex2D(_MainTex, i.uv + d1.xy).rg; // +0.5, +0.5
    half2 v12 = tex2D(_MainTex, i.uv + d1.zy).rg; // +1.5, +0.5

    half2 v13 = tex2D(_MainTex, i.uv + d2.zw).rg; // -1.5, +1.5
    half2 v14 = tex2D(_MainTex, i.uv + d2.xw).rg; // -0.5, +1.5
    half2 v15 = tex2D(_MainTex, i.uv + d1.xw).rg; // +0.5, +1.5
    half2 v16 = tex2D(_MainTex, i.uv + d1.zw).rg; // +1.5, +1.5

    v01 = (v01 * 2.0 - 1.0) * _MaxBlurRadius;
    v02 = (v02 * 2.0 - 1.0) * _MaxBlurRadius;
    v03 = (v03 * 2.0 - 1.0) * _MaxBlurRadius;
    v04 = (v04 * 2.0 - 1.0) * _MaxBlurRadius;

    v05 = (v05 * 2.0 - 1.0) * _MaxBlurRadius;
    v06 = (v06 * 2.0 - 1.0) * _MaxBlurRadius;
    v07 = (v07 * 2.0 - 1.0) * _MaxBlurRadius;
    v08 = (v08 * 2.0 - 1.0) * _MaxBlurRadius;

    v09 = (v09 * 2.0 - 1.0) * _MaxBlurRadius;
    v10 = (v10 * 2.0 - 1.0) * _MaxBlurRadius;
    v11 = (v11 * 2.0 - 1.0) * _MaxBlurRadius;
    v12 = (v12 * 2.0 - 1.0) * _MaxBlurRadius;

    v13 = (v13 * 2.0 - 1.0) * _MaxBlurRadius;
    v14 = (v14 * 2.0 - 1.0) * _MaxBlurRadius;
    v15 = (v15 * 2.0 - 1.0) * _MaxBlurRadius;
    v16 = (v16 * 2.0 - 1.0) * _MaxBlurRadius;

    half2 va = MaxV(MaxV(MaxV(v01, v02), v03), v04);
    half2 vb = MaxV(MaxV(MaxV(v05, v06), v07), v08);
    half2 vc = MaxV(MaxV(MaxV(v09, v10), v11), v12);
    half2 vd = MaxV(MaxV(MaxV(v13, v14), v15), v16);

    half2 vo = MaxV(MaxV(MaxV(va, vb), vc), vd);

    return half4(vo, 0.0, 0.0);
}

// TileMax filter (2 pixels width)
half4 FragTileMax2(VaryingsDefault i) : SV_Target
{
    float4 d = _MainTex_TexelSize.xyxy * float4(-0.5, -0.5, 0.5, 0.5);

    half2 v1 = tex2D(_MainTex, i.uv + d.xy).rg;
    half2 v2 = tex2D(_MainTex, i.uv + d.zy).rg;
    half2 v3 = tex2D(_MainTex, i.uv + d.xw).rg;
    half2 v4 = tex2D(_MainTex, i.uv + d.zw).rg;

    half2 vo = MaxV(MaxV(MaxV(v1, v2), v3), v4);

    return half4(vo, 0.0, 0.0);
}

// TileMax filter (variable width)
half4 FragTileMaxV(VaryingsDefault i) : SV_Target
{
    float2 uv0 = i.uv + _MainTex_TexelSize.xy * _TileMaxOffs.xy;

    float2 du = float2(_MainTex_TexelSize.x, 0.0);
    float2 dv = float2(0, _MainTex_TexelSize.y);

    half2 vo = 0;

    UNITY_LOOP
    for (int ix = 0; ix < _TileMaxLoop; ix++)
    {
        UNITY_LOOP
        for (int iy = 0; iy < _TileMaxLoop; iy++)
        {
            float2 uv = uv0 + du * ix + dv * iy;
            vo = MaxV(vo, tex2D(_MainTex, uv).rg);
        }
    }

    return half4(vo, 0.0, 0.0);
}

// NeighborMax filter
half4 FragNeighborMax(VaryingsDefault i) : SV_Target
{
    const half cw = 1.01; // Center weight tweak

    float4 d = _MainTex_TexelSize.xyxy * float4(1.0, 1.0, -1.0, 0.0);

    half2 v1 = tex2D(_MainTex, i.uv - d.xy).rg;
    half2 v2 = tex2D(_MainTex, i.uv - d.wy).rg;
    half2 v3 = tex2D(_MainTex, i.uv - d.zy).rg;

    half2 v4 = tex2D(_MainTex, i.uv - d.xw).rg;
    half2 v5 = tex2D(_MainTex, i.uv).rg * cw;
    half2 v6 = tex2D(_MainTex, i.uv + d.xw).rg;

    half2 v7 = tex2D(_MainTex, i.uv + d.zy).rg;
    half2 v8 = tex2D(_MainTex, i.uv + d.wy).rg;
    half2 v9 = tex2D(_MainTex, i.uv + d.xy).rg;

    half2 va = MaxV(v1, MaxV(v2, v3));
    half2 vb = MaxV(v4, MaxV(v5, v6));
    half2 vc = MaxV(v7, MaxV(v8, v9));

    return half4(MaxV(va, MaxV(vb, vc)) / cw, 0.0, 0.0);
}

// -----------------------------------------------------------------------------
// Reconstruction

// Strength of the depth filter
static const float kDepthFilterCoeff = 15.0;

// Safer version of vector normalization function
half2 SafeNorm(half2 v)
{
    half l = max(length(v), EPSILON);
    return v / l * (l >= 0.5);
}

// Jitter function for tile lookup
float2 JitterTile(float2 uv)
{
    float rx, ry;
    sincos(GradientNoise(uv + float2(2.0, 0.0)) * UNITY_PI_2, ry, rx);
    return float2(rx, ry) * _NeighborMaxTex_TexelSize.xy / 4.0;
}

// Cone shaped interpolation
half Cone(half T, half l_V)
{
    return saturate(1.0 - T / l_V);
}

// Cylinder shaped interpolation
half Cylinder(half T, half l_V)
{
    return 1.0 - smoothstep(0.95 * l_V, 1.05 * l_V, T);
}

// Depth comparison function
half CompareDepth(half za, half zb)
{
    return saturate(1.0 - kDepthFilterCoeff * (zb - za) / min(za, zb));
}

// Lerp and normalization
half2 RNMix(half2 a, half2 b, half p)
{
    return SafeNorm(lerp(a, b, saturate(p)));
}

// Velocity sampling function
half3 SampleVelocity(float2 uv)
{
    half3 v = tex2D(_VelocityTex, uv).xyz;
    return half3((v.xy * 2.0 - 1.0) * _MaxBlurRadius, v.z);
}

// Sample weighting function
half SampleWeight(half2 d_n, half l_v_c, half z_p, half T, float2 S_uv, half w_A)
{
    half3 temp = tex2Dlod(_VelocityTex, float4(S_uv, 0.0, 0.0));

    half2 v_S = (temp.xy * 2.0 - 1.0) * _MaxBlurRadius;
    half l_v_S = max(length(v_S), 0.5);

    half z_S = temp.z;

    half f = CompareDepth(z_p, z_S);
    half b = CompareDepth(z_S, z_p);

    half w_B = abs(dot(v_S / l_v_S, d_n));

    half weight = 0.0;
    weight += f * Cone(T, l_v_S) * w_B;
    weight += b * Cone(T, l_v_c) * w_A;
    weight += Cylinder(T, min(l_v_S, l_v_c)) * max(w_A, w_B) * 2.0;

    return weight;
}

// Reconstruction filter
half4 FragReconstruction(VaryingsMultitex i) : SV_Target
{
    float2 p = i.uv1 * _ScreenParams.xy;
    float2 p_uv = i.uv1;

    // Nonfiltered source color;
    half4 source = tex2D(_MainTex, i.uv0);

    // Velocity vector at p.
    half3 v_c_t = SampleVelocity(p_uv);
    half2 v_c = v_c_t.xy;
    half2 v_c_n = SafeNorm(v_c);
    half l_v_c = max(length(v_c), 0.5);

    // NeighborMax vector at p (with small).
    half2 v_max = tex2D(_NeighborMaxTex, p_uv + JitterTile(p_uv)).xy;
    half2 v_max_n = SafeNorm(v_max);
    half l_v_max = length(v_max);

    // Escape early if the NeighborMax vector is too short.
    if (l_v_max < 0.5)
        return source;

    // Linearized depth at p.
    half z_p = v_c_t.z;

    // A vector perpendicular to v_max.
    half2 w_p = v_max_n.yx * float2(-1.0, 1.0);
    if (dot(w_p, v_c) < 0.0)
        w_p = -w_p;

    // Secondary sampling direction.
    half2 w_c = RNMix(w_p, v_c_n, (l_v_c - 0.5) / 1.5);

    // The center sample.
    half sampleCount = _LoopCount * 2.0;
    half totalWeight = sampleCount / (l_v_c * 40.0);
    half3 result = source.rgb * totalWeight;

    // Start from t=-1 + small jitter.
    // The width of jitter is equivalent to 4 sample steps.
    half sampleJitter = 4.0 * 2.0 / (sampleCount + 4.0);
    half t = -1.0 + GradientNoise(p_uv) * sampleJitter;
    half dt = (2.0 - sampleJitter) / sampleCount;

    // Precalculate the w_A parameters.
    half w_A1 = dot(w_c, v_c_n);
    half w_A2 = dot(w_c, v_max_n);

#ifndef UNROLL_LOOP_COUNT
    UNITY_LOOP for (int c = 0; c < _LoopCount; c++)
#else
    UNITY_UNROLL for (int c = 0; c < UNROLL_LOOP_COUNT; c++)
#endif
    {
        // Odd-numbered sample: sample along v_c.
        {
            float2 S_uv0 = i.uv0 + t * v_c * _MainTex_TexelSize.xy;
            float2 S_uv1 = i.uv1 + t * v_c * _VelocityTex_TexelSize.xy;
            half weight = SampleWeight(v_c_n, l_v_c, z_p, abs(t * l_v_max), S_uv1, w_A1);

            result += tex2Dlod(_MainTex, float4(S_uv0, 0.0, 0.0)).rgb * weight;
            totalWeight += weight;

            t += dt;
        }
        // Even-numbered sample: sample along v_max.
        {
            float2 S_uv0 = i.uv0 + t * v_max * _MainTex_TexelSize.xy;
            float2 S_uv1 = i.uv1 + t * v_max * _VelocityTex_TexelSize.xy;
            half weight = SampleWeight(v_max_n, l_v_c, z_p, abs(t * l_v_max), S_uv1, w_A2);

            result += tex2Dlod(_MainTex, float4(S_uv0, 0.0, 0.0)).rgb * weight;
            totalWeight += weight;

            t += dt;
        }
    }

    return half4(result / totalWeight, source.a);
}

// -----------------------------------------------------------------------------
// Frame blending

VaryingsDefault VertFrameCompress(AttributesDefault v)
{
    VaryingsDefault o;
    o.pos = v.vertex;
    o.uvSPR = 0;
#if UNITY_UV_STARTS_AT_TOP
    o.uv = v.texcoord * float2(1.0, -1.0) + float2(0.0, 1.0);
#else
    o.uv = v.texcoord;
#endif
    return o;
}

#if !SHADER_API_GLES

// MRT output struct for the compressor
struct CompressorOutput
{
    half4 luma   : SV_Target0;
    half4 chroma : SV_Target1;
};

// Frame compression fragment shader
CompressorOutput FragFrameCompress(VaryingsDefault i)
{
    float sw = _ScreenParams.x;     // Screen width
    float pw = _ScreenParams.z - 1; // Pixel width

    // RGB to YCbCr convertion matrix
    const half3 kY  = half3( 0.299   ,  0.587   ,  0.114   );
    const half3 kCB = half3(-0.168736, -0.331264,  0.5     );
    const half3 kCR = half3( 0.5     , -0.418688, -0.081312);

    // 0: even column, 1: odd column
    half odd = frac(i.uv.x * sw * 0.5) > 0.5;

    // Calculate UV for chroma componetns.
    // It's between the even and odd columns.
    float2 uv_c = i.uv.xy;
    uv_c.x = (floor(uv_c.x * sw * 0.5) * 2.0 + 1.0) * pw;

    // Sample the source texture.
    half3 rgb_y = tex2D(_MainTex, i.uv).rgb;
    half3 rgb_c = tex2D(_MainTex, uv_c).rgb;

    #if !UNITY_COLORSPACE_GAMMA
    rgb_y = LinearToGammaSpace(rgb_y);
    rgb_c = LinearToGammaSpace(rgb_c);
    #endif

    // Convertion and subsampling
    CompressorOutput o;
    o.luma = dot(kY, rgb_y);
    o.chroma = dot(lerp(kCB, kCR, odd), rgb_c) + 0.5;
    return o;
}

#else

// MRT might not be supported. Replace it with a null shader.
half4 FragFrameCompress(VaryingsDefault i) : SV_Target
{
    return 0;
}

#endif

// Sample luma-chroma textures and convert to RGB
half3 DecodeHistory(float2 uvLuma, float2 uvCb, float2 uvCr, sampler2D lumaTex, sampler2D chromaTex)
{
    half y = tex2D(lumaTex, uvLuma).r;
    half cb = tex2D(chromaTex, uvCb).r - 0.5;
    half cr = tex2D(chromaTex, uvCr).r - 0.5;
    return y + half3(1.402 * cr, -0.34414 * cb - 0.71414 * cr, 1.772 * cb);
}

// Frame blending fragment shader
half4 FragFrameBlending(VaryingsMultitex i) : SV_Target
{
    float sw = _MainTex_TexelSize.z; // Texture width
    float pw = _MainTex_TexelSize.x; // Texel width

    // UV for luma
    float2 uvLuma = i.uv1;

    // UV for Cb (even columns)
    float2 uvCb = i.uv1;
    uvCb.x = (floor(uvCb.x * sw * 0.5) * 2.0 + 0.5) * pw;

    // UV for Cr (even columns)
    float2 uvCr = uvCb;
    uvCr.x += pw;

    // Sample from the source image
    half4 src = tex2D(_MainTex, i.uv0);

    // Sampling and blending
    #if UNITY_COLORSPACE_GAMMA
    half3 acc = src.rgb;
    #else
    half3 acc = LinearToGammaSpace(src.rgb);
    #endif

    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History1LumaTex, _History1ChromaTex) * _History1Weight;
    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History2LumaTex, _History2ChromaTex) * _History2Weight;
    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History3LumaTex, _History3ChromaTex) * _History3Weight;
    acc += DecodeHistory(uvLuma, uvCb, uvCr, _History4LumaTex, _History4ChromaTex) * _History4Weight;
    acc /= 1.0 + _History1Weight + _History2Weight +_History3Weight +_History4Weight;

    #if !UNITY_COLORSPACE_GAMMA
    acc = GammaToLinearSpace(acc);
    #endif

    return half4(acc, src.a);
}

// Frame blending fragment shader (without chroma subsampling)
half4 FragFrameBlendingRaw(VaryingsMultitex i) : SV_Target
{
    half4 src = tex2D(_MainTex, i.uv0);
    half3 acc = src.rgb;
    acc += tex2D(_History1LumaTex, i.uv0) * _History1Weight;
    acc += tex2D(_History2LumaTex, i.uv0) * _History2Weight;
    acc += tex2D(_History3LumaTex, i.uv0) * _History3Weight;
    acc += tex2D(_History4LumaTex, i.uv0) * _History4Weight;
    acc /= 1.0 + _History1Weight + _History2Weight +_History3Weight +_History4Weight;
    return half4(acc, src.a);
}

#endif // __MOTION_BLUR__
