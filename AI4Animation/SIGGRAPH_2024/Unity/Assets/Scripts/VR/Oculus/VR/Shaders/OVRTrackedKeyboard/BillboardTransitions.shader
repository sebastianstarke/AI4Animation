Shader "Hands Billboard Edge Fading Mask"
{
  Properties
  {
    _Intensity("Intensity", Range(0,1)) = 1.0

    [Space(5)]
    [Header(PARAMETRIC GLOW)]
    _Scale("Scale", Range(0,.1)) = 0.055
    _Falloff("Falloff", float) = 4.0
    _Power("Power", float) = 15.0

    [Space(5)]
    [Header(EDGE FADING)]
    [Toggle] _EdgeFading("Enable", Int) = 1
    _KeyboardPosition("Keyboard Position", Vector) = (0, 0, 0, 0)
    _KeyboardRotation("Keyboard Rotation", Vector) = (0, 0, 0, 0)
    _KeyboardScale("Keyboard Scale", Vector) = (1, 1, 1, 0)
    _FadingFalloff("Falloff", Range(0, .1)) = 0.05
    _ColorMultiply("Color multiply", Range(0, 3)) = 2.0
      
    // Blend modes
    [Enum(UnityEngine.Rendering.BlendMode)]_SrcBlendMode("Src Blend Factor", Int) = 0 // Zero
    [Enum(UnityEngine.Rendering.BlendMode)]_DstBlendMode("Dst Blend Factor", Int) = 4 // SrcColor
  }
    SubShader
    {
      // Transparent+1 to render after key labels
      Tags {"Queue" = "Transparent+1" "IgnoreProjector" = "True" "RenderType" = "Transparent"}
      ZWrite Off
      Cull Off
      ZTest Always
        
      Blend [_SrcBlendMode] [_DstBlendMode]

      Pass
      {
        CGPROGRAM

        #pragma vertex vert  
        #pragma fragment frag
        #pragma multi_compile_instancing
        #include "UnityCG.cginc"

        uniform float _Intensity;
        uniform float _Scale;
        float _Falloff;
        float _Power;

        int _EdgeFading;
        float4 _KeyboardPosition;
        float4 _KeyboardRotation;
        float4 _KeyboardScale;
        float _FadingFalloff;
        float _ColorMultiply;

        struct vertexInput
        {
          UNITY_VERTEX_INPUT_INSTANCE_ID
          float4 vertex : POSITION;
          float4 tex : TEXCOORD0;
          float4 color    : COLOR;
        };

        struct vertexOutput
        {
          UNITY_VERTEX_INPUT_INSTANCE_ID
          float4 pos : SV_POSITION;
          float4 tex : TEXCOORD0;
          float4 color : COLOR;
        };

        UNITY_INSTANCING_BUFFER_START(Props)
          //UNITY_DEFINE_INSTANCED_PROP(float4, _Color)
        UNITY_INSTANCING_BUFFER_END(Props)

        float4 RotateAroundYInDegrees(float4 vertex, float degrees)
        {
          float alpha = degrees * UNITY_PI / 180.0;
          float sina, cosa;
          sincos(alpha, sina, cosa);
          float2x2 m = float2x2(cosa, -sina, sina, cosa);
          return float4(mul(m, vertex.xz), vertex.yw).xzyw;
        }

        float4 BoundingBox(float4 vert, float4 position, float4 rotation, float4 scale)
        {
          float4 worldPos = mul(unity_ObjectToWorld, vert);
          float4 bboxOrientation = RotateAroundYInDegrees(worldPos - position, rotation.y);
          bboxOrientation = abs(bboxOrientation);

          scale *= 0.5f;

          float distX = (bboxOrientation.x - scale.x) / ((scale.x + _FadingFalloff) - scale.x);
          float distY = (bboxOrientation.y - scale.y) / ((scale.y + _FadingFalloff) - scale.y);
          float distZ = (bboxOrientation.z - scale.z) / ((scale.z + _FadingFalloff) - scale.z);
          float bBoxH = max(distX, distZ);
          float bBoxV = max(bBoxH, distY);
          float bBox = clamp(0,1,bBoxV);
          return float4(bBox, bBox, bBox, bBox);
        }

        vertexOutput vert(vertexInput input)
        {
          UNITY_SETUP_INSTANCE_ID(input);

          vertexOutput output;
          output.pos = mul(UNITY_MATRIX_P,
          mul(UNITY_MATRIX_MV, float4(0.0, 0.0, 0.0, 1.0))
          + float4(input.vertex.x, input.vertex.y, 0.0, 0.0)
          * float4(_Scale, _Scale, 1.0, 1.0));

          output.tex = input.tex;
          output.color = BoundingBox(input.vertex, _KeyboardPosition, _KeyboardRotation, _KeyboardScale);

          return output;
        }

        fixed BorderFade(float2 uv, float falloff, float intensity)
        {
          uv *= 1.0 - uv.yx;
          float fade = uv.x * uv.y * intensity;
          fade = pow(fade, falloff);
          return fade;
        }

        float4 frag(vertexOutput input) : COLOR
        {
          UNITY_SETUP_INSTANCE_ID(input);
          float glow = BorderFade(input.tex, _Falloff, _Power);
          float color = glow * _ColorMultiply;
          float4 circleAlpha = float4(color, color, color, glow) * _Intensity;
          float4 finalAlpha = circleAlpha * (1 - input.color.a);
          float4 transition = 1 - lerp(circleAlpha, finalAlpha, _EdgeFading);

          return transition;
        }

        ENDCG
      }
    }
}
