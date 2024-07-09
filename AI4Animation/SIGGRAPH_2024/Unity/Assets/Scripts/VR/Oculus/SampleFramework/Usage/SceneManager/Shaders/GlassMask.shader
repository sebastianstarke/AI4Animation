Shader "Scene/Glass Mask"
{
  SubShader
  {
    Tags { "RenderType"="Opaque" "Queue"="Geometry-1" }
    ColorMask 0
    ZWrite Off

    Stencil
    {
      Ref 1
      Comp Always
      Pass Replace
    }

    Pass
    {
      Cull Back
      ZTest Less

      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag

      struct appdata
      {
        float4 vertex : POSITION;
      };
      struct v2f
      {
        float4 pos : SV_POSITION;
      };
      v2f vert(appdata v)
      {
        v2f o;
        o.pos = UnityObjectToClipPos(v.vertex);
        return o;
      }
      half4 frag(v2f i) : COLOR
      {
        return half4(1,1,1,1);
      }

      ENDCG
    }
  }
}
