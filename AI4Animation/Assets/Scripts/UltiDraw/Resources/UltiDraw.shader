 Shader "UltiDraw" {
 Properties {
     _Color ("Color" , Color) = (1,1,1,1)
     _Power ("Power", Float) = 0.25
     _Solid ("Filling", Float) = 0.0
     _MainTex ("Base (RGB)", 2D) = "white" {}
     _ZTest ("ZTest", Int) = 0
     _ZWrite ("ZWrite", Int) = 0
 }
 SubShader {
     Pass {
             Name "UltiDraw"
             Tags { "RenderType"="transparent" "Queue" = "Transparent" }
             Blend SrcAlpha OneMinusSrcAlpha
             ZTest [_ZTest]
             Cull Back
             ZWrite [_ZWrite]
             LOD 200                    
            
             CGPROGRAM
             #pragma vertex vert
             #pragma fragment frag
             #include "UnityCG.cginc"
            
             struct v2f {
                 float4 pos : SV_POSITION;
                 float2 uv : TEXCOORD0;
                 float3 normal : TEXCOORD1;      // Normal needed for rim lighting
                 float3 viewDir : TEXCOORD2;     // as is view direction.
             };
            
             sampler2D _MainTex;
             float4 _Color;
             float _Power;
             float _Filling;
            
             float4 _MainTex_ST;
            
             v2f vert (appdata_tan v) {
                 v2f o;
                 o.pos = UnityObjectToClipPos(v.vertex);
                 o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
                 o.normal = normalize(v.normal);
                 o.viewDir = normalize(ObjSpaceViewDir(v.vertex));       //this could also be WorldSpaceViewDir, which would
                 return o;                                               //return the World space view direction.
             }
            
             half4 frag (v2f i) : COLOR {
                 half saturation = saturate(dot(normalize(i.viewDir), i.normal));
                 half Power = (1 - _Filling) * (1 - saturation) + (_Filling * saturation);           
                 half4 PowerOut = _Color * pow(Power, _Power);
                 return PowerOut;
             }
             ENDCG
         }
 
         Pass {
             Name "BASE"
             ZWrite On
             ZTest LEqual
             Blend SrcAlpha OneMinusSrcAlpha
             Material {
                 Diffuse [_Color]
                 Ambient [_Color]
             }
             Lighting On
             SetTexture [_MainTex] {
                 ConstantColor [_Color]
                 Combine texture * constant
             }
             SetTexture [_MainTex] {
                 Combine previous * primary DOUBLE
             }
         }
                
     }
     FallBack "Diffuse"
 }