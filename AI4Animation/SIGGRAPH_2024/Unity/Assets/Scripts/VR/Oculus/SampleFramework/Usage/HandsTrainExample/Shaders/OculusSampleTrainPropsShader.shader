/************************************************************************************

Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

See SampleFramework license.txt for license terms.  Unless required by applicable law
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific
language governing permissions and limitations under the license.

************************************************************************************/

// does baked lighting + one directional light + reflection probe
Shader "Oculus Sample/Train Props Shader" {
  Properties {
    [PowerSlider(6.0)] _Shininess("Shininess", Range(0.02, 1)) = 0.03
    _MainTex("Base (RGB) Gloss (A)", 2D) = "white" {}
    _Color("Color", Color) = (0.5, 0.5, 0.5, 0.5)
    _BumpMap("Normal map", 2D) = "bump" {}
    _ReflectionMap("Reflection map", CUBE) = "" {}
    [MaterialToggle] _ReflectionMapEnabled("Enable Reflection map", Float) = 0
  }
  SubShader {
    Tags{"RenderType" = "Opaque"}
    LOD 300

    CGPROGRAM
    #include "UnityCG.cginc"
    // no deferred. support lightmaps and one light. use half vector instead of
    // view vector (less accurate but faster)
    #pragma surface surf BlinnPhong exclude_path:prepass noforwardadd halfasview 

    sampler2D _MainTex;

    struct Input {
      float2 uv_MainTex;
      float3 worldRefl;
      float3 viewDir;
      INTERNAL_DATA
    };

    half _Shininess;
    fixed4 _Color;
    sampler2D _BumpMap;
    samplerCUBE _ReflectionMap;
    fixed _ReflectionMapEnabled;

    void surf(Input IN, inout SurfaceOutput o) {
      fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex)*_Color;
      fixed gloss = albedo.a*_Shininess;
      fixed4 fakeReflec = texCUBE(_ReflectionMap, WorldReflectionVector(IN, o.Normal));

      o.Albedo = albedo.rgb;
      o.Gloss = gloss;
      o.Alpha = gloss;
      o.Specular = gloss;
      o.Normal = UnpackNormal(tex2D(_BumpMap, IN.uv_MainTex));
      o.Emission = _ReflectionMapEnabled * gloss * fakeReflec.rgb;
    }
    ENDCG
  }

  FallBack "Mobile/VertexLit"
}
