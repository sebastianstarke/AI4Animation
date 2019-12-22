Shader"Voxelizer/Demo/VolumeTest"
{

	Properties
	{
		_MainTex ("Texture", 3D) = "white" {}
        _Offset ("Offset", Vector) = (0.5, 0.5, 0.5, -1)
	}

	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float3 uv : TEXCOORD0;
			};

			sampler3D _MainTex;
            half3 _Offset;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.vertex.xyz + _Offset;
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = tex3D(_MainTex, i.uv);
				return col;
			}
			ENDCG
		}
	}
}
