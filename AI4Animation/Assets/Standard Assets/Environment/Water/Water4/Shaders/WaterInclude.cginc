
#ifndef WATER_CG_INCLUDED
#define WATER_CG_INCLUDED

#include "UnityCG.cginc"

half _GerstnerIntensity;

inline half3 PerPixelNormal(sampler2D bumpMap, half4 coords, half3 vertexNormal, half bumpStrength) 
{
	half3 bump = (UnpackNormal(tex2D(bumpMap, coords.xy)) + UnpackNormal(tex2D(bumpMap, coords.zw))) * 0.5;
	half3 worldNormal = vertexNormal + bump.xxy * bumpStrength * half3(1,0,1);
	return normalize(worldNormal);
} 

inline half3 PerPixelNormalUnpacked(sampler2D bumpMap, half4 coords, half bumpStrength) 
{
	half4 bump = tex2D(bumpMap, coords.xy) + tex2D(bumpMap, coords.zw);
	bump = bump * 0.5;
	half3 normal = UnpackNormal(bump);
	normal.xy *= bumpStrength;
	return normalize(normal);
} 

inline half3 GetNormal(half4 tf) {
	#ifdef WATER_VERTEX_DISPLACEMENT_ON
		return half3(2,1,2) * tf.rbg - half3(1,0,1);
	#else
		return half3(0,1,0);
	#endif	
}

inline half GetDistanceFadeout(half screenW, half speed) {
	return 1.0f / abs(0.5f + screenW * speed);	
}

half4 GetDisplacement3(half4 tileableUv, half4 tiling, half4 directionSpeed, sampler2D mapA, sampler2D mapB, sampler2D mapC)
{
	half4 displacementUv = tileableUv * tiling + _Time.xxxx * directionSpeed;
	#ifdef WATER_VERTEX_DISPLACEMENT_ON			
		half4 tf = tex2Dlod(mapA, half4(displacementUv.xy, 0.0,0.0));
		tf += tex2Dlod(mapB, half4(displacementUv.zw, 0.0,0.0));
		tf += tex2Dlod(mapC, half4(displacementUv.xw, 0.0,0.0));
		tf *= 0.333333; 
	#else
		half4 tf = half4(0.5,0.5,0.5,0.0);
	#endif
	
	return tf;
}

half4 GetDisplacement2(half4 tileableUv, half4 tiling, half4 directionSpeed, sampler2D mapA, sampler2D mapB)
{
	half4 displacementUv = tileableUv * tiling + _Time.xxxx * directionSpeed;
	#ifdef WATER_VERTEX_DISPLACEMENT_ON			
		half4 tf = tex2Dlod(mapA, half4(displacementUv.xy, 0.0,0.0));
		tf += tex2Dlod(mapB, half4(displacementUv.zw, 0.0,0.0));
		tf *= 0.5; 
	#else
		half4 tf = half4(0.5,0.5,0.5,0.0);
	#endif
	
	return tf;
}

inline void ComputeScreenAndGrabPassPos (float4 pos, out float4 screenPos, out float4 grabPassPos) 
{
	#if UNITY_UV_STARTS_AT_TOP
		float scale = -1.0;
	#else
		float scale = 1.0f;
	#endif
	
	screenPos = ComputeNonStereoScreenPos(pos); 
	grabPassPos.xy = ( float2( pos.x, pos.y*scale ) + pos.w ) * 0.5;
	grabPassPos.zw = pos.zw;
}


inline half3 PerPixelNormalUnpacked(sampler2D bumpMap, half4 coords, half bumpStrength, half2 perVertxOffset)
{
	half4 bump = tex2D(bumpMap, coords.xy) + tex2D(bumpMap, coords.zw);
	bump = bump * 0.5;
	half3 normal = UnpackNormal(bump);
	normal.xy *= bumpStrength;
	normal.xy += perVertxOffset;
	return normalize(normal);	
}

inline half3 PerPixelNormalLite(sampler2D bumpMap, half4 coords, half3 vertexNormal, half bumpStrength) 
{
	half4 bump = tex2D(bumpMap, coords.xy);
	bump.xy = bump.wy - half2(0.5, 0.5);
	half3 worldNormal = vertexNormal + bump.xxy * bumpStrength * half3(1,0,1);
	return normalize(worldNormal);
} 

inline half4 Foam(sampler2D shoreTex, half4 coords, half amount) 
{
	half4 foam = ( tex2D(shoreTex, coords.xy) * tex2D(shoreTex,coords.zw) ) - 0.125;
	foam.a = amount;
	return foam;
}

inline half4 Foam(sampler2D shoreTex, half4 coords) 
{
	half4 foam = (tex2D(shoreTex, coords.xy) * tex2D(shoreTex,coords.zw)) - 0.125;
	return foam;
}

inline half Fresnel(half3 viewVector, half3 worldNormal, half bias, half power)
{
	half facing =  clamp(1.0-max(dot(-viewVector, worldNormal), 0.0), 0.0,1.0);	
	half refl2Refr = saturate(bias+(1.0-bias) * pow(facing,power));	
	return refl2Refr;	
}

inline half FresnelViaTexture(half3 viewVector, half3 worldNormal, sampler2D fresnel)
{
	half facing =  saturate(dot(-viewVector, worldNormal));	
	half fresn = tex2D(fresnel, half2(facing, 0.5f)).b;	
	return fresn;
}

inline void VertexDisplacementHQ(	sampler2D mapA, sampler2D mapB,
									sampler2D mapC, half4 uv,
									half vertexStrength, half3 normal,
									out half4 vertexOffset, out half2 normalOffset) 
{	
	half4 tf = tex2Dlod(mapA, half4(uv.xy, 0.0,0.0));
	tf += tex2Dlod(mapB, half4(uv.zw, 0.0,0.0));
	tf += tex2Dlod(mapC, half4(uv.xw, 0.0,0.0));
	tf /= 3.0; 
	
	tf.rga = tf.rga-half3(0.5,0.5,0.0);
				
	// height displacement in alpha channel, normals info in rgb
	
	vertexOffset = tf.a * half4(normal.xyz, 0.0) * vertexStrength;							
	normalOffset = tf.rg;
}

inline void VertexDisplacementLQ(	sampler2D mapA, sampler2D mapB,
									sampler2D mapC, half4 uv,
									half vertexStrength, half normalsStrength,
									out half4 vertexOffset, out half2 normalOffset) 
{
	// @NOTE: for best performance, this should really be properly packed!
	
	half4 tf = tex2Dlod(mapA, half4(uv.xy, 0.0,0.0));
	tf += tex2Dlod(mapB, half4(uv.zw, 0.0,0.0));
	tf *= 0.5; 
	
	tf.rga = tf.rga-half3(0.5,0.5,0.0);
				
	// height displacement in alpha channel, normals info in rgb
	
	vertexOffset = tf.a * half4(0,1,0,0) * vertexStrength;							
	normalOffset = tf.rg * normalsStrength;
}

half4  ExtinctColor (half4 baseColor, half extinctionAmount) 
{
	// tweak the extinction coefficient for different coloring
	 return baseColor - extinctionAmount * half4(0.15, 0.03, 0.01, 0.0);
}

	half3 GerstnerOffsets (half2 xzVtx, half steepness, half amp, half freq, half speed, half2 dir) 
	{
		half3 offsets;
		
		offsets.x =
			steepness * amp * dir.x *
			cos( freq * dot( dir, xzVtx ) + speed * _Time.x); 
			
		offsets.z =
			steepness * amp * dir.y *
			cos( freq * dot( dir, xzVtx ) + speed * _Time.x); 
			
		offsets.y = 
			amp * sin ( freq * dot( dir, xzVtx ) + speed * _Time.x);

		return offsets;			
	}	

	half3 GerstnerOffset4 (half2 xzVtx, half4 steepness, half4 amp, half4 freq, half4 speed, half4 dirAB, half4 dirCD) 
	{
		half3 offsets;
		
		half4 AB = steepness.xxyy * amp.xxyy * dirAB.xyzw;
		half4 CD = steepness.zzww * amp.zzww * dirCD.xyzw;
		
		half4 dotABCD = freq.xyzw * half4(dot(dirAB.xy, xzVtx), dot(dirAB.zw, xzVtx), dot(dirCD.xy, xzVtx), dot(dirCD.zw, xzVtx));
		half4 TIME = _Time.yyyy * speed;
		
		half4 COS = cos (dotABCD + TIME);
		half4 SIN = sin (dotABCD + TIME);
		
		offsets.x = dot(COS, half4(AB.xz, CD.xz));
		offsets.z = dot(COS, half4(AB.yw, CD.yw));
		offsets.y = dot(SIN, amp);

		return offsets;			
	}	

	half3 GerstnerNormal (half2 xzVtx, half steepness, half amp, half freq, half speed, half2 dir) 
	{
		half3 nrml = half3(0,0,0);
		
		nrml.x -=
			dir.x * (amp * freq) * 
			cos(freq * dot( dir, xzVtx ) + speed * _Time.x);
			
		nrml.z -=
			dir.y * (amp * freq) * 
			cos(freq * dot( dir, xzVtx ) + speed * _Time.x);	

		return nrml;			
	}	
	
	half3 GerstnerNormal4 (half2 xzVtx, half4 amp, half4 freq, half4 speed, half4 dirAB, half4 dirCD) 
	{
		half3 nrml = half3(0,2.0,0);
		
		half4 AB = freq.xxyy * amp.xxyy * dirAB.xyzw;
		half4 CD = freq.zzww * amp.zzww * dirCD.xyzw;
		
		half4 dotABCD = freq.xyzw * half4(dot(dirAB.xy, xzVtx), dot(dirAB.zw, xzVtx), dot(dirCD.xy, xzVtx), dot(dirCD.zw, xzVtx));
		half4 TIME = _Time.yyyy * speed;
		
		half4 COS = cos (dotABCD + TIME);
		
		nrml.x -= dot(COS, half4(AB.xz, CD.xz));
		nrml.z -= dot(COS, half4(AB.yw, CD.yw));
		
		nrml.xz *= _GerstnerIntensity;
		nrml = normalize (nrml);

		return nrml;			
	}	
	
	void Gerstner (	out half3 offs, out half3 nrml,
					 half3 vtx, half3 tileableVtx, 
					 half4 amplitude, half4 frequency, half4 steepness, 
					 half4 speed, half4 directionAB, half4 directionCD ) 
	{
		#ifdef WATER_VERTEX_DISPLACEMENT_ON
			offs = GerstnerOffset4(tileableVtx.xz, steepness, amplitude, frequency, speed, directionAB, directionCD);
			nrml = GerstnerNormal4(tileableVtx.xz + offs.xz, amplitude, frequency, speed, directionAB, directionCD);		
		#else
			offs = half3(0,0,0);
			nrml = half3(0,1,0);
		#endif							
	}


#endif
