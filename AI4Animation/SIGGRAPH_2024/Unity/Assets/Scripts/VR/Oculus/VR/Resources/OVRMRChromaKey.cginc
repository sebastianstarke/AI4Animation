fixed4 _ChromaKeyColor;
float _ChromaKeySimilarity;
float _ChromaKeySmoothRange;
float _ChromaKeySpillRange;

// https://en.wikipedia.org/wiki/YUV
float3 RGB2YUV(float3 rgb)
{
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;
	float y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
	float u = -0.09991 * r - 0.33609 * g + 0.436 * b;
	float v = 0.615 * r - 0.55861 * g - 0.05639 * b;
	return float3(y, u, v);
}

float3 YUV2RGB(float3 yuv)
{
	float y = yuv.x;
	float u = yuv.y;
	float v = yuv.z;
	float r = y + 1.28033 * v;
	float g = y - 0.21482 * u - 0.38059 * v;
	float b = y + 2.12798 * u;
	return float3(r, g, b);
}

fixed4 GetCameraColor(float2 colorUV);

float ColorDistance_YUV_YUV(float3 yuv1, float3 yuv2)
{
	float dist = distance(yuv1.yz, yuv2.yz);
	// Increase the distance if the brightness of the first color is too high.
	// It fixed the error on the over exposure areas
	dist += saturate(yuv1.x - 0.9);
	return dist;
}

float ColorDistance_RGB_YUV(float3 rgb1, float3 yuv2)
{
	float3 yuv1 = RGB2YUV(rgb1);
	return ColorDistance_YUV_YUV(yuv1, yuv2);
}


float ColorDistance_RGB_RGB(float3 rgb1, float3 rgb2)
{
	float3 yuv1 = RGB2YUV(rgb1);
	float3 yuv2 = RGB2YUV(rgb2);
	return ColorDistance_YUV_YUV(yuv1, yuv2);
}

float RGB2Gray(float3 rgb)
{
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;
	float y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
	return y;
}

float GetAlphaFromDistance(float dist)
{
	float result = smoothstep(_ChromaKeySimilarity, _ChromaKeySimilarity + _ChromaKeySmoothRange, dist);
	result = result * result;
	return result;
}

float GetSpillFromDistance(float dist)
{
	float result = smoothstep(_ChromaKeySimilarity, _ChromaKeySimilarity + _ChromaKeySpillRange, dist);
	result = result * result * result;
	return result;
}


float4 GetColorAfterChromaKey(float2 UV, float2 deltaUV, float step)
{
	float3 chromaColor = _ChromaKeyColor.rgb;
	float3 chromaYUV = RGB2YUV(chromaColor);
	float dist = 0.0;
	const int samples = 3;
	float offset = ((float)samples - 1.0) / 2.0;
	for (int i = 0; i < samples; ++i)
	{
		for (int j = 0; j < samples; ++j)
		{
			fixed4 color = GetCameraColor(UV + float2((float)i - offset, (float)j - offset) * deltaUV * step);
			float d = ColorDistance_RGB_YUV(color, chromaYUV);
			dist += d;
		}
	}
	dist /= (samples * samples);
	fixed4 centerColor = GetCameraColor(UV);
	float alpha = GetAlphaFromDistance(dist);
	float spill = GetSpillFromDistance(dist);
	float gray = RGB2Gray(centerColor.rgb);
	float4 outColor;
	outColor.rgb = lerp(float3(gray, gray, gray), centerColor.rgb, spill);
	outColor.a = alpha;
	return outColor;
}
