using UnityEngine;

public static class ValueExtensions {

	public static float CatmullRom(float t, float v0, float v1, float v2, float v3) {
		float a = 2f * v1;
		float b = v2 - v0;
		float c = 2f * v0 - 5f * v1 + 4f * v2 - v3;
		float d = -v0 + 3f * v1 - 3f * v2 + v3;
		return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
	}

	public static float Round(this float value, int digits) {
		float scale = Mathf.Pow(10, digits);
		return Mathf.Round(value * scale) / scale;
	}

	public static float Ratio(this int index, int start, int end) {
		if(start == end) {
			return 1f;
		}
		return Mathf.Clamp((float)(index-start) / (float)(end-start), 0f, 1f);
	}

	public static float Ratio(this float current, float start, float end) {
		if(start == end) {
			return 1f;
		}
		return Mathf.Clamp((current-start) / (end-start), 0f, 1f);
	}

	public static float SmoothStep(this float x, float power, float threshold) {
		//Validate
		x = Mathf.Clamp(x, 0f, 1f);
		power = Mathf.Max(power, 0f);
		threshold = Mathf.Clamp(threshold, 0f, 1f);

		//Skew X
		if(threshold == 0f || threshold == 1f) {
			x = 1f - threshold;
		} else {
			if(threshold < 0.5f) {
				x = 1f - Mathf.Pow(1f-x, 0.5f / threshold);
			}
			if(threshold > 0.5f) {
				x = Mathf.Pow(x, 0.5f / (1f-threshold));
			}
		}

		//Evaluate Y
		if(x < 0.5f) {
			return 0.5f*Mathf.Pow(2f*x, power);
		}
		if(x > 0.5f) {
			return 1f - 0.5f*Mathf.Pow(2f-2f*x, power);
		}
		return 0.5f;
	}

	public static float ActivatePeak(this float x, float center, float steepness) {
		return 4f * x.SmoothStep(steepness, center) * (1f-x.SmoothStep(steepness, center));
	}

	public static float ActivateCurve(this float x, float bias, float start, float end) {
		bias = Mathf.Clamp(bias, 0f, 1f);
		if(end < start) {
			return (1f - Mathf.Pow(1f - Mathf.Pow(1f-x, 1f-bias), bias)).Normalize(0f, 1f, end, start);
		} else {
			return (1f - Mathf.Pow(1f - Mathf.Pow(x, 1f-bias), bias)).Normalize(0f, 1f, start, end);
		}
	}

	//In-Place
	public static float[] Normalize(this float[] values, float[] valueMin, float[] valueMax, float[] resultMin, float[] resultMax) {
		float[] result = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			result[i] = values[i].Normalize(valueMin[i], valueMax[i], resultMin[i], resultMax[i]);
		}
		return result;
	}

	//In-Place
	public static double[] Normalize(this double[] values, double[] valueMin, double[] valueMax, double[] resultMin, double[] resultMax) {
		double[] result = new double[values.Length];
		for(int i=0; i<values.Length; i++) {
			result[i] = values[i].Normalize(valueMin[i], valueMax[i], resultMin[i], resultMax[i]);
		}
		return result;
	}

	public static float Normalize(this float value, float valueMin, float valueMax, float resultMin, float resultMax) {
		if(valueMax-valueMin != 0f) {
			return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin;
		} else {
			//Not possible to normalize input value.
			return value;
		}
	}

	public static double Normalize(this double value, double valueMin, double valueMax, double resultMin, double resultMax) {
		if(valueMax-valueMin != 0.0) {
			return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin;
		} else {
			//Not possible to normalize input value.
			return value;
		}
	}

}
