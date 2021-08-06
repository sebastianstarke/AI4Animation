using System;
using System.Collections.Generic;
using UnityEngine;

public static class ValueExtensions {
	
	public static float Ratio(this int index, int start, int end) {
		return (float)(index-start) / (float)(end-start);
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
