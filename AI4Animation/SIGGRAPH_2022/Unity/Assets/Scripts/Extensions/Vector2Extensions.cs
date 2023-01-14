using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Vector2Extensions {

	public static Vector2 GetMirror(this Vector2 vector, Axis axis) {
		if(axis == Axis.XPositive) {
			vector.x *= -1f;
		}
		if(axis == Axis.YPositive) {
			vector.y *= -1f;
		}
		return vector;
	}

	public static float[] Magnitudes(this Vector2[] vectors) {
		float[] magnitudes = new float[vectors.Length];
		for(int i=0; i<vectors.Length; i++) {
			magnitudes[i] = vectors[i].magnitude;
		}
		return magnitudes;
	}

	public static Vector2 Mean(this Vector2[] vectors) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		if(vectors.Length == 2) {
			return 0.5f*(vectors[0]+vectors[1]);
		}
		Vector2 mean = Vector2.zero;
		for(int i=0; i<vectors.Length; i++) {
			mean += vectors[i];
		}
		return mean / vectors.Length;
	}

	public static Vector2 Zero(this Vector2 vector, Axis axis) {
		if(axis == Axis.XPositive) {
			return vector.ZeroX();
		}
		if(axis == Axis.YPositive) {
			return vector.ZeroY();
		}
		return vector;
	}

	public static Vector2 ZeroX(this Vector2 vector) {
		vector.x = 0f;
		return vector;
	}

	public static Vector2 ZeroY(this Vector2 vector) {
		vector.y = 0f;
		return vector;
	}

	public static Vector2 ScaleX(this Vector2 vector, float value) {
		vector.x *= value;
		return vector;
	}

	public static Vector2 ScaleY(this Vector2 vector, float value) {
		vector.y *= value;
		return vector;
	}

	public static Vector2 ShiftX(this Vector2 vector, float value) {
		vector.x += value;
		return vector;
	}

	public static Vector2 ShiftY(this Vector2 vector, float value) {
		vector.y += value;
		return vector;
	}

	public static Vector2 Positive(this Vector2 vector) {
		return new Vector2(Mathf.Abs(vector.x), Mathf.Abs(vector.y));
	}

	public static Vector2 Negative(this Vector2 vector) {
		return new Vector2(-Mathf.Abs(vector.x), -Mathf.Abs(vector.y));
	}

	public static float Sum(this Vector2 vector) {
		return vector.x + vector.y;
	}

	public static float[] ToArray(this Vector2 vector) {
		return new float[2]{vector.x, vector.y};
	}

	public static float[][] ToArray(this Vector2[] vectors) {
		float[][] values = new float[vectors.Length][];
		for(int i=0; i<values.Length; i++) {
			values[i] = vectors[i].ToArray();
		}
		return values;
	}

}
