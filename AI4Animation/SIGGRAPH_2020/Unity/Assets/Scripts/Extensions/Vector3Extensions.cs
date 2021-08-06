using UnityEngine;

public static class Vector3Extensions {

	public static Vector3 GetRelativePositionFrom(this Vector3 position, Matrix4x4 from) {
		//return from.MultiplyPoint(position);
		return from.MultiplyPoint3x4(position);
	}

	public static Vector3 GetRelativePositionTo(this Vector3 position, Matrix4x4 to) {
		//return to.inverse.MultiplyPoint(position);
		return to.inverse.MultiplyPoint3x4(position);
	}

	public static Vector3 GetRelativeDirectionFrom(this Vector3 direction, Matrix4x4 from) {
		return from.MultiplyVector(direction);
	}

	public static Vector3 GetRelativeDirectionTo(this Vector3 direction, Matrix4x4 to) {
		return to.inverse.MultiplyVector(direction);
	}

	public static Vector3 GetPositionFromTo(this Vector3 position, Matrix4x4 from, Matrix4x4 to) {
		return position.GetRelativePositionTo(from).GetRelativePositionFrom(to);
	}

	public static Vector3 GetDirectionFromTo(this Vector3 direction, Matrix4x4 from, Matrix4x4 to) {
		return direction.GetRelativeDirectionTo(from).GetRelativeDirectionFrom(to);
	}
	
	public static Vector3 RotateAround(this Vector3 vector, Vector3 pivot, float angle, Vector3 axis) {
		return pivot + Quaternion.AngleAxis(angle, axis) * (vector-pivot);
	}

	public static float SignedAngle(Vector3 A, Vector3 B, Vector3 axis) {
		return Mathf.Atan2(
			Vector3.Dot(axis, Vector3.Cross(A, B)),
			Vector3.Dot(A, B)
			) * Mathf.Rad2Deg;
	}

	public static Vector3 GetMirror(this Vector3 vector, Axis axis) {
		if(axis == Axis.XPositive) {
			vector.x *= -1f;
		}
		if(axis == Axis.YPositive) {
			vector.y *= -1f;
		}
		if(axis == Axis.ZPositive) {
			vector.z *= -1f;
		}
		return vector;
	}

	public static float[] Magnitudes(this Vector3[] vectors) {
		float[] magnitudes = new float[vectors.Length];
		for(int i=0; i<vectors.Length; i++) {
			magnitudes[i] = vectors[i].magnitude;
		}
		return magnitudes;
	}

	public static Vector3 Mean(this Vector3[] vectors) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		if(vectors.Length == 2) {
			return 0.5f*(vectors[0]+vectors[1]);
		}
		Vector3 mean = Vector3.zero;
		for(int i=0; i<vectors.Length; i++) {
			mean += vectors[i];
		}
		return mean / vectors.Length;
	}

	public static Vector3 Mean(this Vector3[] vectors, float[] weights) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		if(vectors.Length != weights.Length) {
			Debug.Log("Failed to compute mean because size of vectors and weights does not match.");
			return Vector3.zero;
		}
		float sum = 0f;
		Vector3 mean = Vector3.zero;
		for(int i=0; i<vectors.Length; i++) {
			sum += weights[i];
			mean += weights[i] * vectors[i];
		}
		if(sum == 0f) {
			Debug.Log("Failed to compute mean because size of sum of weights is zero.");
			return Vector3.zero;
		}
		return mean / sum;
	}

	public static Vector3 Gaussian(this Vector3[] values, float power=1f) {
		if(values.Length == 0) {
			return Vector3.zero;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float window = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		Vector3 value = Vector3.zero;
		for(int i=0; i<values.Length; i++) {
			float weight = Mathf.Exp(-Mathf.Pow((float)i - window, 2f) / Mathf.Pow(0.5f * window, 2f));
			if(power != 1f) {
				weight = Mathf.Pow(weight, power);
			}
			value += weight * values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static Vector3 Gaussian(this Vector3[] values, bool[] mask, float power=1f) {
		if(values.Length == 0) {
			return Vector3.zero;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float window = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		Vector3 value = Vector3.zero;
		for(int i=0; i<values.Length; i++) {
			if(mask[i]) {
				float weight = Mathf.Exp(-Mathf.Pow((float)i - window, 2f) / Mathf.Pow(0.5f * window, 2f));
				if(power != 1f) {
					weight = Mathf.Pow(weight, power);
				}
				value += weight * values[i];
				sum += weight;
			}
		}
		return value / sum;
	}

	public static Vector3 Zero(this Vector3 vector, Axis axis) {
		if(axis == Axis.XPositive) {
			return vector.ZeroX();
		}
		if(axis == Axis.YPositive) {
			return vector.ZeroY();
		}
		if(axis == Axis.ZPositive) {
			return vector.ZeroZ();
		}
		return vector;
	}

	public static Vector3 ZeroX(this Vector3 vector) {
		vector.x = 0f;
		return vector;
	}

	public static Vector3 ZeroY(this Vector3 vector) {
		vector.y = 0f;
		return vector;
	}

	public static Vector3 ZeroZ(this Vector3 vector) {
		vector.z = 0f;
		return vector;
	}

	public static Vector3 SetX(this Vector3 vector, float value) {
		vector.x = value;
		return vector;
	}

	public static Vector3 SetY(this Vector3 vector, float value) {
		vector.y = value;
		return vector;
	}

	public static Vector3 SetZ(this Vector3 vector, float value) {
		vector.z = value;
		return vector;
	}
	
	public static Vector3 Positive(this Vector3 vector) {
		return new Vector3(Mathf.Abs(vector.x), Mathf.Abs(vector.y), Mathf.Abs(vector.z));
	}

	public static Vector3 Negative(this Vector3 vector) {
		return new Vector3(-Mathf.Abs(vector.x), -Mathf.Abs(vector.y), -Mathf.Abs(vector.z));
	}

	public static float Sum(this Vector3 vector) {
		return vector.x + vector.y + vector.z;
	}

	public static Vector3 ClampMagnitudeXZ(this Vector3 vector, float maxLength) {
		return Vector3.ClampMagnitude(vector.ZeroY(), maxLength).SetY(vector.y);
	}

	public static Vector3 ClampMagnitudeXZ(this Vector3 vector, float maxLength, Vector3 pivot) {
		return pivot + (vector-pivot).ClampMagnitudeXZ(maxLength);
	}

	public static float MagnitudeXZ(this Vector3 vector) {
		return vector.ZeroY().magnitude;
	}

	public static Vector3 NormalizeXZ(this Vector3 vector) {
		return vector.ZeroY().normalized.SetY(vector.y);
	}

	public static float[] ToArray(this Vector3 vector) {
		return new float[3]{vector.x, vector.y, vector.z};
	}

	public static float[][] ToArray(this Vector3[] vectors) {
		float[][] values = new float[vectors.Length][];
		for(int i=0; i<values.Length; i++) {
			values[i] = vectors[i].ToArray();
		}
		return values;
	}

}
