using UnityEngine;

public static class Vector3Extensions {

	public static Vector3 PositionFrom(this Vector3 position, Matrix4x4 from) {
		//return from.MultiplyPoint(position);
		return from.MultiplyPoint3x4(position);
	}

	public static Vector3 PositionTo(this Vector3 position, Matrix4x4 to) {
		//return to.inverse.MultiplyPoint(position);
		return to.inverse.MultiplyPoint3x4(position);
	}

	public static Vector3 DirectionFrom(this Vector3 direction, Matrix4x4 from) {
		return direction.DirectionFrom(from.GetRotation());
	}

	public static Vector3 DirectionFrom(this Vector3 direction, Quaternion from) {
		return from * direction;
	}

	public static Vector3 DirectionTo(this Vector3 direction, Matrix4x4 to) {
		return direction.DirectionTo(to.GetRotation());
	}

	public static Vector3 DirectionTo(this Vector3 direction, Quaternion to) {
		return Quaternion.Inverse(to) * direction;
	}

	public static Vector3 PositionFromTo(this Vector3 position, Matrix4x4 from, Matrix4x4 to) {
		return position.PositionTo(from).PositionFrom(to);
	}

	public static Vector3 DirectionFromTo(this Vector3 direction, Matrix4x4 from, Matrix4x4 to) {
		return direction.DirectionTo(from).DirectionFrom(to);
	}

	public static Vector3[] PositionsFrom(this Vector3[] positions, Matrix4x4 from, bool inplace) {
		Vector3[] result = inplace ? positions : new Vector3[positions.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = positions[i].PositionFrom(from);
		}
		return result;
	}

	public static Vector3[] PositionsTo(this Vector3[] positions, Matrix4x4 to, bool inplace) {
		Vector3[] result = inplace ? positions : new Vector3[positions.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = positions[i].PositionTo(to);
		}
		return result;
	}

	public static Vector3[] DirectionsFrom(this Vector3[] directions, Matrix4x4 from, bool inplace) {
		Vector3[] result = inplace ? directions : new Vector3[directions.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = directions[i].DirectionFrom(from);
		}
		return result;
	}

	public static Vector3[] DirectionsTo(this Vector3[] directions, Matrix4x4 to, bool inplace) {
		Vector3[] result = inplace ? directions : new Vector3[directions.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = directions[i].DirectionTo(to);
		}
		return result;
	}

	public static Vector3[] PositionsFromTo(this Vector3[] positions, Matrix4x4 from, Matrix4x4 to, bool inplace) {
		Vector3[] result = inplace ? positions : new Vector3[positions.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = positions[i].PositionFromTo(from, to);
		}
		return result;
	}

	public static Vector3[] DirectionsFromTo(this Vector3[] directions, Matrix4x4 from, Matrix4x4 to, bool inplace) {
		Vector3[] result = inplace ? directions : new Vector3[directions.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = directions[i].DirectionFromTo(from, to);
		}
		return result;
	}
	
	public static Vector3 CatmullRom(float t, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 v3) {
		Vector3 a = 2f * v1;
		Vector3 b = v2 - v0;
		Vector3 c = 2f * v0 - 5f * v1 + 4f * v2 - v3;
		Vector3 d = -v0 + 3f * v1 - 3f * v2 + v3;
		return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
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

	public static Vector3Int ToVector3Int(this Vector3 vector) {
		return new Vector3Int(
				Mathf.RoundToInt(vector.x),
				Mathf.RoundToInt(vector.y),
				Mathf.RoundToInt(vector.z)
			);
	}

	public static float[] Magnitudes(this Vector3[] vectors) {
		float[] magnitudes = new float[vectors.Length];
		for(int i=0; i<vectors.Length; i++) {
			magnitudes[i] = vectors[i].magnitude;
		}
		return magnitudes;
	}

	public static Vector3 MaxMagnitude(this Vector3[] values) {
		if(values.Length == 0) {
			return Vector3.zero;
		}
		if(values.Length == 1) {
			return values[0];
		}
		Vector3 max = values[0];
		float mag = max.magnitude;
		for(int i=1; i<values.Length; i++) {
			float m = values[i].magnitude;
			if(m > mag) {
				max = values[i];
				mag = m;
			}
		}
		return max;
	}

	public static Vector3 Max(this Vector3[] vectors) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		Vector3 result = new Vector3(float.MinValue, float.MinValue, float.MinValue);
		foreach(Vector3 vector in vectors) {
			result = Vector3.Max(result, vector);
		}
		return result;
	}

	public static Vector3 Min(this Vector3[] vectors) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		Vector3 result = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
		foreach(Vector3 vector in vectors) {
			result = Vector3.Min(result, vector);
		}
		return result;
	}

	public static Vector3 Sum(this Vector3[] vectors) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		if(vectors.Length == 2) {
			return vectors[0]+vectors[1];
		}
		Vector3 sum = Vector3.zero;
		for(int i=0; i<vectors.Length; i++) {
			sum += vectors[i];
		}
		return sum;
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

	public static Vector3 Gaussian(this Vector3[] values, float power=1f, bool[] mask=null) {
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
			if(mask == null || mask != null && mask[i]) {
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

	public static Vector2 ToXZ(this Vector3 vector) {
		return new Vector2(vector.x, vector.z);
	}
	
	public static Vector3 Positive(this Vector3 vector) {
		return new Vector3(Mathf.Abs(vector.x), Mathf.Abs(vector.y), Mathf.Abs(vector.z));
	}

	public static Vector3 Negative(this Vector3 vector) {
		return new Vector3(-Mathf.Abs(vector.x), -Mathf.Abs(vector.y), -Mathf.Abs(vector.z));
	}

	public static Vector3 ClampMagnitudeXY(this Vector3 vector, float maxLength) {
		return Vector3.ClampMagnitude(vector.ZeroZ(), maxLength).SetZ(vector.z);
	}

	public static Vector3 ClampMagnitudeXY(this Vector3 vector, float maxLength, Vector3 pivot) {
		return pivot + (vector-pivot).ClampMagnitudeXY(maxLength);
	}

	public static Vector3 ClampMagnitudeXZ(this Vector3 vector, float maxLength) {
		return Vector3.ClampMagnitude(vector.ZeroY(), maxLength).SetY(vector.y);
	}

	public static Vector3 ClampMagnitudeXZ(this Vector3 vector, float maxLength, Vector3 pivot) {
		return pivot + (vector-pivot).ClampMagnitudeXZ(maxLength);
	}

	public static Vector3 ClampMagnitude(this Vector3 v, float min, float max) {
		float sm = v.sqrMagnitude;
		if(sm > max * max) return v.normalized * max;
		else if(sm < min * min) return v.normalized * min;
		return v;
	}

	public static float MagnitudeXZ(this Vector3 vector) {
		return vector.ZeroY().magnitude;
	}

	public static float Sum(this Vector3 vector) {
		return vector.x + vector.y + vector.z;
	}

	public static float Mean(this Vector3 vector) {
		return (vector.x + vector.y + vector.z) / 3f;
	}

	public static float MeanXZ(this Vector3 vector) {
		return (vector.x + vector.z) / 2f;
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

	public static float[] ToArrayX(this Vector3[] vectors) {
		float[] values = new float[vectors.Length];
		for(int i=0; i<vectors.Length; i++) {
			values[i] = vectors[i].x;
		}
		return values;
	}

	public static float[] ToArrayY(this Vector3[] vectors) {
		float[] values = new float[vectors.Length];
		for(int i=0; i<vectors.Length; i++) {
			values[i] = vectors[i].y;
		}
		return values;
	}

	public static float[] ToArrayZ(this Vector3[] vectors) {
		float[] values = new float[vectors.Length];
		for(int i=0; i<vectors.Length; i++) {
			values[i] = vectors[i].z;
		}
		return values;
	}
}
