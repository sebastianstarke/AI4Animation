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

	public static Vector3 GetAverage(this Vector3[] vectors) {
		if(vectors.Length == 0) {
			return Vector3.zero;
		}
		if(vectors.Length == 1) {
			return vectors[0];
		}
		if(vectors.Length == 2) {
			return 0.5f*(vectors[0]+vectors[1]);
		}
		Vector3 avg = Vector3.zero;
		for(int i=0; i<vectors.Length; i++) {
			avg += vectors[i];
		}
		return avg / vectors.Length;
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

	public static Vector3 Positive(this Vector3 vector) {
		return new Vector3(Mathf.Abs(vector.x), Mathf.Abs(vector.y), Mathf.Abs(vector.z));
	}

	public static Vector3 Negative(this Vector3 vector) {
		return new Vector3(-Mathf.Abs(vector.x), -Mathf.Abs(vector.y), -Mathf.Abs(vector.z));
	}

	public static float Sum(this Vector3 vector) {
		return vector.x + vector.y + vector.z;
	}

}
