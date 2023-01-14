using UnityEngine;

public static class Vector4Extensions {

	public static float MagnitudeXZ(this Vector4 vector) {
		return vector.ZeroY().ZeroW().magnitude;
	}

	public static Vector4 NormalizeXZ(this Vector4 vector) {
		return vector.ZeroY().ZeroW().normalized.SetY(vector.y).SetW(vector.w);
	}

    public static Vector4 ZeroX(this Vector4 vector) {
		vector.x = 0f;
		return vector;
	}

	public static Vector4 ZeroY(this Vector4 vector) {
		vector.y = 0f;
		return vector;
	}

	public static Vector4 ZeroZ(this Vector4 vector) {
		vector.z = 0f;
		return vector;
	}

	public static Vector4 ZeroW(this Vector4 vector) {
		vector.w = 0f;
		return vector;
	}

	public static Vector4 SetX(this Vector4 vector, float value) {
		vector.x = value;
		return vector;
	}

	public static Vector4 SetY(this Vector4 vector, float value) {
		vector.y = value;
		return vector;
	}

	public static Vector4 SetZ(this Vector4 vector, float value) {
		vector.z = value;
		return vector;
	}

	public static Vector4 SetW(this Vector4 vector, float value) {
		vector.w = value;
		return vector;
	}

}