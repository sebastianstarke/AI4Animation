using UnityEngine;

public static class QuaternionExtensions {

	public static Quaternion GetRelativeRotationFrom(this Quaternion rotation, Matrix4x4 from) {
		return from.GetRotation() * rotation;
	}

	public static Quaternion GetRelativeRotationTo(this Quaternion rotation, Matrix4x4 to) {
		return Quaternion.Inverse(to.GetRotation()) * rotation;
	}

	public static Vector3 GetRight(this Quaternion quaternion) {
		return quaternion * Vector3.right;
	}

	public static Vector3 GetUp(this Quaternion quaternion) {
		return quaternion * Vector3.up;
	}

	public static Vector3 GetForward(this Quaternion quaternion) {
		return quaternion * Vector3.forward;
	}

	public static Quaternion GetMirror(this Quaternion quaternion, Vector3 axis) {
		Quaternion mirror = quaternion;
		if(axis == Vector3.right) {
			mirror.x *= -1f;
			mirror.w *= -1f;
		}
		if(axis == Vector3.up) {
			mirror.y *= -1f;
			mirror.w *= -1f;
		}
		if(axis == Vector3.forward) {
			mirror.z *= -1f;
			mirror.w *= -1f;
		}
		return Quaternion.Slerp(quaternion, mirror, 1f);
	}

	public static Quaternion GetNormalised(this Quaternion rotation) {
		float length = rotation.GetMagnitude();
		rotation.x /= length;
		rotation.y /= length;
		rotation.z /= length;
		rotation.w /= length;
		return rotation;
	}

	public static float GetMagnitude(this Quaternion rotation) {
		return Mathf.Sqrt(rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z + rotation.w*rotation.w);
	}

	public static Quaternion GetInverse(this Quaternion rotation) {
		return Quaternion.Inverse(rotation);
	}

	public static Quaternion GetLog(this Quaternion rotation) {
		float mag = rotation.GetMagnitude();
		float arg = (float)System.Math.Atan2(mag, rotation.w) / mag;
		rotation.x *= arg;
		rotation.y *= arg;
		rotation.z *= arg;
		rotation.w = 0f;
		return rotation;
	}
	
    public static Quaternion GetExp(this Quaternion rotation) {
		float w = (float)System.Math.Sqrt(rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z);
		Quaternion exp = Quaternion.identity;
		exp.x = rotation.x * (float)System.Math.Sin(w) / w;
		exp.y = rotation.y * (float)System.Math.Sin(w) / w;
		exp.z = rotation.z * (float)System.Math.Sin(w) / w;
		exp.w = (float)System.Math.Cos(w);
		return exp;//exp.GetNormalised();
    }

}
