using UnityEngine;

public static class Vector3Extensions {

	public static Vector3 GetRelativePositionFrom(this Vector3 position, Matrix4x4 from) {
		return from.MultiplyPoint(position);
	}

	public static Vector3 GetRelativePositionTo(this Vector3 position, Matrix4x4 to) {
		return to.inverse.MultiplyPoint(position);
	}

	public static Vector3 GetRelativeDirectionFrom(this Vector3 direction, Matrix4x4 from) {
		return from.MultiplyVector(direction);
	}

	public static Vector3 GetRelativeDirectionTo(this Vector3 direction, Matrix4x4 to) {
		return to.inverse.MultiplyVector(direction);
	}

	public static Vector3 GetMirror(this Vector3 vector, Vector3 axis) {
		if(axis == Vector3.right) {
			vector.x *= -1f;
		}
		if(axis == Vector3.up) {
			vector.y *= -1f;
		}
		if(axis == Vector3.forward) {
			vector.z *= -1f;
		}
		return vector;
	}

}
