using UnityEngine;

public static class Transformations {

	public static Matrix4x4 GetLocalMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.localPosition, transform.localRotation, Vector3.one);
	}

	public static Matrix4x4 GetWorldMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
	}

	public static void SetPosition(ref Matrix4x4 matrix, Vector3 position) {
		matrix = Matrix4x4.TRS(position, matrix.GetRotation(), matrix.GetScale());
	}

	public static void SetRotation(ref Matrix4x4 matrix, Quaternion rotation) {
		matrix = Matrix4x4.TRS(matrix.GetPosition(), rotation, matrix.GetScale());
	}

	public static void SetScale(ref Matrix4x4 matrix, Vector3 scale) {
		matrix = Matrix4x4.TRS(matrix.GetPosition(), matrix.GetRotation(), scale);
	}

	public static Vector3 GetPosition(this Matrix4x4 matrix) {
		return matrix.GetColumn(3);
	}
	
	public static Quaternion GetRotation(this Matrix4x4 matrix) {
		return Quaternion.LookRotation(matrix.GetColumn(2).normalized, matrix.GetColumn(1).normalized);
	}

	public static Vector3 GetScale(this Matrix4x4 matrix) {
		return new Vector3(matrix.GetColumn(0).magnitude, matrix.GetColumn(1).magnitude, matrix.GetColumn(2).magnitude);
	}

	public static Vector3 GetRight(this Matrix4x4 matrix) {
		return matrix.GetColumn(0);
	}

	public static Vector3 GetUp(this Matrix4x4 matrix) {
		return matrix.GetColumn(1);
	}

	public static Vector3 GetForward(this Matrix4x4 matrix) {
		return matrix.GetColumn(2);
	}

	public static Matrix4x4 GetRelativeTransformationFrom(this Matrix4x4 matrix, Matrix4x4 from) {
		return from * matrix;
	}

	public static Matrix4x4 GetRelativeTransformationTo(this Matrix4x4 matrix, Matrix4x4 to) {
		return to.inverse * matrix;
	}

	public static Vector3 GetRelativePositionFrom(this Vector3 position, Matrix4x4 from) {
		return from.MultiplyPoint(position);
	}

	public static Vector3 GetRelativePositionTo(this Vector3 position, Matrix4x4 to) {
		return to.inverse.MultiplyPoint(position);
	}

	public static Quaternion GetRelativeRotationFrom(this Quaternion rotation, Matrix4x4 from) {
		return (from * Matrix4x4.TRS(Vector3.zero, rotation, Vector3.one)).GetRotation();
	}

	public static Quaternion GetRelativeRotationTo(this Quaternion rotation, Matrix4x4 to) {
		return (to.inverse * Matrix4x4.TRS(Vector3.zero, rotation, Vector3.one)).GetRotation();
	}

	public static Vector3 GetRelativeDirectionFrom(this Vector3 direction, Matrix4x4 from) {
		return from.MultiplyVector(direction);
	}

	public static Vector3 GetRelativeDirectionTo(this Vector3 direction, Matrix4x4 to) {
		return to.inverse.MultiplyVector(direction);
	}

	public static Matrix4x4 GetMirror(this Matrix4x4 matrix) {
		matrix[2, 3] *= -1f; //Pos
		matrix[0, 2] *= -1f; //Rot
		matrix[1, 2] *= -1f; //Rot
		matrix[2, 0] *= -1f; //Rot
		matrix[2, 1] *= -1f; //Rot
		return matrix;
	}

	public static Matrix4x4 GetMirror(this Matrix4x4 matrix, Vector3 axisPos) {
		if(axisPos == Vector3.right) {
			matrix[0, 3] *= -1f; //Pos
		}
		if(axisPos == Vector3.up) {
			matrix[1, 3] *= -1f; //Pos
		}
		if(axisPos == Vector3.forward) {
			matrix[2, 3] *= -1f; //Pos
			matrix[0, 2] *= -1f; //Rot
			matrix[1, 2] *= -1f; //Rot
			matrix[2, 0] *= -1f; //Rot
			matrix[2, 1] *= -1f; //Rot
		}
		return matrix;
	}

	public static Vector3 GetMirror(this Vector3 vector) {
		vector.z *= -1f;
		return vector;
	}

	public static Quaternion GetMirror(this Quaternion quaternion) {
		Quaternion mirror = quaternion;
		mirror.z *= -1f;
		mirror.w *= -1f;
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

	public static Quaternion Log(this Quaternion rotation) {
		float mag = rotation.GetMagnitude();
		float arg = Mathf.Atan2(mag, rotation.w) / mag;
		rotation.x *= arg;
		rotation.y *= arg;
		rotation.z *= arg;
		rotation.w = 0f;
		return rotation;
	}
	
    public static Quaternion Exp(this Quaternion rotation) {
		float w = Mathf.Sqrt(rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z);
		Quaternion exp = new Quaternion(
			rotation.x * Mathf.Sin(w) / w,
			rotation.y * Mathf.Sin(w) / w,
			rotation.z * Mathf.Sin(w) / w,
			Mathf.Cos(w)
		);
		return exp.GetNormalised();
    }

}