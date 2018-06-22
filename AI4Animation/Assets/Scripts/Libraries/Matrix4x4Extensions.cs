using UnityEngine;

public static class Matrix4x4Extensions {

	public static void SetPosition(ref Matrix4x4 matrix, Vector3 position) {
		matrix[0,3] = position.x;
		matrix[1,3] = position.y;
		matrix[2,3] = position.z;
	}

	public static void SetRotation(ref Matrix4x4 matrix, Quaternion rotation) {
		Vector3 right = rotation.GetRight();
		Vector3 up = rotation.GetUp();
		Vector3 forward = rotation.GetForward();
		matrix[0,0] = right.x;
		matrix[1,0] = right.y;
		matrix[2,0] = right.z;
		matrix[0,1] = up.x;
		matrix[1,1] = up.y;
		matrix[2,1] = up.z;
		matrix[0,2] = forward.x;
		matrix[1,2] = forward.y;
		matrix[2,2] = forward.z;
	}

	public static void SetScale(ref Matrix4x4 matrix, Vector3 scale) {
		matrix = Matrix4x4.TRS(matrix.GetPosition(), matrix.GetRotation(), scale);
	}

	public static Vector3 GetPosition(this Matrix4x4 matrix) {
		return new Vector3(matrix[0,3], matrix[1,3], matrix[2,3]);
	}
	
	public static Quaternion GetRotation(this Matrix4x4 matrix) {
		return Quaternion.LookRotation(matrix.GetColumn(2).normalized, matrix.GetColumn(1).normalized);
	}

	public static Vector3 GetScale(this Matrix4x4 matrix) {
		return new Vector3(matrix.GetColumn(0).magnitude, matrix.GetColumn(1).magnitude, matrix.GetColumn(2).magnitude);
	}

	public static Vector3 GetRight(this Matrix4x4 matrix) {
		return new Vector3(matrix[0,0], matrix[1,0], matrix[2,0]).normalized;
	}

	public static Vector3 GetUp(this Matrix4x4 matrix) {
		return new Vector3(matrix[0,1], matrix[1,1], matrix[2,1]).normalized;
	}

	public static Vector3 GetForward(this Matrix4x4 matrix) {
		return new Vector3(matrix[0,2], matrix[1,2], matrix[2,2]).normalized;
	}

	public static Matrix4x4 GetRelativeTransformationFrom(this Matrix4x4 matrix, Matrix4x4 from) {
		return from * matrix;
	}

	public static Matrix4x4 GetRelativeTransformationTo(this Matrix4x4 matrix, Matrix4x4 to) {
		return to.inverse * matrix;
	}

	public static Matrix4x4 GetMirror(this Matrix4x4 matrix, Vector3 axis) {
		if(axis == Vector3.right) { //X-Axis
			matrix[0, 3] *= -1f; //Pos
			matrix[0, 1] *= -1f; //Rot
			matrix[0, 2] *= -1f; //Rot
			matrix[1, 0] *= -1f; //Rot
			matrix[2, 0] *= -1f; //Rot
		}
		if(axis == Vector3.up) { //Y-Axis
			Debug.Log("Mirroring against Y-axis not yet implemented");
		}
		/*
			matrix[1, 3] *= -1f; //Pos
			matrix[0, 0] *= -1f; //Rot
			matrix[0, 2] *= -1f; //Rot
			matrix[2, 0] *= -1f; //Rot
			matrix[2, 2] *= -1f; //Rot
		*/
		// should be 0,1 1,0 1,2 2,1 maybe
		if(axis == Vector3.forward) { //Z-Axis
			matrix[2, 3] *= -1f; //Pos
			matrix[0, 2] *= -1f; //Rot
			matrix[1, 2] *= -1f; //Rot
			matrix[2, 0] *= -1f; //Rot
			matrix[2, 1] *= -1f; //Rot
		}
		return matrix;
	}

}
