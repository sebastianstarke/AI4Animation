using UnityEngine;

public static class Matrix4x4Extensions {

	public static Matrix4x4 TransformationFrom(this Matrix4x4 matrix, Matrix4x4 from) {
		return from * matrix;
	}

	public static Matrix4x4 TransformationTo(this Matrix4x4 matrix, Matrix4x4 to) {
		return to.inverse * matrix;
	}

	public static Matrix4x4 TransformationFromTo(this Matrix4x4 matrix, Matrix4x4 from, Matrix4x4 to) {
		return matrix.TransformationTo(from).TransformationFrom(to);
	}

	public static Matrix4x4[] TransformationsFrom(this Matrix4x4[] matrices, Matrix4x4 from, bool inplace) {
		Matrix4x4[] result = inplace ? matrices : new Matrix4x4[matrices.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = matrices[i].TransformationFrom(from);
		}
		return result;
	}

	public static Matrix4x4[] TransformationsTo(this Matrix4x4[] matrices, Matrix4x4 to, bool inplace) {
		Matrix4x4[] result = inplace ? matrices : new Matrix4x4[matrices.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = matrices[i].TransformationTo(to);
		}
		return result;
	}

	public static Matrix4x4[] TransformationsFromTo(this Matrix4x4[] matrices, Matrix4x4 from, Matrix4x4 to, bool inplace) {
		Matrix4x4[] result = inplace ? matrices : new Matrix4x4[matrices.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = matrices[i].TransformationFromTo(from, to);
		}
		return result;
	}

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
		return matrix.rotation;
		// return Quaternion.LookRotation(matrix.GetColumn(2), matrix.GetColumn(1));
	}

	public static Vector3 GetScale(this Matrix4x4 matrix) {
		return matrix.lossyScale;
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

	public static Matrix4x4 GetMirror(this Matrix4x4 matrix, Axis axis) {
		if(axis == Axis.XPositive) { //X-Axis
			matrix[0, 3] *= -1f; //Pos
			matrix[0, 1] *= -1f; //Rot
			matrix[0, 2] *= -1f; //Rot
			matrix[1, 0] *= -1f; //Rot
			matrix[2, 0] *= -1f; //Rot
		}
		if(axis == Axis.YPositive) { //Y-Axis
			matrix[1, 3] *= -1f; //Pos
			matrix[1, 0] *= -1f; //Rot
			matrix[1, 2] *= -1f; //Rot
			matrix[0, 1] *= -1f; //Rot
			matrix[2, 1] *= -1f; //Rot
		}
		if(axis == Axis.ZPositive) { //Z-Axis
			matrix[2, 3] *= -1f; //Pos
			matrix[2, 0] *= -1f; //Rot
			matrix[2, 1] *= -1f; //Rot
			matrix[0, 2] *= -1f; //Rot
			matrix[1, 2] *= -1f; //Rot
		}
		return matrix;
	}

	public static Matrix4x4 Mean(this Matrix4x4[] matrices) {
		if(matrices.Length == 0) {
			return Matrix4x4.identity;
		}
		if(matrices.Length == 1) {
			return matrices[0];
		}
		if(matrices.Length == 2) {
			return Utility.Interpolate(matrices[0], matrices[1], 0.5f);
		}
		return Matrix4x4.TRS(matrices.GetPositions().Mean(), matrices.GetRotations().Mean(), matrices.GetScales().Mean());
	}

	public static Matrix4x4 Mean(this Matrix4x4[] matrices, float[] weights) {
		if(matrices.Length == 0) {
			return Matrix4x4.identity;
		}
		if(matrices.Length == 1) {
			return matrices[0];
		}
		if(matrices.Length != weights.Length) {
			Debug.Log("Failed to compute mean because size of matrices and weights does not match.");
			return Matrix4x4.identity;
		}
		float sum = 0f;
		Vector3 position = Vector3.zero;
		Vector3 forward = Vector3.zero;
		Vector3 up = Vector3.zero;
		for(int i=0; i<matrices.Length; i++) {
			sum += weights[i];
			position += weights[i] * matrices[i].GetPosition();
			forward += weights[i] * matrices[i].GetForward();
			up += weights[i] * matrices[i].GetUp();
		}
		if(sum == 0f) {
			Debug.Log("Failed to compute mean because size of sum of weights is zero.");
			return Matrix4x4.identity;
		}
		return Matrix4x4.TRS(
			position/sum,
			Quaternion.LookRotation((forward/sum).normalized, (up/sum).normalized),
			Vector3.one
		);
	}

	public static Vector3[] GetPositions(this Matrix4x4[] matrices) {
		Vector3[] values = new Vector3[matrices.Length];
		for(int i=0; i<values.Length; i++) {
			values[i] = matrices[i].GetPosition();
		}
		return values;
	}

	public static Quaternion[] GetRotations(this Matrix4x4[] matrices) {
		Quaternion[] values = new Quaternion[matrices.Length];
		for(int i=0; i<values.Length; i++) {
			values[i] = matrices[i].GetRotation();
		}
		return values;
	}

	public static Vector3[] GetScales(this Matrix4x4[] matrices) {
		Vector3[] values = new Vector3[matrices.Length];
		for(int i=0; i<values.Length; i++) {
			values[i] = matrices[i].GetScale();
		}
		return values;
	}

}
