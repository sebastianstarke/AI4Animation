using UnityEngine;

public static class QuaternionExtensions {

	public static Quaternion GetRelativeRotationFrom(this Quaternion rotation, Matrix4x4 from) {
		return from.GetRotation() * rotation;
	}

	public static Quaternion GetRelativeRotationTo(this Quaternion rotation, Matrix4x4 to) {
		return Quaternion.Inverse(to.GetRotation()) * rotation;
	}

	public static Quaternion GetRotationFromTo(this Quaternion rotation, Matrix4x4 from, Matrix4x4 to) {
		return rotation.GetRelativeRotationTo(from).GetRelativeRotationFrom(to);
	}

	public static Quaternion GetRelativeRotationFrom(this Quaternion rotation, Quaternion from) {
		return from * rotation;
	}

	public static Quaternion GetRelativeRotationTo(this Quaternion rotation, Quaternion to) {
		return Quaternion.Inverse(to) * rotation;
	}

	public static Quaternion GetRotationFromTo(this Quaternion rotation, Quaternion from, Quaternion to) {
		return rotation.GetRelativeRotationTo(from).GetRelativeRotationFrom(to);
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

	public static Quaternion GetMirror(this Quaternion quaternion, Axis axis) {
		Quaternion mirror = quaternion;
		if(axis == Axis.XPositive) {
			mirror.x *= -1f;
			mirror.w *= -1f;
		}
		if(axis == Axis.YPositive) {
			mirror.y *= -1f;
			mirror.w *= -1f;
		}
		if(axis == Axis.ZPositive) {
			mirror.z *= -1f;
			mirror.w *= -1f;
		}
		return Quaternion.Slerp(quaternion, mirror, 1f);
	}

	public static Quaternion LookRotation(Vector3 forward, Vector3 up) {
		Vector3 cross = Vector3.Cross(forward, up);
		Vector3 mean = ((forward + up) / 2f).normalized;
		forward = Quaternion.AngleAxis(-45f, cross) * mean;
		up = Quaternion.AngleAxis(45f, cross) * mean;
		return Quaternion.LookRotation(forward, up);
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

	public static Quaternion Mean(this Quaternion[] quaternions) {
		if(quaternions.Length == 0) {
			return Quaternion.identity;
		}
		if(quaternions.Length == 1) {
			return quaternions[0];
		}
		if(quaternions.Length == 2) {
			return Quaternion.Slerp(quaternions[0], quaternions[1], 0.5f);
		}
		Vector3 forward = Vector3.zero;
		Vector3 upwards = Vector3.zero;
		for(int i=0; i<quaternions.Length; i++) {
			forward += quaternions[i] * Vector3.forward;
			upwards += quaternions[i] * Vector3.up;
		}
		forward /= quaternions.Length;
		upwards /= quaternions.Length;
		return Quaternion.LookRotation(forward, upwards);
	}

	public static Quaternion Mean(this Quaternion[] quaternions, float[] weights) {
		if(quaternions.Length == 0) {
			return Quaternion.identity;
		}
		if(quaternions.Length == 1) {
			return quaternions[0];
		}
		if(quaternions.Length != weights.Length) {
			Debug.Log("Failed to compute mean because size of vectors and weights does not match.");
			return Quaternion.identity;
		}
		float sum = 0f;
		Vector3 forwards = Vector3.zero;
		Vector3 upwards = Vector3.zero;
		for(int i=0; i<quaternions.Length; i++) {
			forwards += weights[i] * (quaternions[i] * Vector3.forward);
			upwards += weights[i] * (quaternions[i] * Vector3.up);
			sum += weights[i];
		}
		forwards /= quaternions.Length;
		upwards /= quaternions.Length;
		if(sum == 0f) {
			Debug.Log("Failed to compute mean because size of sum of weights is zero.");
			return Quaternion.identity;
		}
		return Quaternion.LookRotation((forwards/sum).normalized, (upwards/sum).normalized);
	}

	private static Vector3[] GaussianForwards = null;
	private static Vector3[] GaussianUpwards = null;
	public static Quaternion Gaussian(this Quaternion[] values, float power=1f) {
		if(values.Length == 0) {
			return Quaternion.identity;
		}
		if(values.Length == 1) {
			return values[0];
		}
		GaussianForwards = GaussianForwards.Validate(values.Length);
		GaussianUpwards = GaussianUpwards.Validate(values.Length);
		for(int i=0; i<values.Length; i++) {
			GaussianForwards[i] = values[i].GetForward();
			GaussianUpwards[i] = values[i].GetUp();
		}
		return Quaternion.LookRotation(GaussianForwards.Gaussian(power).normalized, GaussianUpwards.Gaussian(power).normalized);
	}

	private static Vector3[] MaskedGaussianForwards = null;
	private static Vector3[] MaskedGaussianUpwards = null;
	public static Quaternion Gaussian(this Quaternion[] values, bool[] mask, float power=1f) {
		if(values.Length == 0) {
			return Quaternion.identity;
		}
		if(values.Length == 1) {
			return values[0];
		}
		MaskedGaussianForwards = MaskedGaussianForwards.Validate(values.Length);
		MaskedGaussianUpwards = MaskedGaussianUpwards.Validate(values.Length);
		for(int i=0; i<values.Length; i++) {
			MaskedGaussianForwards[i] = values[i].GetForward();
			MaskedGaussianUpwards[i] = values[i].GetUp();
		}
		return Quaternion.LookRotation(MaskedGaussianForwards.Gaussian(mask, power).normalized, MaskedGaussianUpwards.Gaussian(mask, power).normalized);
	}

	public static float[] ToArray(this Quaternion vector) {
		return new float[4]{vector.x, vector.y, vector.z, vector.w};
	}

	public static float[][] ToArray(this Quaternion[] vectors) {
		float[][] values = new float[vectors.Length][];
		for(int i=0; i<values.Length; i++) {
			values[i] = vectors[i].ToArray();
		}
		return values;
	}

}
