using UnityEngine;

public static class Transformation {

	public static Matrix4x4 SetPosition(this Matrix4x4 matrix, Vector3 position) {
		return Matrix4x4.TRS(position, matrix.GetRotation(), matrix.GetScale());
	}

	public static Vector3 GetPosition(this Matrix4x4 matrix) {
		return matrix.GetColumn(3);
	}

	public static Matrix4x4 SetRotation(this Matrix4x4 matrix, Quaternion rotation) {
		return Matrix4x4.TRS(matrix.GetPosition(), rotation, matrix.GetScale());
	}
	
	public static Quaternion GetRotation(this Matrix4x4 matrix) {
		return Quaternion.LookRotation(matrix.GetColumn(2).normalized, matrix.GetColumn(1).normalized);
	}

	public static Matrix4x4 SetScale(this Matrix4x4 matrix, Vector3 scale) {
		return Matrix4x4.TRS(matrix.GetPosition(), matrix.GetRotation(), scale);
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

	public static Matrix4x4 GetMirroredZ(this Matrix4x4 matrix) {
		matrix[2, 3] *= -1f; //Pos
		matrix[0, 2] *= -1f; //Rot
		matrix[1, 2] *= -1f; //Rot
		matrix[2, 0] *= -1f; //Rot
		matrix[2, 1] *= -1f; //Rot
		return matrix;
	}

	/*
	public static Quaternion GetAbsolute(this Quaternion rotation) {
		rotation = rotation.GetNormalised();
		//float imag = rotation.x + rotation.y + rotation.z;
		//float real = rotation.w;
		//if(imag < 0f && real > 0f) {
		
		//float scalar = Mathf.Sign(rotation.x) * rotation.x *rotation.x + Mathf.Sign(rotation.y) * rotation.y *rotation.y + Mathf.Sign(rotation.z) * rotation.z *rotation.z;
		//if(rotation.w < 0f || scalar < 0f) {
		if(rotation.w < 0f) {
			rotation.x *= -1f;
			rotation.y *= -1f;
			rotation.z *= -1f;
			rotation.w *= -1f;
		}

        return rotation;
	}
	*/

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
	
	/*
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
	*/

	/*
	public static Quaternion Log2(this Quaternion rotation, bool makeAbs) {
		Quaternions.Quaternion quat = new Quaternions.Quaternion(rotation.w, rotation.x, rotation.y, rotation.z).Log();
		return new Quaternion((float)quat.ImagX, (float)quat.ImagY, (float)quat.ImagZ, (float)quat.Real);
	}

	public static Quaternion Exp2(this Quaternion rotation) {
		Quaternions.Quaternion quat = new Quaternions.Quaternion(rotation.w, rotation.x, rotation.y, rotation.z).Exp();
		return new Quaternion((float)quat.ImagX, (float)quat.ImagY, (float)quat.ImagZ, (float)quat.Real);
	}

	public static Quaternion LogNew(this Quaternion rotation) {
		float v_length = new Vector3(rotation.x, rotation.y, rotation.z).magnitude;
		float q_length = rotation.GetMagnitude();
		Vector3 vec = new Vector3(rotation.x, rotation.y, rotation.z) / v_length;
		vec *= Mathf.Acos(rotation.w / q_length);
		return new Quaternion(vec.x, vec.y, vec.z, Mathf.Log(q_length));
	}

	public static Quaternion LogMap(this Quaternion quaternion) {
		return ;
	}
	*/

	/*
	public static Quaternion ExpNew(this Quaternion rotation) {
    @classmethod
    def exp(cls, q):
        """Quaternion Exponential.
           
        Find the exponential of a quaternion amount.
        Params:
             q: the input quaternion/argument as a Quaternion object.
        Returns:
             A quaternion amount representing the exp(q). See [Source](https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation for more information and mathematical background).
           
        Note:
             The method can compute the exponential of any quaternion.
        """
        tolerance = 1e-17
        v_norm = np.linalg.norm(q.vector)
        vec = q.vector
        if v_norm > tolerance:
            vec = vec / v_norm
        magnitude = exp(q.scalar)
        return Quaternion(scalar = magnitude * cos(v_norm), vector = magnitude * sin(v_norm) * vec)

	}
	*/

	/*
	public static Quaternion Exp(this Quaternion rotation) {
		float ts = Mathf.Sqrt(rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z);
		float ls = Mathf.Sin(ts) / ts;
		
    
        qs = np.empty(ws.shape[:-1] + (4,))
        qs[...,0] = np.cos(ts)
        qs[...,1] = ws[...,0] * ls
        qs[...,2] = ws[...,1] * ls
        qs[...,3] = ws[...,2] * ls
        
        return Quaternions(qs).normalized()
		return quaternion;
	}
	*/
}