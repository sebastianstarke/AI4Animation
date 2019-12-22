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

	public static Quaternion GetAverage(this Quaternion[] quaternions) {
		if(quaternions.Length == 0) {
			return Quaternion.identity;
		}
		if(quaternions.Length == 1) {
			return quaternions[1];
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

	public static Vector3 GetLog(this Quaternion rotation) {
		//Quaternion exp_w = rotation.GetNormalised();
		//Quaternion w = 
		/*
		Vector3 log = Vector3.zero;
		float mag = rotation.GetMagnitude();
		float arg = (float)System.Math.Atan2(mag, rotation.w) / mag;
		log.x = rotation.x * arg;
		log.y = rotation.y * arg;
		log.z = rotation.z * arg;
		return log;
		*/

	/*
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _QUATERNION_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      // fprintf(stderr, "Input quaternion(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
      if(fabs(q.w+1)>_QUATERNION_EPS) {
        quaternion r = {log(-q.w), M_PI, 0., 0.};
        return r;
      } else {
        quaternion r = {0., M_PI, 0., 0.};
        return r;
      }
    } else {
      quaternion r = {log(q.w), 0., 0., 0.};
      return r;
    }
  } else {
    double v = atan2(b, q.w);
    double f = v/b;
    quaternion r = { log(q.w*q.w+b*b)/2.0, f*q.x, f*q.y, f*q.z };
    return r;
  }
	*/

		Quaternion q = rotation.GetNormalised();
		float b = q.GetMagnitude();
		float v = Mathf.Atan2(b, q.w);
		float f = v/b;
		Quaternion r = new Quaternion(f*q.x, f*q.y, f*q.z, Mathf.Log(q.w*q.w+b*b)/2f);

		Vector3 log = new Vector3(r.x, r.y, r.z);
		log.x = Mathf.Abs(log.x);
		log.y = Mathf.Abs(log.y);
		log.z = Mathf.Abs(log.z);

		return log;
	}
	
    public static Quaternion GetExp(this Vector3 rotation) {
		float w = (float)System.Math.Sqrt(rotation.x*rotation.x + rotation.y*rotation.y + rotation.z*rotation.z);
		Quaternion exp = Quaternion.identity;
		exp.x = rotation.x * (float)System.Math.Sin(w) / w;
		exp.y = rotation.y * (float)System.Math.Sin(w) / w;
		exp.z = rotation.z * (float)System.Math.Sin(w) / w;
		exp.w = (float)System.Math.Cos(w);
		return exp;//exp.GetNormalised();
    }

}
