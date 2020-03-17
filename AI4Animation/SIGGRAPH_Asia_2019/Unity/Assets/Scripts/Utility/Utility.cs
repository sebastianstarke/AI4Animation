using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

#if UNITY_EDITOR
using UnityEditor;
#endif

public static class Utility {

	private static Collider[] Colliders = new Collider[128];

	public static bool InvalidClosestPointCheck(Collider collider) {
		if(collider.isTrigger) {
			//Debug.Log("Invalid Closest Point Check: " + collider.name + " Parent: " + collider.transform.parent.name + " Type: Trigger");
			return true;
		}
		if(collider is MeshCollider) {
			Debug.Log("Invalid Closest Point Check: " + collider.name + " Parent: " + collider.transform.name + " Parent: " + collider.transform.parent.name + " Type: MeshCollider");
			return true;
		}
		return false;
	}

	public static Vector3 GetClosestPointOverlapBox(Vector3 center, Vector3 halfExtents, Quaternion rotation, LayerMask mask, out Collider collider) {
		Collider[] colliders = Physics.OverlapBox(center, halfExtents, rotation, mask);
		if(colliders.Length == 0) {
			collider = null;
			return center;
		}
		int pivot = 0;
		while(InvalidClosestPointCheck(colliders[pivot])) {
			pivot++;
			if(pivot == colliders.Length) {
				collider = null;
				return center;
			}
		}
		Vector3 point = colliders[pivot].ClosestPoint(center);
		float x = (point.x-center.x)*(point.x-center.x);
		float y = (point.y-center.y)*(point.y-center.y);
		float z = (point.z-center.z)*(point.z-center.z);
		float min = x*x + y*y + z*z;
		collider = colliders[pivot];
		for(int i=pivot+1; i<colliders.Length; i++) {
			if(!InvalidClosestPointCheck(colliders[pivot])) {
				Vector3 candidate = colliders[i].ClosestPoint(center);
				x = (candidate.x-center.x)*(candidate.x-center.x);
				y = (candidate.y-center.y)*(candidate.y-center.y);
				z = (candidate.z-center.z)*(candidate.z-center.z);
				float d = x*x + y*y + z*z;
				if(d < min) {
					point = candidate;
					min = d;
					collider = colliders[i];
				}
			}
		}
		return point;
	}

	public static float SmoothStep(float edge0, float edge1, float x) {
		// Scale, bias and saturate x to 0..1 range
		x = Mathf.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
		// Evaluate polynomial
		return  x * x * (3f - 2f * x);
	}

	public static Vector3 GetClosestPointOverlapSphere(Vector3 center, float radius, LayerMask mask, out Collider collider) {
		Collider[] colliders = Physics.OverlapSphere(center, radius, mask);
		if(colliders.Length == 0) {
			collider = null;
			return center;
		}
		int pivot = 0;
		while(InvalidClosestPointCheck(colliders[pivot])) {
			pivot++;
			if(pivot == colliders.Length) {
				collider = null;
				return center;
			}
		}
		Vector3 point = colliders[pivot].ClosestPoint(center);
		float x = (point.x-center.x)*(point.x-center.x);
		float y = (point.y-center.y)*(point.y-center.y);
		float z = (point.z-center.z)*(point.z-center.z);
		float min = x*x + y*y + z*z;
		collider = colliders[pivot];
		for(int i=pivot+1; i<colliders.Length; i++) {
			if(!InvalidClosestPointCheck(colliders[pivot])) {
				Vector3 candidate = colliders[i].ClosestPoint(center);
				x = (candidate.x-center.x)*(candidate.x-center.x);
				y = (candidate.y-center.y)*(candidate.y-center.y);
				z = (candidate.z-center.z)*(candidate.z-center.z);
				float d = x*x + y*y + z*z;
				if(d < min) {
					point = candidate;
					min = d;
					collider = colliders[i];
				}
			}
		}
		return point;
	}

	public static System.Random RNG;

	public static System.Random GetRNG() {
		if(RNG == null) {
			RNG = new System.Random();
		}
		return RNG;
	}

	public static Quaternion QuaternionEuler(float roll, float pitch, float yaw) {
		roll *= Mathf.Deg2Rad / 2f;
		pitch *= Mathf.Deg2Rad / 2f;
		yaw *= Mathf.Deg2Rad / 2f;

		Vector3 Z = Vector3.forward;
		Vector3 X = Vector3.right;
		Vector3 Y = Vector3.up;

		float sin, cos;

		sin = (float)System.Math.Sin(roll);
		cos = (float)System.Math.Cos(roll);
		Quaternion q1 = new Quaternion(0f, 0f, Z.z * sin, cos);
		sin = (float)System.Math.Sin(pitch);
		cos = (float)System.Math.Cos(pitch);
		Quaternion q2 = new Quaternion(X.x * sin, 0f, 0f, cos);
		sin = (float)System.Math.Sin(yaw);
		cos = (float)System.Math.Cos(yaw);
		Quaternion q3 = new Quaternion(0f, Y.y * sin, 0f, cos);

		return MultiplyQuaternions(MultiplyQuaternions(q1, q2), q3);
	}

	public static Quaternion MultiplyQuaternions(Quaternion q1, Quaternion q2) {
		float x =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
		float y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
		float z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
		float w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
		return new Quaternion(x, y, z, w);
	}

	public static void Swap<T>(ref T lhs, ref T rhs) {
        T temp = lhs;
        lhs = rhs;
        rhs = temp;
    }

	public static void Screenshot(string name, int x, int y, int width, int height) {
    	Texture2D tex = new Texture2D(width, height);
		tex.ReadPixels(new Rect(x, y, width, height), 0, 0);
		tex.Apply();
		byte[] bytes = tex.EncodeToPNG();
    	File.WriteAllBytes(name + ".png", bytes);
		Destroy(tex);
	}

	public static void SetFPS(int fps) {
		#if UNITY_EDITOR
		QualitySettings.vSyncCount = 0;
		#else
		QualitySettings.vSyncCount = 0;
		#endif
		Application.targetFrameRate = fps;
	}

	public static System.DateTime GetTimestamp() {
		return System.DateTime.Now;
	}

	public static double GetElapsedTime(System.DateTime timestamp) {
		return (System.DateTime.Now-timestamp).Duration().TotalSeconds;
	}

	public static float Exponential01(float value) {
		float basis = 2f;
		return (Mathf.Pow(basis, Mathf.Clamp(value, 0f, 1f)) - 1f) / (basis-1f);
	}

	public static float Normalise(float value, float valueMin, float valueMax, float resultMin, float resultMax) {
		if(valueMax-valueMin != 0f) {
			return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin;
		} else {
			//Not possible to normalise input value.
			return value;
		}
	}

	public static double Normalise(double value, double valueMin, double valueMax, double resultMin, double resultMax) {
		if(valueMax-valueMin != 0.0) {
			return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin;
		} else {
			//Not possible to normalise input value.
			return value;
		}
	}

	public static Vector3 Normalise(Vector3 value, Vector3 valueMin, Vector3 valueMax, Vector3 resultMin, Vector3 resultMax) {
		return new Vector3(
			Normalise(value.x, valueMin.x, valueMax.x, resultMin.x, resultMax.x),
			Normalise(value.y, valueMin.y, valueMax.y, resultMin.y, resultMax.y),
			Normalise(value.z, valueMin.z, valueMax.z, resultMin.z, resultMax.z)
		);
	}

	//0 = Amplitude, 1 = Frequency, 2 = Shift, 3 = Offset, 4 = Slope, 5 = Time
	public static float LinSin(float a, float f, float s, float o, float m, float t) {
		return a * Mathf.Sin(f * (t - s) * 2f * Mathf.PI) + o + m * t;
	}

	public static float LinSin1(float a, float f, float s, float o, float m, float t) {
		return a * f * Mathf.Cos(f * (t - s) * 2f * Mathf.PI) + m;
	}

	public static float LinSin2(float a, float f, float s, float o, float m, float t) {
		return a * f * f * -Mathf.Sin(f * (t - s) * 2f * Mathf.PI);
	}

	public static float LinSin3(float a, float f, float s, float o, float m, float t) {
		return a * f * f * f * -Mathf.Cos(f * (t - s) * 2f * Mathf.PI);
	}

	public static float Gaussian(float mean, float std, float x) {
		return 1f/(std*Mathf.Sqrt(2f*Mathf.PI)) * Mathf.Exp(-0.5f*(x*x)/(std*std));
	}
	
	public static double Sigmoid(double x) {
		return 1f / (1f + System.Math.Exp(-x));
	}

	public static float Sigmoid(float x) {
		return 1f / (1f + Mathf.Exp(-x));
	}

	public static double TanH(double x) {
		double positive = System.Math.Exp(x);
		double negative = System.Math.Exp(-x);
		return (positive-negative) / (positive+negative);
	}

	public static float TanH(float x) {
		float positive = Mathf.Exp(x);
		float negative = Mathf.Exp(-x);
		return (positive-negative) / (positive+negative);
	}

	public static float Interpolate(float from, float to, float amount) {
		amount = Mathf.Clamp(amount,0f,1f);
		return (1f-amount)*from + amount*to;
	}

	public static double Interpolate(double from, double to, float amount) {
		amount = Mathf.Clamp(amount,0f,1f);
		return (1f-amount)*from + amount*to;
	}

	public static Vector2 Interpolate(Vector2 from, Vector2 to, float amount) {
		amount = Mathf.Clamp(amount,0f,1f);
		return (1f-amount)*from + amount*to;
	}

	public static Vector3 Interpolate(Vector3 from, Vector3 to, float amount) {
		amount = Mathf.Clamp(amount,0f,1f);
		return (1f-amount)*from + amount*to;
	}

	public static Quaternion Interpolate(Quaternion from, Quaternion to, float amount) {
		amount = Mathf.Clamp(amount,0f,1f);
		return Quaternion.Slerp(from,to,amount);
	}

	public static Matrix4x4 Interpolate(Matrix4x4 from, Matrix4x4 to, float amount) {
		return Matrix4x4.TRS(Interpolate(from.GetPosition(), to.GetPosition(), amount), Interpolate(from.GetRotation(), to.GetRotation(), amount), Vector3.one);
	}

	public static float[] Interpolate(float[] from, float[] to, float amount) {
		if(from.Length != to.Length) {
			Debug.Log("Interpolation not possible.");
			return from;
		}
		float[] result = new float[from.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = Interpolate(from[i], to[i], amount);
		}
		return result;
	}

	public static Vector3 Lerp(Vector3 from, Vector3 to, Vector3 weights) {
		return new Vector3(
			Mathf.Lerp(from.x, to.x, weights.x),
			Mathf.Lerp(from.y, to.y, weights.y),
			Mathf.Lerp(from.z, to.z, weights.z)
		);
	}

	public static Bounds GetBounds(GameObject instance) {
		Bounds bounds = new Bounds();
		foreach(MeshFilter filter in instance.GetComponentsInChildren<MeshFilter>()) {
			bounds.SetMinMax(Vector3.Min(bounds.min, filter.sharedMesh.bounds.min), Vector3.Max(bounds.max, filter.sharedMesh.bounds.max));
		}
		return bounds;
	}

	public static float GetSignedAngle(Vector3 A, Vector3 B, Vector3 axis) {
		return Mathf.Atan2(
			Vector3.Dot(axis, Vector3.Cross(A, B)),
			Vector3.Dot(A, B)
			) * Mathf.Rad2Deg;
	}

	public static Vector3 RotateAround(Vector3 vector, Vector3 pivot, Vector3 axis, float angle) {
		return Quaternion.AngleAxis(angle, axis) * (vector - pivot) + vector;
	}

	public static Vector3 ProjectCollision(Vector3 start, Vector3 end, LayerMask mask) {
		RaycastHit hit;
		if(Physics.Raycast(start, end-start, out hit, Vector3.Magnitude(end-start), mask)) {
			return hit.point;
		}
		return end;
	}

	public static Vector3 ProjectGround(Vector3 position, LayerMask mask) {
		position.y = GetHeight(position, mask);
		return position;
	}

	public static float GetHeight(Vector3 origin, LayerMask mask) {
		RaycastHit[] hits = Physics.RaycastAll(new Vector3(origin.x, 1000f, origin.z), Vector3.down, float.PositiveInfinity, mask);
		if(hits.Length == 0) {
			return origin.y;
		} else {
			bool updated = false;
			float height = float.MinValue;
			foreach(RaycastHit hit in hits) {
				if(!hit.collider.isTrigger) {
					height = Mathf.Max(hit.point.y, height);
					updated = true;
				}
			}
			return updated ? height : origin.y;
		}
	}

	public static float GetSlope(Vector3 origin, LayerMask mask) {
		RaycastHit[] upHits = Physics.RaycastAll(origin+Vector3.down, Vector3.up, float.PositiveInfinity, mask);
		RaycastHit[] downHits = Physics.RaycastAll(origin+Vector3.up, Vector3.down, float.PositiveInfinity, mask);
		if(upHits.Length == 0 && downHits.Length == 0) {
			return 0f;
		}
		Vector3 normal = Vector3.up;
		float height = float.MinValue;
		for(int i=0; i<downHits.Length; i++) {
			if(downHits[i].point.y > height && !downHits[i].collider.isTrigger) {
				height = downHits[i].point.y;
				normal = downHits[i].normal;
			}
		}
		for(int i=0; i<upHits.Length; i++) {
			if(upHits[i].point.y > height && !upHits[i].collider.isTrigger) {
				height = upHits[i].point.y;
				normal = upHits[i].normal;
			}
		}
		return Vector3.Angle(normal, Vector3.up) / 90f;
	}

	public static Vector3 GetNormal(Vector3 position, Vector3 point, Collider collider, float radius, LayerMask mask) {
		if(position == point) {
			List<RaycastHit> hits = new List<RaycastHit>();
			Quaternion rotation = collider.transform.rotation;

			Vector3 x = rotation * Vector3.right;
			Vector3 y = rotation * Vector3.up;
			Vector3 z = rotation * Vector3.forward;

			RaycastHit XP;
			if(Physics.Raycast(point + radius * x, -x, out XP, 2f*radius, mask)) {
				hits.Add(XP);
			}
			RaycastHit XN;
			if(Physics.Raycast(point + radius * -x, x, out XN, 2f*radius, mask)) {
				hits.Add(XN);
			}
			RaycastHit YP;
			if(Physics.Raycast(point + radius * y, -y, out YP, 2f*radius, mask)) {
				hits.Add(YP);
			}
			RaycastHit YN;
			if(Physics.Raycast(point + radius * -y, y, out YN, 2f*radius, mask)) {
				hits.Add(YN);
			}
			RaycastHit ZP;
			if(Physics.Raycast(point + radius * z, -z, out ZP, 2f*radius, mask)) {
				hits.Add(ZP);
			}
			RaycastHit ZN;
			if(Physics.Raycast(point + radius * -z, z, out ZN, 2f*radius, mask)) {
				hits.Add(ZN);
			}
			
			if(hits.Count > 0) {
				RaycastHit closest = hits[0];
				for(int k=1; k<hits.Count; k++) {
					if(Vector3.Distance(hits[k].point, point) < Vector3.Distance(closest.point, point)) {
						closest = hits[k];
					}
				}
				return closest.normal;
			} else {
				Debug.Log("Could not compute normal for collider " + collider.name + ".");
				return Vector3.zero;
			}
		} else {
			RaycastHit hit;
			if(Physics.Raycast(position, (point - position).normalized, out hit, 2f*radius, mask)) {
				return hit.normal;
			} else {
				Debug.Log("Could not compute normal for collider " + collider.name + ".");
				return Vector3.zero;
			}
			
		}
	}

	public static Vector3 GetNormal(Vector3 origin, LayerMask mask) {
		RaycastHit[] upHits = Physics.RaycastAll(origin+Vector3.down, Vector3.up, float.PositiveInfinity, mask);
		RaycastHit[] downHits = Physics.RaycastAll(origin+Vector3.up, Vector3.down, float.PositiveInfinity, mask);
		if(upHits.Length == 0 && downHits.Length == 0) {
			return Vector3.up;
		}
		Vector3 normal = Vector3.up;
		float height = float.MinValue;
		for(int i=0; i<downHits.Length; i++) {
			if(downHits[i].point.y > height && !downHits[i].collider.isTrigger) {
				height = downHits[i].point.y;
				normal = downHits[i].normal;
			}
		}
		for(int i=0; i<upHits.Length; i++) {
			if(upHits[i].point.y > height && !upHits[i].collider.isTrigger) {
				height = upHits[i].point.y;
				normal = upHits[i].normal;
			}
		}
		return normal;
	}

	public static void Destroy(UnityEngine.Object o, bool allowDestroyingAssets = true) {
		if(Application.isPlaying) {
			GameObject.Destroy(o);
		} else {
			GameObject.DestroyImmediate(o, allowDestroyingAssets);
		}
	}

	public static void SetGUIColor(Color color) {
		GUI.backgroundColor = color;
	}

	public static void ResetGUIColor() {
		SetGUIColor(Color.white);
	}
	
	public static GUIStyle GetFontColor(Color color) {
		GUIStyle style = new GUIStyle();
		style.normal.textColor = color;
		style.hover.textColor = color;
		return style;
	}

	public static bool GUIButton(string label, Color backgroundColor, Color textColor) {
		GUIStyle style = new GUIStyle("Button");
		style.normal.textColor = textColor;
		style.alignment = TextAnchor.MiddleCenter;
		SetGUIColor(backgroundColor);
		bool clicked = GUILayout.Button(label, style);
		ResetGUIColor();
		return clicked;
	}

	public static bool GUIButton(string label, Color backgroundColor, Color textColor, float width, float height) {
		GUIStyle style = new GUIStyle("Button");
		style.normal.textColor = textColor;
		style.alignment = TextAnchor.MiddleCenter;
		SetGUIColor(backgroundColor);
		bool clicked = GUILayout.Button(label, style, GUILayout.Width(width), GUILayout.Height(height));
		ResetGUIColor();
		return clicked;
	}

	public static bool GUIButton(string label, Color backgroundColor, Color textColor, float width) {
		GUIStyle style = new GUIStyle("Button");
		style.normal.textColor = textColor;
		style.alignment = TextAnchor.MiddleCenter;
		SetGUIColor(backgroundColor);
		bool clicked = GUILayout.Button(label, style, GUILayout.Width(width));
		ResetGUIColor();
		return clicked;
	}

	public static Rect GetGUIRect(float x, float y, float width, float height) {
		return new Rect(x*Screen.width, y*Screen.height, width*Screen.width, height*Screen.height);
	}

	/*
	public static void Normalise(ref float[] values) {
		float sum = 0f;
		for(int i=0; i<values.Length; i++) {
			sum += Mathf.Abs(values[i]);
		}
		if(sum != 0f) {
			for(int i=0; i<values.Length; i++) {
				values[i] = Mathf.Abs(values[i]) / sum;
			}
		}
	}
	*/

	public static float[] Normalise(float[] values) {
		float min = values.Min();
		float max = values.Max();
		float[] result = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			result[i] = Utility.Normalise(values[i], min, max, 0f, 1f);
		}
		return result;
	}

	public static void SoftMax(ref float[] values) {
		float frac = 0f;
		for(int i=0; i<values.Length; i++) {
			values[i] = Mathf.Exp(values[i]);
			frac += values[i];
		}
		for(int i=0; i<values.Length; i++) {
			values[i] /= frac;
		}
	}

	public static float GaussianValue(float mean, float sigma) {
		if(mean == 0f && sigma == 0f) {
			return 0f;
		}
		// The method requires sampling from a uniform random of (0,1]
		// but Random.NextDouble() returns a sample of [0,1).
		double x1 = 1f - UnityEngine.Random.value;
		double x2 = 1f - UnityEngine.Random.value;
		double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
		return (float)(y1 * sigma + mean);
	}

	public static Vector3 GaussianVector3(float mean, float sigma) {
		return new Vector3(GaussianValue(mean, sigma), GaussianValue(mean, sigma), GaussianValue(mean, sigma));
	}

	public static Vector3 GaussianVector3(Vector3 mean, Vector3 sigma) {
		return new Vector3(GaussianValue(mean.x, sigma.x), GaussianValue(mean.y, sigma.y), GaussianValue(mean.z, sigma.z));
	}

	public static Vector3 UniformVector3(Vector3 min, Vector3 max) {
		return new Vector3(UnityEngine.Random.Range(min.x, max.x), UnityEngine.Random.Range(min.y, max.y), UnityEngine.Random.Range(min.z, max.z));
	}

	/*
	public static float[] AngleToArray(float angle) {
		angle = Mathf.Repeat(angle, 360f);
		angle = angle / 360f;
		float[] array = new float[4];
		if(angle < 0.25f) {
			float ratio = angle / 0.25f;
			array[0] = 1f - ratio;
			array[1] = ratio;
			array[2] = 0f;
			array[3] = 0f;
			return array;
		}
		if(angle < 0.5f) {
			float ratio = (angle - 0.25f) / 0.25f;
			array[0] = 0f;
			array[1] = 1f - ratio;
			array[2] = ratio;
			array[3] = 0f;
			return array;
		}
		if(angle < 0.75f) {
			float ratio = (angle - 0.5f) / 0.25f;
			array[0] = 0f;
			array[1] = 0f;
			array[2] = 1f - ratio;
			array[3] = ratio;
			return array;
		}
		if(angle < 1f) {
			float ratio = (angle - 0.75f) / 0.25f;
			array[0] = ratio;
			array[1] = 0f;
			array[2] = 0f;
			array[3] = 1f - ratio;
			return array;
		}
		return array;
	}

	public static Vector2 AngleToVector(float angle) {
		angle = Mathf.Repeat(angle, 360f);
		angle = angle / 360f * 2f * Mathf.PI;
		return new Vector2(Mathf.Sin(angle), Mathf.Cos(angle));
	}
	*/

	public static float PhaseUpdate(float[] values) {
		float delta = 0f;
		for(int i=0; i<values.Length-1; i++) {
			delta += Utility.PhaseUpdate(values[i], values[i+1]);
		}
		return delta;
	}

	/*
	public static float PhaseUpdate(Vector2 from, Vector2 to, bool forcePositive=false) {
		float delta = -Vector2.SignedAngle(from, to) / 360f;
		return forcePositive ? Mathf.Abs(delta) : delta;
	}
	*/

	// public static float PhaseUpdate(Vector2 from, Vector2 to) {
	// 	if(from.magnitude == 0f || to.magnitude == 0f) {
	// 		Debug.LogWarning("Magnitude for computing phase update was zero.");
	// 	}
	// 	//return Vector2.Angle(PhaseVector(from), PhaseVector(to)) / 360f;
	// 	return -Vector2.SignedAngle(from, to) / 360f;
	// 	//return Mathf.Repeat((to - from + 1f), 1f);
	// }

	public static float PhaseUpdate(float from, float to) {
		return -Vector2.SignedAngle(PhaseVector(from), PhaseVector(to)) / 360f;
	}

	public static Vector2 PhaseVector(float phase, float magnitude=1f) {
		phase *= 2f*Mathf.PI;
		return magnitude * new Vector2(Mathf.Sin(phase), Mathf.Cos(phase));
	}

	public static float PhaseValue(Vector2 phase) {
		if(phase.magnitude == 0f) {
			// Debug.LogWarning("Magnitude for computing phase value was zero. Returning zero.");
			return 0f;
		}
		float angle = -Vector2.SignedAngle(Vector2.up, phase.normalized);
		if(angle < 0f) {
			angle = 360f + angle;
		}
		return Mathf.Repeat(angle / 360f, 1f);
	}

	public static float FilterPhaseLinear(float[] values) {
		float[] x = new float[values.Length];
		float[] y = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			Vector2 v = PhaseVector(values[i]);
			x[i] = v.x;
			y[i] = v.y;
		}
		return Mathf.Repeat(-Vector2.SignedAngle(Vector2.up, new Vector2(x.Mean(), y.Mean()).normalized) / 360f, 1f);
	}

	public static double FilterPhaseLinear(double[] values) {
		float[] x = new float[values.Length];
		float[] y = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			Vector2 v = PhaseVector((float)values[i]);
			x[i] = v.x;
			y[i] = v.y;
		}
		return Mathf.Repeat(-Vector2.SignedAngle(Vector2.up, new Vector2(x.Mean(), y.Mean()).normalized) / 360f, 1f);
	}

	public static float FilterPhaseGaussian(float[] values) {
		float[] x = new float[values.Length];
		float[] y = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			Vector2 v = PhaseVector(values[i]);
			x[i] = v.x;
			y[i] = v.y;
		}
		return Mathf.Repeat(-Vector2.SignedAngle(Vector2.up, new Vector2(FilterGaussian(x), FilterGaussian(y)).normalized) / 360f, 1f);
	}

	public static double FilterPhaseGaussian(double[] values) {
		float[] x = new float[values.Length];
		float[] y = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			Vector2 v = PhaseVector((float)values[i]);
			x[i] = v.x;
			y[i] = v.y;
		}
		return Mathf.Repeat(-Vector2.SignedAngle(Vector2.up, new Vector2(FilterGaussian(x), FilterGaussian(y)).normalized) / 360f, 1f);
	}

	public static Collider GetSelectedCollider() {
		Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
		RaycastHit hit;
		if(Physics.Raycast(ray, out hit)) {
			return hit.collider;
		} else {
			return null;
		}
	}

	public static double FilterGaussian(double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		if(values.Length == 1) {
			return values[0];
		}
		double window = ((double)values.Length - 1.0) / 2.0;
		double sum = 0.0;
		double value = 0.0;
		for(int i=0; i<values.Length; i++) {
			double weight = System.Math.Exp(-System.Math.Pow((double)i - window, 2.0) / System.Math.Pow(0.5 * window, 2.0));
			value += weight * values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static float FilterGaussian(float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float window = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		float value = 0f;
		for(int i=0; i<values.Length; i++) {
			float weight = Mathf.Exp(-Mathf.Pow((float)i - window, 2f) / Mathf.Pow(0.5f * window, 2f));
			value += weight * values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static float FilterGaussian(float[] values, bool[] mask) {
		if(values.Length == 0) {
			return 0f;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float window = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		float value = 0f;
		for(int i=0; i<values.Length; i++) {
			if(mask[i]) {
				float weight = Mathf.Exp(-Mathf.Pow((float)i - window, 2f) / Mathf.Pow(0.5f * window, 2f));
				value += weight * values[i];
				sum += weight;
			}
		}
		return value / sum;
	}

	public static Vector3 FilterGaussian(Vector3[] values) {
		if(values.Length == 0) {
			return Vector3.zero;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float window = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		Vector3 value = Vector3.zero;
		for(int i=0; i<values.Length; i++) {
			float weight = Mathf.Exp(-Mathf.Pow((float)i - window, 2f) / Mathf.Pow(0.5f * window, 2f));
			value += weight * values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static Vector3 FilterGaussian(Vector3[] values, bool[] mask) {
		if(values.Length == 0) {
			return Vector3.zero;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float window = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		Vector3 value = Vector3.zero;
		for(int i=0; i<values.Length; i++) {
			if(mask[i]) {
				float weight = Mathf.Exp(-Mathf.Pow((float)i - window, 2f) / Mathf.Pow(0.5f * window, 2f));
				value += weight * values[i];
				sum += weight;
			}
		}
		return value / sum;
	}

	public static Quaternion FilterGaussian(Quaternion[] values) {
		if(values.Length == 0) {
			return Quaternion.identity;
		}
		if(values.Length == 1) {
			return values[0];
		}
		Vector3[] forwards = new Vector3[values.Length];
		Vector3[] upwards = new Vector3[values.Length];
		for(int i=0; i<values.Length; i++) {
			forwards[i] = values[i].GetForward();
			upwards[i] = values[i].GetUp();
		}
		return Quaternion.LookRotation(FilterGaussian(forwards).normalized, FilterGaussian(upwards).normalized);
	}

	public static Quaternion FilterGaussian(Quaternion[] values, bool[] mask) {
		if(values.Length == 0) {
			return Quaternion.identity;
		}
		if(values.Length == 1) {
			return values[0];
		}
		Vector3[] forwards = new Vector3[values.Length];
		Vector3[] upwards = new Vector3[values.Length];
		for(int i=0; i<values.Length; i++) {
			forwards[i] = values[i].GetForward();
			upwards[i] = values[i].GetUp();
		}
		return Quaternion.LookRotation(FilterGaussian(forwards, mask).normalized, FilterGaussian(upwards, mask).normalized);
	}

	public static T GetMostCommonItem<T>(T[] items) {
		Dictionary<T,int> dic = new Dictionary<T, int>();
		int index = 0;
		T mostCommon = items[0];
		while(mostCommon == null && index<items.Length-1) {
			index += 1;
			mostCommon = items[index];
		}
		if(mostCommon != null) {
			dic.Add(mostCommon, 1);
			for(int i=index+1; i<items.Length; i++) {
				if(items[i] != null) {
					if(dic.ContainsKey(items[i])) {
						dic[items[i]] += 1;
						if(dic[items[i]] > dic[mostCommon]) {
							mostCommon = items[i];
						}
					} else {
						dic.Add(items[i], 1);
					}
				}
			}
		}
		return mostCommon;
	}

	public static Vector3 GetMostCenteredVector(Vector3[] vectors, bool[] mask) {
		if(mask.Length == 0) {
			Debug.Log("No items were given.");
			return Vector3.zero;
		}
		if(mask.Length % 2 != 1) {
			float pivot = (float)mask.Length / 2f;
			return 0.5f * (vectors[Mathf.FloorToInt(pivot)] + vectors[Mathf.CeilToInt(pivot)]);
		} else {
			int width = (mask.Length-1) / 2;
			int center = width;
			if(mask[center]) {
				return vectors[center];
			} else {
				int step = 0;
				while(step < width) {
					step += 1;
					if(mask[center-step] && mask[center+step]) {
						return 0.5f * (vectors[center-step] + vectors[center+step]);
					}
					if(mask[center-step]) {
						return vectors[center-step];
					}
					if(mask[center+step]) {
						return vectors[center+step];
					}
				}
			}
		}
		Debug.Log("No item could be found.");
		return Vector3.zero;
	}

	public static float Repeat(float value, float min, float max) {
		float n = Normalise(value, min, max, 0f, 1f);
		float v = Mathf.Repeat(n, 1f);
		return Normalise(v, 0f, 1f, min, max);
	}

	public static GameObject[] Unroll(GameObject[] instances) {
		List<GameObject> candidates = new List<GameObject>();
		for(int k=0; k<instances.Length; k++) {
			Action<GameObject> recursion = null;
			recursion = new Action<GameObject>(instance => {
				candidates.Add(instance);
				for(int i=0; i<instance.transform.childCount; i++) {
					recursion(instance.transform.GetChild(i).gameObject);
				}
			});
			recursion(instances[k]);
		}
		return candidates.ToArray();
	}

	public static Type StringToType(string value) {
		return Type.GetType(value);
	}

}