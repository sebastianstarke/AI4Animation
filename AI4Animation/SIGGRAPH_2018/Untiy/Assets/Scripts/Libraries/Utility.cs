using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

public sealed class Gaussian {
    private bool _hasDeviate;
    private double _storedDeviate;
    private readonly System.Random _random;

    public Gaussian(System.Random random = null) {
        _random = random ?? new System.Random();
    }

    /// <summary>
    /// Obtains normally (Gaussian) distributed random numbers, using the Box-Muller
    /// transformation.  This transformation takes two uniformly distributed deviates
    /// within the unit circle, and transforms them into two independently
    /// distributed normal deviates.
    /// </summary>
    /// <param name="mu">The mean of the distribution.  Default is zero.</param>
    /// <param name="sigma">The standard deviation of the distribution.  Default is one.</param>
    /// <returns></returns>
    public double NextGaussian(double mu = 0, double sigma = 1) {
        if(sigma <= 0)
            throw new ArgumentOutOfRangeException("sigma", "Must be greater than zero.");

        if(_hasDeviate) {
            _hasDeviate = false;
            return _storedDeviate*sigma + mu;
        }

        double v1, v2, rSquared;
        do {
            // two random values between -1.0 and 1.0
            v1 = 2*_random.NextDouble() - 1;
            v2 = 2*_random.NextDouble() - 1;
            rSquared = v1*v1 + v2*v2;
            // ensure within the unit circle
        } while (rSquared >= 1 || rSquared == 0);

        // calculate polar tranformation for each deviate
        var polar = Math.Sqrt(-2*Math.Log(rSquared)/rSquared);
        // store first deviate
        _storedDeviate = v2*polar;
        _hasDeviate = true;
        // return second deviate
        return v1*polar*sigma + mu;
    }
}

public static class Utility {

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
		QualitySettings.vSyncCount = 1;
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
		if(valueMax-valueMin != 0) {
			return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin;
		} else {
			//Not possible to normalise input value.
			return value;
		}
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
	
	public static float Sigmoid(float x) {
		return 1f / (1f + Mathf.Exp(-x));
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
			float height = float.MinValue;
			foreach(RaycastHit hit in hits) {
				height = Mathf.Max(hit.point.y, height);
			}
			return height;
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

	public static Rect GetGUIRect(float x, float y, float width, float height) {
		return new Rect(x*Screen.width, y*Screen.height, width*Screen.width, height*Screen.height);
	}

	public static int ReadInt(string value) {
		value = FilterValueField(value);
		return ParseInt(value);
	}

	public static float ReadFloat(string value) {
		value = FilterValueField(value);
		return ParseFloat(value);
	}

	public static float[] ReadArray(string value) {
		value = FilterValueField(value);
		if(value.StartsWith(" ")) {
			value = value.Substring(1);
		}
		if(value.EndsWith(" ")) {
			value = value.Substring(0, value.Length-1);
		}
		string[] values = value.Split(' ');
		float[] array = new float[values.Length];
		for(int i=0; i<array.Length; i++) {
			array[i] = ParseFloat(values[i]);
		}
		return array;
	}

	public static string FilterValueField(string value) {
		while(value.Contains("  ")) {
			value = value.Replace("  "," ");
		}
		while(value.Contains("< ")) {
			value = value.Replace("< ","<");
		}
		while(value.Contains(" >")) {
			value = value.Replace(" >",">");
		}
		while(value.Contains(" .")) {
			value = value.Replace(" ."," 0.");
		}
		while(value.Contains(". ")) {
			value = value.Replace(". ",".0");
		}
		while(value.Contains("<.")) {
			value = value.Replace("<.","<0.");
		}
		return value;
	}

	public static int ParseInt(string value) {
		int parsed = 0;
		if(int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			//Debug.Log("Error parsing " + value + "!");
			return 0;
		}
	}

	public static float ParseFloat(string value) {
		float parsed = 0f;
		if(float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			//Debug.Log("Error parsing " + value + "!");
			return 0f;
		}
	}

	public static void Normalise(ref float[] values) {
		/*
		float min = values.Min();
		float max = values.Max();
		for(int i=0; i<values.Length; i++) {
			values[i] = Utility.Normalise(values[i], min, max, 0f, 1f);
		}
        float frac = 0.0f;
        for(int i=0; i<values.Length; i++) {
            frac += values[i];
        }
		if(frac != 0f) {
        	for(int i=0; i<values.Length; i++) {
            	values[i] /= frac;
        	}
		}
		*/
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

	public static Quaternion QuaternionAverage(Quaternion[] quaternions) {
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

	public static float GaussianValue(float mean, float sigma) {
		if(mean == 0f && sigma == 0f) {
			return 0f;
		}
		// The method requires sampling from a uniform random of (0,1]
		// but Random.NextDouble() returns a sample of [0,1).
		double x1 = 1 - GetRNG().NextDouble();
		double x2 = 1 - GetRNG().NextDouble();
		double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
		return (float)(y1 * sigma + mean);
	}

	public static Vector3 GaussianVector3(float mean, float sigma) {
		return new Vector3(GaussianValue(mean, sigma), GaussianValue(mean, sigma), GaussianValue(mean, sigma));
	}
	
	/*
	public static float GetLinearPhase(float value) {
		return value;
	}

	public static float GetLinearPhaseUpdate(float from, float to) {
		return Mathf.Repeat(((GetLinearPhase(to)-GetLinearPhase(from)) + 1f), 1f);
	}

	public static float GetWavePhase(float value) {
		return Mathf.Sin(value*2f*Mathf.PI);
	}

	public static float GetWavePhaseUpdate(float from, float to) {
		return GetWavePhase(to) - GetWavePhase(from);
	}

	public static Vector2 GetBarPhase(float value) {
		return new Vector2(2f * Mathf.Abs(0.5f - value), 1f - 2f * Mathf.Abs(0.5f - value));
	}

	public static Vector2 GetBarPhaseUpdate(float from, float to) {
		return GetBarPhase(to) - GetBarPhase(from);
	}

	public static Vector2 GetCirclePhase(float value) {
		value *= 2f*Mathf.PI;
		return new Vector2(Mathf.Sin(value), Mathf.Cos(value));
		//return Quaternion.AngleAxis(-value*360f, Vector3.forward) * Vector2.up;
	}

	public static Vector2 GetCirclePhaseUpdate(float from, float to) {
		return GetCirclePhase(to) - GetCirclePhase(from);
	}
	*/

	public static float PhaseUpdate(float from, float to) {
		return Mathf.Repeat((to - from + 1f), 1f);
	}

	public static Vector2 PhaseVector(float phase) {
		phase *= 2f*Mathf.PI;
		return new Vector2(Mathf.Sin(phase), Mathf.Cos(phase));
	}

	public static float PhaseAverage(float[] values) {
		float[] x = new float[values.Length];
		float[] y = new float[values.Length];
		for(int i=0; i<values.Length; i++) {
			Vector2 v = PhaseVector(values[i]);
			x[i] = v.x;
			y[i] = v.y;
		}
		return Mathf.Repeat(-Vector2.SignedAngle(Vector2.up, new Vector2(FilterGaussian(x), FilterGaussian(y)).normalized) / 360f, 1f);
	}

	public static float FilterGaussian(float[] values) {
		if(values.Length == 0) {
			return 0f;
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

}