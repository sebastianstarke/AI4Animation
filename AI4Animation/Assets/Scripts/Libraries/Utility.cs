using UnityEngine;
using System;
using System.IO;
using System.Reflection;

[System.Serializable]
public class GUIRect {
	[Range(0f, 1f)] public float X = 0.5f;
	[Range(0f, 1f)] public float Y = 0.5f;
	[Range(0f, 1f)] public float W = 0.5f;
	[Range(0f, 1f)] public float H = 0.5f;
}

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

	public static float GetSignedAngle(Vector3 A, Vector3 B, Vector3 axis) {
		return Mathf.Atan2(
			Vector3.Dot(axis, Vector3.Cross(A, B)),
			Vector3.Dot(A, B)
			) * Mathf.Rad2Deg;
	}

	public static Vector3 RotateAround(Vector3 vector, Vector3 pivot, Vector3 axis, float angle) {
		return Quaternion.AngleAxis(angle, axis) * (vector - pivot) + vector;
	}

	public static void OverridePosition(this Transform transform, Vector3 position) {
		Transform[] childs = new Transform[transform.childCount];
		for(int i=0; i<childs.Length; i++) {
			childs[i] = transform.GetChild(i);
		}
		transform.DetachChildren();
		transform.position = position;
		for(int i=0; i<childs.Length; i++) {
			childs[i].SetParent(transform);
		}
	}

	public static void OverrideRotation(this Transform transform, Quaternion rotation) {
		Transform[] childs = new Transform[transform.childCount];
		for(int i=0; i<childs.Length; i++) {
			childs[i] = transform.GetChild(i);
		}
		transform.DetachChildren();
		transform.rotation = rotation;
		for(int i=0; i<childs.Length; i++) {
			childs[i].SetParent(transform);
		}
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
		RaycastHit[] upHits = Physics.RaycastAll(origin+Vector3.down, Vector3.up, float.PositiveInfinity, mask);
		RaycastHit[] downHits = Physics.RaycastAll(origin+Vector3.up, Vector3.down, float.PositiveInfinity, mask);
		if(upHits.Length == 0 && downHits.Length == 0) {
			return origin.y;
		}
		float height = float.MinValue;
		for(int i=0; i<downHits.Length; i++) {
			if(downHits[i].point.y > height && !downHits[i].collider.isTrigger) {
				height = downHits[i].point.y;
			}
		}
		for(int i=0; i<upHits.Length; i++) {
			if(upHits[i].point.y > height && !upHits[i].collider.isTrigger) {
				height = upHits[i].point.y;
			}
		}
		return height;
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

	public static void Destroy(UnityEngine.Object o) {
		if(Application.isPlaying) {
			GameObject.Destroy(o);
		} else {
			GameObject.DestroyImmediate(o);
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
			Debug.Log("Error parsing " + value + "!");
			return 0;
		}
	}

	public static float ParseFloat(string value) {
		float parsed = 0f;
		if(float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + "!");
			return 0f;
		}
	}

	public static int ComputeMin(int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		int min = int.MaxValue;
		for(int i=0; i<values.Length; i++) {
			min = Mathf.Min(min, values[i]);
		}
		return min;
	}

	public static float ComputeMin(float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float min = float.MaxValue;
		for(int i=0; i<values.Length; i++) {
			min = Mathf.Min(min, values[i]);
		}
		return min;
	}

	public static double ComputeMin(double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double min = double.MaxValue;
		for(int i=0; i<values.Length; i++) {
			min = System.Math.Min(min, values[i]);
		}
		return min;
	}

	public static int ComputeMax(int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		int max = int.MinValue;
		for(int i=0; i<values.Length; i++) {
			max = Mathf.Max(max, values[i]);
		}
		return max;
	}

	public static float ComputeMax(float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float max = float.MinValue;
		for(int i=0; i<values.Length; i++) {
			max = Mathf.Max(max, values[i]);
		}
		return max;
	}

	public static double ComputeMax(double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double max = double.MinValue;
		for(int i=0; i<values.Length; i++) {
			max = System.Math.Max(max, values[i]);
		}
		return max;
	}

	public static float ComputeMean(int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		float mean = 0f;
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			mean += values[i];
			args += 1f;
		}
		mean /= args;
		return mean;
	}

	public static float ComputeMean(float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float mean = 0f;
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			mean += values[i];
			args += 1f;
		}
		mean /= args;
		return mean;
	}

	public static double ComputeMean(double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double mean = 0.0;
		double args = 0.0;
		for(int i=0; i<values.Length; i++) {
			mean += values[i];
			args += 1.0;
		}
		mean /= args;
		return mean;
	}

	public static float ComputeSigma(int[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float variance = 0f;
		float mean = ComputeMean(values);
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			variance += Mathf.Pow(values[i] - mean, 2f);
			args += 1f;
		}
		variance /= args;
		return Mathf.Sqrt(variance);
	}

	public static float ComputeSigma(float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float variance = 0f;
		float mean = ComputeMean(values);
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			variance += Mathf.Pow(values[i] - mean, 2f);
			args += 1f;
		}
		variance /= args;
		return Mathf.Sqrt(variance);
	}

	public static double ComputeSigma(double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double variance = 0.0;
		double mean = ComputeMean(values);
		double args = 1.0;
		for(int i=0; i<values.Length; i++) {
			variance += System.Math.Pow(values[i] - mean, 2.0);
			args += 1.0;
		}
		variance /= args;
		return System.Math.Sqrt(variance);
	}

	public static Quaternion AverageQuaternions(Quaternion[] quaternions) {
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
}