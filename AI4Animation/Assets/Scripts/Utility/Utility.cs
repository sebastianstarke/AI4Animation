using UnityEngine;
using System.Collections.Generic;

public static class Utility {

	public static Color White = Color.white;
	public static Color Black = Color.black;
	public static Color Red = Color.red;
	public static Color DarkRed = new Color(0.75f, 0f, 0f, 1f);
	public static Color Green = Color.green;
	public static Color DarkGreen = new Color(0f, 0.75f, 0f, 1f);
	public static Color Blue = Color.blue;
	public static Color Cyan = Color.cyan;
	public static Color Magenta = Color.magenta;
	public static Color Yellow = Color.yellow;
	public static Color Grey = Color.grey;
	public static Color LightGrey = new Color(0.75f, 0.75f, 0.75f, 1f);
	public static Color DarkGrey = new Color(0.25f, 0.25f, 0.25f, 1f);
	public static Color Orange = new Color(1f, 0.5f, 0f, 1f);
	public static Color Brown = new Color(0.5f, 0.25f, 0f, 1f);
	public static Color Mustard = new Color(1f, 0.75f, 0.25f, 1f);
	public static Color Teal = new Color(0f, 0.75f, 0.75f, 1f);

	private static Dictionary<PrimitiveType, Mesh> primitiveMeshes = new Dictionary<PrimitiveType, Mesh>();

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

	public static Mesh GetPrimitiveMesh(PrimitiveType type) {
		if(!primitiveMeshes.ContainsKey(type)) {
			CreatePrimitiveMesh(type);
		}

		return primitiveMeshes[type];
	}

	private static Mesh CreatePrimitiveMesh(PrimitiveType type) {
		GameObject gameObject = GameObject.CreatePrimitive(type);
		gameObject.GetComponent<MeshRenderer>().enabled = false;
		Mesh mesh = gameObject.GetComponent<MeshFilter>().sharedMesh;
		Destroy(gameObject);

		primitiveMeshes[type] = mesh;
		return mesh;
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

	//0 = Amplitude, 1 = Frequency, 2 = Shift, 3 = Offset, 4 = Slope, 5 = time
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

	public static float Gaussian(float peak, float shift, float sigma, float x) {
		float upper = (x-shift)*(x-shift);
		float lower = 2f*sigma*sigma;
		return peak * Mathf.Exp(-upper/lower);
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
		position.y = GetHeight(position,mask);
		return position;
	}

	public static float GetHeight(Vector3 origin, LayerMask mask) {
		RaycastHit[] upHits = Physics.RaycastAll(origin+Vector3.down, Vector3.up, mask);
		RaycastHit[] downHits = Physics.RaycastAll(origin+Vector3.up, Vector3.down, mask);
		if(upHits.Length == 0 && downHits.Length == 0) {
			return origin.y;
		}
		float height = float.MinValue;
		for(int i=0; i<downHits.Length; i++) {
			if(downHits[i].point.y > height) {
				height = downHits[i].point.y;
			}
		}
		for(int i=0; i<upHits.Length; i++) {
			if(upHits[i].point.y > height) {
				height = upHits[i].point.y;
			}
		}
		return height;
	}

	public static float GetRise(Vector3 origin, LayerMask mask) {
		RaycastHit[] upHits = Physics.RaycastAll(origin+Vector3.down, Vector3.up, mask);
		RaycastHit[] downHits = Physics.RaycastAll(origin+Vector3.up, Vector3.down, mask);
		if(upHits.Length == 0 && downHits.Length == 0) {
			return 0f;
		}
		Vector3 normal = Vector3.up;
		float height = float.MinValue;
		for(int i=0; i<downHits.Length; i++) {
			if(downHits[i].point.y > height) {
				height = downHits[i].point.y;
				normal = downHits[i].normal;
			}
		}
		for(int i=0; i<upHits.Length; i++) {
			if(upHits[i].point.y > height) {
				height = upHits[i].point.y;
				normal = upHits[i].normal;
			}
		}
		return Vector3.Angle(normal, Vector3.up) / 90f;
	}

	public static Color[] GetRainbowColors(int number) {
		Color[] colors = new Color[number];
		for(int i=0; i<number; i++) {
			float frequency = 5f/number;
			colors[i].r = Utility.Normalise(Mathf.Sin(frequency*i + 0f) * (127f) + 128f, 0f, 255f, 0f, 1f);
			colors[i].g = Utility.Normalise(Mathf.Sin(frequency*i + 2f) * (127f) + 128f, 0f, 255f, 0f, 1f);
			colors[i].b = Utility.Normalise(Mathf.Sin(frequency*i + 4f) * (127f) + 128f, 0f, 255f, 0f, 1f);
			colors[i].a = 1f;
		}
		return colors;
	}

	public static void Destroy(Object o) {
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
}

namespace Quaternions {
	using System; 
	using MathNet.Numerics;
	using MathNet.Numerics.LinearAlgebra.Double;

	/// <summary>Quaternion Number</summary>
	/// <remarks>
	/// http://en.wikipedia.org/wiki/Quaternion
	/// http://mathworld.wolfram.com/Quaternion.html
	/// http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf
	/// http://www.lce.hut.fi/~ssarkka/pub/quat.pdf
	/// </remarks>
	public struct Quaternion : IEquatable<Quaternion>, IFormattable
	{
		readonly double _w; // real part 
		readonly double _x, _y, _z; // imaginary part   
		/// <summary>
		/// Initializes a new instance of the Quatpow
		/// 
		/// ernion.
		/// </summary>
		public Quaternion(double real, double imagX, double imagY, double imagZ)
		{
			_x = imagX;
			_y = imagY;
			_z = imagZ;
			_w = real;
		}
		/// <summary>
		/// Given a Vector (w,x,y,z), transforms it into a Quaternion = w+xi+yj+zk
		/// </summary>
		/// <param name="v">The vector to transform into a Quaternion</param>
		public Quaternion(DenseVector v) : this(v[0], v[1], v[2], v[3])
		{ }
		/// <summary>
		/// Neutral element for multiplication
		/// </summary>
		public static readonly Quaternion One = new Quaternion(1, 0, 0, 0);
		/// <summary>
		/// Neutral element for sum
		/// </summary>
		public static readonly Quaternion Zero = new Quaternion(0, 0, 0, 0);
		/// <summary>
		/// Calculates norm of quaternion from it's algebraical notation
		/// </summary> 
		private static double ToNormSquared(double real, double imagX, double imagY, double imagZ)
		{
			return (imagX*imagX) + (imagY*imagY) + (imagZ*imagZ) + (real*real);
		}
		/// <summary>
		/// Creates unit quaternion (it's norm == 1) from it's algebraical notation
		/// </summary> 
		private static Quaternion ToUnitQuaternion(double real, double imagX, double imagY, double imagZ)
		{
			double norm = Math.Sqrt(ToNormSquared(real, imagX, imagY, imagZ));
			return new Quaternion(real / norm, imagX / norm, imagY / norm, imagZ / norm);
		}

		/// <summary>
		/// Gets the real part of the quaternion.
		/// </summary>
		public double Real
		{
			get { return _w; }
		}

		/// <summary>
		/// Gets the imaginary X part (coefficient of complex I) of the quaternion.
		/// </summary>
		public double ImagX
		{
			get { return _x; }
		}

		/// <summary>
		/// Gets the imaginary Y part (coefficient of complex J) of the quaternion.
		/// </summary>
		public double ImagY
		{
			get { return _y; }
		}

		/// <summary>
		/// Gets the imaginary Z part (coefficient of complex K) of the quaternion.
		/// </summary>
		public double ImagZ
		{
			get { return _z; }
		}

		/// <summary>
		/// Gets the the sum of the squares of the four components.
		/// </summary>
		public double NormSquared
		{
			get { return ToNormSquared(Real, ImagX, ImagY, ImagZ); }
		}

		/// <summary>
		/// Gets the norm of the quaternion q: square root of the sum of the squares of the four components.
		/// </summary>
		public double Norm
		{
			get { return Math.Sqrt(NormSquared); } //TODO : robust Norm calculation
		}

		/// <summary>
		/// Gets the argument phi = arg(q) of the quaternion q, such that q = r*(cos(phi) +
		/// u*sin(phi)) = r*exp(phi*u) where r is the absolute and u the unit vector of
		/// q.
		/// </summary>
		public double Arg
		{
			get { return Math.Acos(Real / Norm); }
		}

		/// <summary>
		/// True if the quaternion q is of length |q| = 1.
		/// </summary>
		/// <remarks>
		/// To normalize a quaternion to a length of 1, use the <see cref="Normalized"/> method.
		/// All unit quaternions form a 3-sphere.
		/// </remarks>
		public bool IsUnitQuaternion
		{
			get { return NormSquared.AlmostEqual(1); }
		}

		/// <summary>
		/// Returns a new Quaternion q with the Scalar part only.
		/// If you need a Double, use the Real-Field instead.
		/// </summary>
		public Quaternion Scalar
		{
			get { return new Quaternion(_w, 0, 0, 0); }
		}

		/// <summary>
		/// Returns a new Quaternion q with the Vector part only.
		/// </summary>
		public Quaternion Vector
		{
			get { return new Quaternion(0, _x, _y, _z); }
		}

		/// <summary>
		/// Returns a new normalized Quaternion u with the Vector part only, such that ||u|| = 1.
		/// Q may then be represented as q = r*(cos(phi) + u*sin(phi)) = r*exp(phi*u) where r is the absolute and phi the argument of q.
		/// </summary>
		public Quaternion NormalizedVector
		{
			get { return ToUnitQuaternion(0, _x, _y, _z); }
		}

		/// <summary>
		/// Returns a new normalized Quaternion q with the direction of this quaternion.
		/// </summary>
		public Quaternion Normalized
		{
			get
			{
				return this == Zero ? this : ToUnitQuaternion(_w, _x, _y, _z);
			}
		}

		/// <summary>
		/// Roatates the provided rotation quaternion with this quaternion
		/// </summary>
		/// <param name="rotation">The rotation quaternion to rotate</param>
		/// <returns></returns>
		public Quaternion RotateRotationQuaternion(Quaternion rotation)
		{
			if (!rotation.IsUnitQuaternion) throw new ArgumentException("The quaternion provided is not a rotation", "rotation");
			return rotation*this;
		}

		/// <summary>
		/// Roatates the provided unit quaternion with this quaternion
		/// </summary>
		/// <param name="unitQuaternion">The unit quaternion to rotate</param>
		/// <returns></returns>
		public Quaternion RotateUnitQuaternion(Quaternion unitQuaternion)
		{
			if (!IsUnitQuaternion) throw new InvalidOperationException("You cannot rotate with this quaternion as it is not a Unit Quaternion");
			if (!unitQuaternion.IsUnitQuaternion) throw new ArgumentException("The quaternion provided is not a Unit Quaternion");

			return (this*unitQuaternion)*Conjugate();
		}

		/////// <summary>
		/////// Returns a new Quaternion q with the Sign of the components.
		/////// </summary>
		/////// <returns>
		/////// <list type="bullet">
		/////// <item>1 if Positive</item>
		/////// <item>0 if Neutral</item>
		/////// <item>-1 if Negative</item>
		/////// </list>
		/////// </returns>
		////public Quaternion ComponentSigns()
		////{
		////    return new Quaternion(
		////        Math.Sign(_x),
		////        Math.Sign(_y),
		////        Math.Sign(_z),
		////        Math.Sign(_w));
		////}

		/// <summary>
		/// (nop)
		/// </summary>
		public static Quaternion operator +(Quaternion q)
		{
			return q;
		}

		/// <summary>
		/// Negate a quaternion.
		/// </summary>
		public static Quaternion operator -(Quaternion q)
		{
			return q.Negate();
		}

		/// <summary>
		/// Add a quaternion to a quaternion.
		/// </summary>
		public static Quaternion operator +(Quaternion r, Quaternion q)
		{

			return new Quaternion(r._w + q._w, r._x + q._x, r._y + q._y, r._z + q._z);
		}

		/// <summary>
		/// Add a floating point number to a quaternion.
		/// </summary>
		public static Quaternion operator +(Quaternion q1, double d)
		{
			return new Quaternion(q1.Real + d, q1.ImagX, q1.ImagY, q1.ImagZ);
		}
		/// <summary>
		/// Add a quaternion to a floating point number.
		/// </summary>
		public static Quaternion operator +(double d, Quaternion q1)
		{
			return q1 + d;
		}
		/// <summary>
		/// Subtract a quaternion from a quaternion.
		/// </summary>
		public static Quaternion operator -(Quaternion q1, Quaternion q2)
		{
			return new Quaternion(q1._w - q2._w, q1._x - q2._x, q1._y - q2._y, q1._z - q2._z);
		}
		/// <summary>
		/// Subtract a floating point number from a quaternion.
		/// </summary>
		public static Quaternion operator -(double d, Quaternion q)
		{
			return new Quaternion(d - q.Real, q._x, q._y, q._z);
		}
		/// <summary>
		/// Subtract a floating point number from a quaternion.
		/// </summary>
		public static Quaternion operator -(Quaternion q, double d)
		{
			return new Quaternion(q.Real - d, q._x, q._y, q._z);
		}
		/// <summary>
		/// Multiply a quaternion with a quaternion.
		/// </summary>
		public static Quaternion operator *(Quaternion q1, Quaternion q2)
		{
			double ci = (q1._x*q2._w) + (q1._y*q2._z) - (q1._z*q2._y) + (q1._w*q2._x);
			double cj = (-q1._x*q2._z) + (q1._y*q2._w) + (q1._z*q2._x) + (q1._w*q2._y);
			double ck = (q1._x*q2._y) - (q1._y*q2._x) + (q1._z*q2._w) + (q1._w*q2._z);
			double cr = (-q1._x*q2._x) - (q1._y*q2._y) - (q1._z*q2._z) + (q1._w*q2._w);
			return new Quaternion(cr, ci, cj, ck);
		}
		/// <summary>
		/// Multiply a floating point number with a quaternion.
		/// </summary>
		public static Quaternion operator *(Quaternion q1, double d)
		{
			return new Quaternion(q1.Real*d, q1.ImagX*d, q1.ImagY*d, q1.ImagZ*d);
		}
		/// <summary>
		/// Multiply a floating point number with a quaternion.
		/// </summary>
		public static Quaternion operator *(double d, Quaternion q1)
		{
			return new Quaternion(q1.Real*d, q1.ImagX*d, q1.ImagY*d, q1.ImagZ*d);
		}

		/// <summary>
		/// Divide a quaternion by a quaternion.
		/// </summary> 
		public static Quaternion operator /(Quaternion q, Quaternion r)
		{

			if (r == Zero)
			{
				if (q == Zero)
					return new Quaternion(double.NaN, double.NaN, double.NaN, double.NaN);
				return new Quaternion(double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity);
			}
			double normSquared = r.NormSquared;
			var t0 = (r._w*q._w + r._x*q._x + r._y*q._y + r._z*q._z) / normSquared;
			var t1 = (r._w*q._x - r._x*q._w - r._y*q._z + r._z*q._y) / normSquared;
			var t2 = (r._w*q._y + r._x*q._z - r._y*q._w - r._z*q._x) / normSquared;
			var t3 = (r._w*q._z - r._x*q._y + r._y*q._x - r._z*q._w) / normSquared;
			return new Quaternion(t0, t1, t2, t3);
		}
		/// <summary>
		/// Divide a quaternion by a floating point number.
		/// </summary>
		public static Quaternion operator /(Quaternion q1, double d)
		{
			return new Quaternion(q1.Real / d, q1.ImagX / d, q1.ImagY / d, q1.ImagZ / d);
		}

		/// <summary>
		/// Raise a quaternion to a quaternion.
		/// </summary>
		public static Quaternion operator ^(Quaternion q1, Quaternion q2)
		{
			return q1.Pow(q2);
		}

		/// <summary>
		/// Raise a quaternion to a floating point number.
		/// </summary>
		public static Quaternion operator ^(Quaternion q1, double d)
		{
			return q1.Pow(d);
		}
		/// <summary>
		/// Equality operator for two quaternions
		/// </summary> 
		public static bool operator ==(Quaternion q1, Quaternion q2)
		{
			return q1.Equals(q2);
		}
		/// <summary>
		/// Equality operator for quaternion and double
		/// </summary> 
		public static bool operator ==(Quaternion q, double d)
		{
			return q.Real.AlmostEqual(d)
					&& q.ImagX.AlmostEqual(0)
					&& q.ImagY.AlmostEqual(0)
					&& q.ImagZ.AlmostEqual(0);
		}
		/// <summary>
		/// Equality operator for quaternion and double
		/// </summary> 
		public static bool operator ==(double d, Quaternion q)
		{
			return q.Real.AlmostEqual(d)
					&& q.ImagX.AlmostEqual(0)
					&& q.ImagY.AlmostEqual(0)
					&& q.ImagZ.AlmostEqual(0);
		}

		/// <summary>
		/// Inequality operator for two quaternions
		/// </summary> 
		public static bool operator !=(Quaternion q1, Quaternion q2)
		{
			return !(q1 == q2);
		}
		/// <summary>
		/// Inequality operator for quaternion and double
		/// </summary> 
		public static bool operator !=(Quaternion q1, double d)
		{
			return !(q1 == d);
		}
		/// <summary>
		/// Inequality operator for quaternion and double
		/// </summary> 
		public static bool operator !=(double d, Quaternion q1)
		{
			return !(q1 == d);
		}
		///// <summary>
		///// Convert a floating point number to a quaternion.
		///// </summary>
		//public static implicit operator Quaternion(double d)
		//{
		//    return new Quaternion(d, 0, 0, 0);
		//}
		/// <summary>
		/// Negate this quaternion.
		/// </summary>
		public Quaternion Negate()
		{
			return new Quaternion(-_w, -_x, -_y, -_z);
		}
		/// <summary>
		/// Inverts this quaternion. Inversing Zero returns Zero
		/// </summary>
		public Quaternion Inversed
		{
			get
			{
				if (this == Zero)
					return this;
				var normSquared = NormSquared;
				return new Quaternion(_w / normSquared, -_x / normSquared, -_y / normSquared, -_z / normSquared);
			}
		}

		/// <summary>
		/// Returns the distance |a-b| of two quaternions, forming a metric space.
		/// </summary>
		public static double Distance(Quaternion a, Quaternion b)
		{
			return (a - b).Norm;
		}

		/// <summary>
		/// Conjugate this quaternion.
		/// </summary>
		public Quaternion Conjugate()
		{
			return new Quaternion(_w, -_x, -_y, -_z);
		}

		/// <summary>
		/// Logarithm to a given base.
		/// </summary>
		public Quaternion Log(double lbase)
		{
			return Log() / Math.Log(lbase);
		}

		/// <summary>
		/// Natural Logarithm to base E.
		/// </summary>
		public Quaternion Log()
		{
			if (this == One)
				return One;
			var quat = NormalizedVector*Arg;
			return new Quaternion(Math.Log(Norm), quat.ImagX, quat.ImagY, quat.ImagZ);
		}

		/// <summary>
		/// Common Logarithm to base 10.
		/// </summary>
		public Quaternion Log10()
		{
			return Log() / Math.Log(10);
		}

		/// <summary>
		/// Exponential Function.
		/// </summary>
		/// <returns></returns>
		public Quaternion Exp()
		{
			double real = Math.Pow(Math.E, Real);
			var vector = Vector;
			double vectorNorm = vector.Norm;
			double cos = Math.Cos(vectorNorm);
			var sgn = vector == Zero ? Zero : vector / vectorNorm;
			double sin = Math.Sin(vectorNorm);
			return real*(cos + sgn*sin);
		}

		/// <summary>
		/// Raise the quaternion to a given power.
		/// </summary>
		/// <remarks>
		/// This algorithm is not very accurate and works only for normalized quaternions
		/// </remarks>
		public Quaternion Pow(double power)
		{
			if (this == Zero)
				return Zero;
			if (this == One)
				return One;
			return (power*Log()).Exp();
		}

		public Quaternion Pow(int power)
		{
			Quaternion quat = new Quaternion(this.Real, this.ImagX, this.ImagY, this.ImagZ);
			if (power == 0)
				return One;
			if (power == 1)
				return this;
			if (this == Zero || this == One)
				return this;
			return quat*quat.Pow(power - 1);
		}
		/// <summary>
		/// Returns cos(n*arccos(x)) = 2*Cos((n-1)arccos(x))cos(arccos(x)) - cos((n-2)*arccos(x))
		/// </summary> 
		public static double ChybyshevCosPoli(int n, double x)
		{
			if (n == 0)
				return 1.0;
			if (n == 1)
				return x;
			return 2*ChybyshevCosPoli(n - 1, x)*x - ChybyshevCosPoli(n - 2, x);
		}
		/// <summary>
		/// Returns sin(n*x)
		/// </summary> 
		public static double ChybyshevSinPoli(int n, double x)
		{
			if (n == 0)
				return 1;
			if (n == 1)
				return 2*x;
			return 2*x*ChybyshevSinPoli(n - 1, x) - ChybyshevSinPoli(n - 2, x);
		}
		/// <summary>
		/// Raise the quaternion to a given power.
		/// </summary>
		public Quaternion Pow(Quaternion power)
		{
			if (this == Zero)
				return Zero;
			if (this == One)
				return One;
			return (power*Log()).Exp();
		}

		/// <summary>
		/// Square root of the Quaternion: q^(1/2).
		/// </summary>
		public Quaternion Sqrt()
		{
			double arg = Arg*0.5;
			return NormalizedVector*((Math.Sin(arg)) + (Math.Cos(arg))*(Math.Sqrt(_w)));
		}

		public bool IsNan
		{
			get
			{
				return
					double.IsNaN(Real) ||
					double.IsNaN(ImagX) ||
					double.IsNaN(ImagY) ||
					double.IsNaN(ImagZ);

			}
		}

		public bool IsInfinity
		{
			get
			{
				return
					double.IsInfinity(Real) ||
					double.IsInfinity(ImagX) ||
					double.IsInfinity(ImagY) ||
					double.IsInfinity(ImagZ);
			}
		}
		/// <summary>
		/// returns quaternion as real+ImagXi+ImagYj+ImagZk based on format provided 
		/// </summary> 
		public string ToString(string format, IFormatProvider formatProvider)
		{
			return string.Format(formatProvider, "{0}{1}{2}i{3}{4}j{5}{6}k",
				Real.ToString(format, formatProvider),
				(ImagX < 0) ? "" : "+",
				ImagX.ToString(format, formatProvider),
				(ImagY < 0) ? "" : "+",
				ImagY.ToString(format, formatProvider),
				(ImagZ < 0) ? "" : "+",
				ImagZ.ToString(format, formatProvider));
		}
		/// <summary>
		/// returns quaternion as real+ImagXi+ImagYj+ImagZk 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format("{0}{1}{2}i{3}{4}j{5}{6}k",
				Real,
				(ImagX < 0) ? "" : "+",
				ImagX,
				(ImagY < 0) ? "" : "+",
				ImagY,
				(ImagZ < 0) ? "" : "+",
				ImagZ);
		}
		/// <summary>
		/// Equality for quaternions
		/// </summary> 
		public bool Equals(Quaternion other)
		{
			if (other.IsNan && IsNan || other.IsInfinity && IsInfinity)
				return true;
			return Real.AlmostEqual(other.Real)
				&& ImagX.AlmostEqual(other.ImagX)
				&& ImagY.AlmostEqual(other.ImagY)
				&& ImagZ.AlmostEqual(other.ImagZ);
		}
		/// <summary>
		/// Equality for quaternion
		/// </summary> 
		public override bool Equals(object obj)
		{
			if (ReferenceEquals(null, obj)) return false;
			return obj is Quaternion && Equals((Quaternion)obj);
		}
		/// <summary>
		/// Quaternion hashcode based on all members.
		/// </summary> 
		public override int GetHashCode()
		{
			unchecked
			{
				var hashCode = _w.GetHashCode();
				hashCode = (hashCode*397) ^ _x.GetHashCode();
				hashCode = (hashCode*397) ^ _y.GetHashCode();
				hashCode = (hashCode*397) ^ _z.GetHashCode();
				return hashCode;
			}
		}
	}
}