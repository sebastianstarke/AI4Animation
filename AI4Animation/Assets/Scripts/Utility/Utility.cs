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

	public static Vector3 TransformPosition(Transform frame, Vector3 position) {
		return frame.position + frame.rotation * position;
	}

	public static Vector3 DeltaPosition(Transform frame, Vector3 position) {
		return Quaternion.Inverse(frame.rotation) * (position - frame.position);
	}

	public static Vector3 TransformDirection(Transform frame, Vector3 direction) {
		return frame.rotation * direction;
	}

	public static Vector3 DeltaDirection(Transform frame, Vector3 direction) {
		return Quaternion.Inverse(frame.rotation) * direction;
	}

	public static Quaternion TransformRotation(Transform frame, Quaternion rotation) {
		return frame.rotation * rotation;
	}

	public static Quaternion DeltaRotation(Transform frame, Quaternion rotation) {
		return Quaternion.Inverse(frame.rotation) * rotation;
	}

	public static Vector3 ExtractPosition(Matrix4x4 mat) {
		return mat.GetColumn(3);
	}

	public static Quaternion ExtractRotation(Matrix4x4 mat) {
		return Quaternion.LookRotation(mat.GetColumn(2), mat.GetColumn(1));
	}

	public static Vector3 ExtractScale(Matrix4x4 mat) {
		return new Vector3(mat.GetColumn(0).magnitude, mat.GetColumn(1).magnitude, mat.GetColumn(2).magnitude);
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

	public static Vector3 RelativePositionFrom(this Vector3 vector, Transformation from) {
		return from.Position + from.Rotation * vector;
	}

	public static Vector3 RelativeDirectionFrom(this Vector3 vector, Transformation from) {
		return from.Rotation * vector;
	}

	public static Quaternion RelativeRotationFrom(this Quaternion quaternion, Transformation from) {
		return from.Rotation * quaternion;
	}

	public static Vector3 RelativePositionTo(this Vector3 vector, Transformation to) {
		return Quaternion.Inverse(to.Rotation) * (vector - to.Position);
	}

	public static Vector3 RelativeDirectionTo(this Vector3 vector, Transformation to) {
		return Quaternion.Inverse(to.Rotation) * vector;
	}

	public static Quaternion RelativeRotationTo(this Quaternion quaternion, Transformation to) {
		return Quaternion.Inverse(to.Rotation) * quaternion;
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

}