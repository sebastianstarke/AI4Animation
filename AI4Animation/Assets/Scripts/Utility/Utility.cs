using UnityEngine;
using System.Collections.Generic;

public static class Utility {

	private static Dictionary<PrimitiveType, Mesh> PrimitiveMeshes = new Dictionary<PrimitiveType, Mesh>();

	public static void SetFPS(int fps) {
		#if UNITY_EDITOR
		QualitySettings.vSyncCount = 0;
		#else
		QualitySettings.vSyncCount = 1;
		#endif
		Application.targetFrameRate = fps;
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

	public static Vector3 ProjectCollision(Vector3 start, Vector3 end, LayerMask mask) {
		RaycastHit hit;
		if(Physics.Raycast(start, end-start, out hit, Vector3.Magnitude(end-start), mask)) {
			return hit.point;
		}
		return end;
	}

	public static float GetHeight(float x, float z, LayerMask mask) {
		RaycastHit hit;
		bool intersection = Physics.Raycast(new Vector3(x,1f,z), Vector3.up, out hit, mask);
		if(!intersection) {
			intersection = Physics.Raycast(new Vector3(x,1f,z), Vector3.down, out hit, mask);
		}
		if(intersection) {
			return hit.point.y;
		} else {
			return 0f;
		}
	}

	public static float GetRise(float x, float z, LayerMask mask) {
		RaycastHit hit;
		bool intersection = Physics.Raycast(new Vector3(x,1f,z), Vector3.up, out hit, mask);
		if(!intersection) {
			intersection = Physics.Raycast(new Vector3(x,1f,z), Vector3.down, out hit, mask);
		}
		if(intersection) {
			return Vector3.Angle(hit.normal, Vector3.up) / 90f;
		} else {
			return 0f;
		}
	}
 
	public static GameObject CreatePrimitive(PrimitiveType type, bool withCollider) {
		if (withCollider) { return GameObject.CreatePrimitive(type); }

		GameObject gameObject = new GameObject(type.ToString());
		MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();
		meshFilter.sharedMesh = GetPrimitiveMesh(type);
		gameObject.AddComponent<MeshRenderer>();

		return gameObject;
	}

	public static Mesh GetPrimitiveMesh(PrimitiveType type) {
		if (!PrimitiveMeshes.ContainsKey(type)) {
			CreatePrimitiveMesh(type);
		}

		return PrimitiveMeshes[type];
	}

	public static Mesh CreatePrimitiveMesh(PrimitiveType type) {
		GameObject gameObject = GameObject.CreatePrimitive(type);
		Mesh mesh = gameObject.GetComponent<MeshFilter>().sharedMesh;
		Destroy(gameObject);

		PrimitiveMeshes[type] = mesh;
		return mesh;
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
	
	public static bool GUIButton(string label, Color backgroundColor, Color textColor, TextAnchor alignment) {
		GUIStyle style = new GUIStyle("Button");
		style.normal.textColor = textColor;
		style.alignment = alignment;
		SetGUIColor(backgroundColor);
		bool clicked = GUILayout.Button(label, style);
		ResetGUIColor();
		return clicked;
	}

	public static bool GUIButton(string label, Color backgroundColor, Color textColor, TextAnchor alignment, float width, float height) {
		GUIStyle style = new GUIStyle("Button");
		style.normal.textColor = textColor;
		style.alignment = alignment;
		SetGUIColor(backgroundColor);
		bool clicked = GUILayout.Button(label, style, GUILayout.Width(width), GUILayout.Height(height));
		ResetGUIColor();
		return clicked;
	}

}