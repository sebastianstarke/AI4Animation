using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class GizmosGL : MonoBehaviour {

	private enum PROGRAM {NONE, LINES, TRIANGLES, TRIANGLE_STRIP, QUADS};
	private enum SPACE {NONE, SCENE, GUI}

	private static GizmosGL Instance;

	private static int CallIndex = 0;
	private static int VertexIndex = 0;
	private static GLCall[] Calls = new GLCall[32];
	private static Vector3[] Vertices = new Vector3[1024];

	private static PROGRAM Program = PROGRAM.NONE;
	private static SPACE Space = SPACE.NONE;

	private static Material SceneMaterial;

	private static bool Initialised;

	private static void Initialise() {
		if(Initialised) {
			return;
		}

		Resources.UnloadUnusedAssets();

		Shader colorShader = Shader.Find("Hidden/Internal-Colored");

		SceneMaterial = new Material(colorShader);
		SceneMaterial.hideFlags = HideFlags.HideAndDontSave;
		SceneMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
		SceneMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
		SceneMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
		SceneMaterial.SetInt("_ZWrite", 1);

		Initialised = true;
	}

	void Update() {
		DrawLine(Vector3.one, 2f*Vector3.one, Color.green);
		Debug.Log(CallIndex);
	}

	void OnRenderObject() {
		Initialise();

		int index = 0;

		SceneMaterial.SetPass(0);
		GL.Begin(GL.LINES);
		for(int i=0; i<CallIndex; i++) {
			GL.Color(Calls[i].Color);
			for(int j=0; j<Calls[i].VertexCount; j++) {
				GL.Vertex(Vertices[index]);
				index += 1;
			}
		}
		GL.End();
		
		Clear();

	}

	public static void Clear() {
		System.Array.Resize(ref Calls, Mathf.RoundToInt(Mathf.Pow(2f, Mathf.Max(5, Mathf.RoundToInt(Mathf.Log(CallIndex, 2f))))));
		System.Array.Resize(ref Vertices, Mathf.RoundToInt(Mathf.Pow(2f, Mathf.Max(10, Mathf.RoundToInt(Mathf.Log(VertexIndex, 2f))))));
		CallIndex = 0;
		VertexIndex = 0;
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	public static void DrawLine(Vector3 start, Vector3 end, Color color) {
		if(!DetectInstance()) {
			return;
		}
		AddCall(new GLCall(2, color, SPACE.SCENE));
		AddVertex(start);
		AddVertex(end);
	}

	private static void AddCall(GLCall call) {
		if(Calls.Length == CallIndex) {
			System.Array.Resize(ref Calls, 2*Calls.Length);
		}
		Calls[CallIndex] = call;
		CallIndex += 1;
	}

	private static void AddVertex(Vector3 vertex) {
		if(Vertices.Length == VertexIndex) {
			System.Array.Resize(ref Vertices, 2*Vertices.Length);
		}
		Vertices[VertexIndex] = vertex;
		VertexIndex += 1;
	}

	private static bool DetectInstance() {
		if(Instance != null) {
			return true;
		} else {
			Instance = GameObject.FindObjectOfType<GizmosGL>();
			if(Instance != null) {
				return true;
			} else {
				Debug.LogError("No 'GizmosGL' component available.");
				return false;
			}
		}
	}

	private struct GLCall {
		public int VertexCount;
		public Color Color;
		public SPACE Space;

		public GLCall(int vertexCount, Color color, SPACE space) {
			VertexCount = vertexCount;
			Color = color;
			Space = space;
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(GizmosGL))]
	public class GizmosGL_Editor : Editor {

		public GizmosGL Target;

		void Awake() {
			Target = (GizmosGL)target;
		}

		public override void OnInspectorGUI() {
			DrawDefaultInspector();
			EditorGUILayout.HelpBox("Call Buffer: " + Calls.Length, MessageType.None);
			EditorGUILayout.HelpBox("Vertex Buffer: " + Vertices.Length, MessageType.None);
		}

	}
	#endif

}
