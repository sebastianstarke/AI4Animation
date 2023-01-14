using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class FunctionDemo : MonoBehaviour {

	public float NoiseMin = 0.25f;
	public float NoiseMax = 0.75f;

	public float YMin = 0f;
	public float YMax = 1f;

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		UltiDraw.Begin();

		float[] values = new float[1000];
		for(int i=0; i<values.Length; i++) {
			values[i] = Random.Range(NoiseMin, NoiseMax);
		}
		UltiDraw.PlotFunction(new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), values, yMin: YMin, yMax: YMax, backgroundColor: UltiDraw.DarkGrey, lineColor: UltiDraw.Cyan);

		UltiDraw.End();
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(FunctionDemo))]
	public class FunctionDemo_Editor : Editor {

		public FunctionDemo Target;

		void Awake() {
			Target = (FunctionDemo)target;
		}

		public override void OnInspectorGUI() {
			EditorGUILayout.HelpBox("Run Play Mode", MessageType.Info);
			DrawDefaultInspector();
		}

	}
	#endif

}
