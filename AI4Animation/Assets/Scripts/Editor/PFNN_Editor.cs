using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(PFNN))]
public class PFNN_Editor : Editor {

		public PFNN Target;

		void Awake() {
			Target = (PFNN)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			DrawDefaultInspector();
			if(GUILayout.Button("Load Parameters")) {
				Target.LoadParameters();
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
}