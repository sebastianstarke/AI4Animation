using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(B_PFNN))]
public class PFNN_Editor : Editor {

		public B_PFNN Target;

		void Awake() {
			Target = (B_PFNN)target;
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