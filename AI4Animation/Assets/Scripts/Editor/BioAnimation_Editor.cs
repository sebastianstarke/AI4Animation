using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(BioAnimation))]
public class BioAnimation_Editor : Editor {

		public BioAnimation Target;

		void Awake() {
			Target = (BioAnimation)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Target.Controller.Inspector();
			Target.Character.Inspector();
			Target.Trajectory.Inspector();
			Target.PFNN.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

}