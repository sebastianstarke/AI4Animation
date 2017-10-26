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

			Inspector();
			Target.Controller.Inspector();
			Target.Character.Inspector();
			Target.Trajectory.Inspector();
			Target.PFNN.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {
			Utility.SetGUIColor(Color.grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				if(GUILayout.Button("Animation")) {
					Target.Inspect = !Target.Inspect;
				}

				if(Target.Inspect) {
					Target.SetRoot((Transform)EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true));
					Target.SetJointCount(EditorGUILayout.IntField("Joint Count", Target.Joints.Length));
					for(int i=0; i<Target.Joints.Length; i++) {
						Target.SetJoint(i, (Transform)EditorGUILayout.ObjectField("Joint " + (i+1), Target.Joints[i], typeof(Transform), true));
					}
				}
			}
		}

}