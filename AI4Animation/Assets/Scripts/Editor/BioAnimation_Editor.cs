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
			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				if(GUILayout.Button("Animation")) {
					Target.Inspect = !Target.Inspect;
				}

				if(Target.Inspect) {
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Target.TargetBlending = EditorGUILayout.Slider("Target Blending", Target.TargetBlending, 0f, 1f);
						Target.GaitTransition = EditorGUILayout.Slider("Gait Transition", Target.GaitTransition, 0f, 1f);
						Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);
						Target.SetRoot((Transform)EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true));
						Target.SetJointCount(EditorGUILayout.IntField("Joint Count", Target.Joints.Length));
						for(int i=0; i<Target.Joints.Length; i++) {
							if(Target.Joints[i] != null) {
								Utility.SetGUIColor(Utility.Green);
							} else {
								Utility.SetGUIColor(Utility.Red);
							}
							Target.SetJoint(i, (Transform)EditorGUILayout.ObjectField("Joint " + (i+1), Target.Joints[i], typeof(Transform), true));
							Utility.ResetGUIColor();
						}
					}
				}
			}
		}

}