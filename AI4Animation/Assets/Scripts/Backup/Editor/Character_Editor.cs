using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(B_Character))]
public class B_Character_Editor : Editor {

		public B_Character Target;

		void Awake() {
			Target = (B_Character)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			//DrawDefaultInspector();

			/*
			if(GUILayout.Button("+")) {
				Target.AddJoint(0);
			}
			*/
			for(int i=0; i<Target.Joints.Length; i++) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20));
					Target.Joints[i].Transform = (Transform)EditorGUILayout.ObjectField(Target.Joints[i].Transform, typeof(Transform), true);
					/*
					if(GUILayout.Button("x")) {
						Target.RemoveJoint(i);
					}
					*/
					EditorGUILayout.EndHorizontal();
				}
				/*
				if(GUILayout.Button("+")) {
					Target.AddJoint(i+1);
				}
				*/
			}
			if(GUILayout.Button("+")) {
				Target.AddJoint(Target.Joints.Length);
			}
			if(GUILayout.Button("-")) {
				Target.RemoveJoint(Target.Joints.Length);
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
}