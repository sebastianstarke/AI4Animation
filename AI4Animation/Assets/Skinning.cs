using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using Winterdust;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class Skinning : MonoBehaviour {

	public GameObject Skeleton;

	public void Process() {
		Utility.Destroy(GetComponent<MeshSkinnerDebugWeights>());
		GameObject skeleton = Instantiate(Skeleton);
		Utility.Destroy(skeleton.GetComponent<BioAnimation>());
		new MeshSkinner(gameObject, skeleton).work(0.25f, 0.25f, false, false).debug().finish();
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(Skinning))]
	public class Skinning_Editor : Editor {

		public Skinning Target;

		void Awake() {
			Target = (Skinning)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {
			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Target.Skeleton = (GameObject)EditorGUILayout.ObjectField("Skeleton", Target.Skeleton, typeof(GameObject), true);
				if(Utility.GUIButton("Process", Utility.DarkGrey, Utility.White)) {
					Target.Process();
				}
			}
		}
	}
	#endif

}