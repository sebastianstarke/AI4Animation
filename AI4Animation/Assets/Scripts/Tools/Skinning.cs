using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using Winterdust;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class Skinning : MonoBehaviour {

	public GameObject NewSkeleton;
	public float BendRange = 0.25f;
	public float BendFactor = 0.25f;

	public void Process() {
		Utility.Destroy(GetComponent<MeshSkinnerDebugWeights>());
		GameObject skeleton = Instantiate(NewSkeleton);
		skeleton.name = "Instance";
		new MeshSkinner(gameObject, skeleton).work(BendRange, BendFactor).debug().finish();
		Utility.Destroy(FindSkeleton());
		skeleton.name = "Skeleton";
		Cleanup(skeleton.transform);
		GetComponent<MeshSkinnerDebugWeights>().enabled = false;
	}

	private GameObject FindSkeleton() {
		for(int i=0; i<transform.childCount; i++) {
			if(transform.GetChild(i).name.Contains("Skeleton")) {
				return transform.GetChild(i).gameObject;
			}
		}
		return null;
	}

	private void Cleanup(Transform transform) {
		Component[] c = transform.GetComponents<Component>();
		for(int i=0; i<c.Length; i++) {
			if(!(c[i] is Transform)) {
				Utility.Destroy(c[i]);
			}
		}
		for(int i=0; i<transform.childCount; i++) {
			Cleanup(transform.GetChild(i));
		}
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
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Target.NewSkeleton = (GameObject)EditorGUILayout.ObjectField("Skeleton", Target.NewSkeleton, typeof(GameObject), true);
				Target.BendRange = EditorGUILayout.FloatField("Bend Range", Target.BendRange);
				Target.BendFactor = EditorGUILayout.FloatField("Bend Factor", Target.BendFactor);
				if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.Process();
				}
			}
		}
	}
	#endif

}