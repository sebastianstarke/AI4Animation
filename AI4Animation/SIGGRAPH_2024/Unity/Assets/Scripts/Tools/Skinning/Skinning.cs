using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using Winterdust;
using AI4Animation;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class Skinning : MonoBehaviour {

	public GameObject NewSkeleton;
	public float BendRange = 0.25f;
	public float BendFactor = 0.25f;

	public void Process() {
		//MAYBE NEED REWORK DUE TO MAJOR ACTOR UPDATE
		Transform root = transform;
		Transform[] bones = GetComponent<Actor>() != null ? GetComponent<Actor>().GetBoneTransforms() : null;
		if(GetComponent<Actor>() != null) {
			Utility.Destroy(GetComponent<Actor>());
		}
		GameObject.DestroyImmediate(GetComponent<MeshSkinnerDebugWeights>());
		GameObject skeleton = Instantiate(NewSkeleton);
		skeleton.name = "Instance";
		new MeshSkinner(gameObject, skeleton).work(BendRange, BendFactor).debug().finish();
		GameObject.DestroyImmediate(FindSkeleton());
		skeleton.name = "Skeleton";
		Cleanup(skeleton.transform);
		GetComponent<MeshSkinnerDebugWeights>().enabled = false;
		if(GetComponent<Actor>() != null) {
			GetComponent<Actor>().Create(bones);
		}
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
				GameObject.DestroyImmediate(c[i]);
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

			DrawDefaultInspector();
			if(GUILayout.Button("Process")) {
				Target.Process();
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}
	#endif

}