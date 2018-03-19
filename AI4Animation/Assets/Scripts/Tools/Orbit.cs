using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteInEditMode]
public class Orbit : MonoBehaviour {

	public Transform Pivot;
	public float Distance = 0f;
	//public bool Override = false;

	public void Update() {
		if(Pivot == null) {
			return;
		}
		Vector3 direction = transform.position - Pivot.position;
		//if(Override) {
		//	transform.OverridePosition(Pivot.position + Distance * direction.normalized);
		//} else {
			transform.position = Pivot.position + Distance * direction.normalized;
		//}
	}

	public void SetPivot(Transform pivot) {
		if(Pivot == pivot) {
			return;
		}
		if(pivot != null) {
			Pivot = pivot;
			Distance = Vector3.Distance(Pivot.position, transform.position);
		} else {
			Distance = 0f;
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(Orbit))]
	public class Orbit_Editor : Editor {
		public Orbit Target;

		void Awake() {
			Target = (Orbit)target;
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
				Target.SetPivot((Transform)EditorGUILayout.ObjectField("Pivot", Target.Pivot, typeof(Transform), true));
				Target.Distance = EditorGUILayout.FloatField("Distance", Target.Distance);
				//Target.Override = EditorGUILayout.Toggle("Override", Target.Override);
			}
		}
	}
	#endif

}
