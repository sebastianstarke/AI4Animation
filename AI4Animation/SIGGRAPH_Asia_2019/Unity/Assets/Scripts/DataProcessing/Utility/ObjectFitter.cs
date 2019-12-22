#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class ObjectFitter : MonoBehaviour {

	public Transform Reference = null;
	public Vector3 Translation = Vector3.zero;
	public Vector3 Rotation = Vector3.zero;

	public void Fit() {
		Matrix4x4 t = Reference.GetWorldMatrix() * Matrix4x4.TRS(Translation, Quaternion.Euler(Rotation), Vector3.one);
		transform.position = t.GetPosition();
		transform.rotation = t.GetRotation();
	}

	[CustomEditor(typeof(ObjectFitter))]
	public class ObjectFitter_Editor : Editor {

		public ObjectFitter Target;

		void Awake() {
			Target = (ObjectFitter)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			DrawDefaultInspector();

			if(Utility.GUIButton("Fit", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.Fit();
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}

}
#endif
