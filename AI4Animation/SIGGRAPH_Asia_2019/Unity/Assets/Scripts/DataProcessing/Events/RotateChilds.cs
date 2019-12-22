#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class RotateChilds : SceneEvent {

	public Vector3 Min = Vector3.zero;
	public Vector3 Max = Vector3.zero;
	public Vector3 Default = Vector3.zero;

	void Reset() {
		Default = transform.localEulerAngles;
	}

	public override void Callback(MotionEditor editor) {
        if(Blocked) {
            Identity(editor);
            return;
        }
		Random.InitState(editor.GetCurrentSeed());
		transform.localEulerAngles = Default + new Vector3(Random.Range(Min.x, Max.x), Random.Range(Min.y, Max.y), Random.Range(Min.z, Max.z));
	}

	public override void Identity(MotionEditor editor) {
		transform.localEulerAngles = Default;
	}

}
#endif