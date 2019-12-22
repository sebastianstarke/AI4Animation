#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class RescaleChilds : SceneEvent {

	public Vector3 Default = Vector3.one;

	void Reset() {
		Default = transform.localScale;
	}

	public override void Callback(MotionEditor editor) {
        if(Blocked) {
            Identity(editor);
            return;
        }
		Random.InitState(editor.GetCurrentSeed());
		int index = editor.GetCurrentSeed() % transform.childCount;
		RescaleInfo info = transform.GetChild(index).GetComponent<RescaleInfo>();
		transform.localScale = info == null ? Default : Vector3.Scale(Default, Utility.UniformVector3(info.Min, info.Max));
	}

	public override void Identity(MotionEditor editor) {
		transform.localScale = Default;
	}

}
#endif