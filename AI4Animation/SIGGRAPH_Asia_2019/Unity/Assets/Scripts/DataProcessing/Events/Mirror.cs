#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Mirror : SceneEvent {

	public bool X, Y, Z;

	public override void Callback(MotionEditor editor) {
        if(Blocked) {
            Identity(editor);
            return;
        }
		Vector3 scale = transform.localScale;
		scale.x = Mathf.Abs(scale.x) * (editor.Mirror && X ? -1f : 1f);
		scale.y = Mathf.Abs(scale.y) * (editor.Mirror && Y ? -1f : 1f);
		scale.z = Mathf.Abs(scale.z) * (editor.Mirror && Z ? -1f : 1f);
		transform.localScale = scale;
	}

	public override void Identity(MotionEditor editor) {
		Vector3 scale = transform.localScale;
		scale.x = Mathf.Abs(scale.x);
		scale.y = Mathf.Abs(scale.y);
		scale.z = Mathf.Abs(scale.z);
		transform.localScale = scale;
	}

}
#endif