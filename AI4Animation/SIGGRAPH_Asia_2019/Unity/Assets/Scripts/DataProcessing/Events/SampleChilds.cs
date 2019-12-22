#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SampleChilds : SceneEvent {

	public override void Callback(MotionEditor editor) {
        if(Blocked) {
            Identity(editor);
            return;
        }
        int index = editor.GetCurrentSeed() % transform.childCount;
        for(int i=0; i<transform.childCount; i++) {
            transform.GetChild(i).gameObject.SetActive(i == index);
        }
	}

	public override void Identity(MotionEditor editor) {

	}

}
#endif