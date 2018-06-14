#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class StyleModule : DataModule {

	public override TYPE Type() {
		return TYPE.Style;
	}

	public override DataModule Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		return this;
	}

	protected override void DerivedInspector(MotionEditor editor) {
		EditorGUILayout.LabelField("Content.");
	}

}
#endif