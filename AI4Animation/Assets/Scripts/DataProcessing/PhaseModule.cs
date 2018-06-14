#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class PhaseModule : DataModule {

	public override void Inspector() {
		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField(Type.ToString());
		if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White)) {
			Data.RemoveModule(Type);
		}
		EditorGUILayout.EndHorizontal();
	}

}
#endif