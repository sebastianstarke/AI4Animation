#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public abstract class Module : ScriptableObject {

	public enum TYPE {Trajectory, Style, Phase, Contact, DepthMap, HeightMap, UmbrellaMap, Keypoint, Length};

	public MotionData Data;
	public bool Inspect;

	public void Inspector(MotionEditor editor) {
		Utility.SetGUIColor(UltiDraw.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Mustard);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.BeginHorizontal();
				Inspect = EditorGUILayout.Toggle(Inspect, GUILayout.Width(20f));
				EditorGUILayout.LabelField(Type().ToString() + " Module");
				GUILayout.FlexibleSpace();
				if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f)) {
					Data.RemoveModule(Type());
				}
				EditorGUILayout.EndHorizontal();
			}

			if(Inspect) {
				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					DerivedInspector(editor);
				}
			}
		}
	}

	public abstract TYPE Type();
	public abstract Module Initialise(MotionData data);
	public abstract void Draw(MotionEditor editor);
	protected abstract void DerivedInspector(MotionEditor editor);

}
#endif