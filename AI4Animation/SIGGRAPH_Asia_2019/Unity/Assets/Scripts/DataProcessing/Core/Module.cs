#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public abstract class Module : ScriptableObject {

	public enum ID {TimeSeries, Root, Style, Goal, Phase, Contact, HeightMap, DepthMap, CylinderMap, CuboidMap, SphereMap, DistanceMap, Motion, Length};

	public MotionData Data;
	[System.NonSerialized] public bool Inspect = false;
	[System.NonSerialized] public bool Visualise = true;

	private static string[] Names = null;

	public static string[] GetIDNames() {
		if(Names == null) {
			Names = new string[(int)Module.ID.Length+1];
			for(int i=0; i<Names.Length-1; i++) {
				Names[i] = ((Module.ID)i).ToString();
			}
		}
		return Names;
	}

	public void Inspector(MotionEditor editor) {
		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Mustard);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.BeginHorizontal();
				Inspect = EditorGUILayout.Toggle(Inspect, GUILayout.Width(20f));
				if(Utility.GUIButton("V", Visualise ? UltiDraw.DarkGreen : UltiDraw.DarkGrey, UltiDraw.White, 20f, 16f)) {
					Visualise = !Visualise;
				}
				EditorGUILayout.LabelField(GetID().ToString() + " Module");
				GUILayout.FlexibleSpace();
				if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 25f, 20f)) {
					Data.RemoveModule(GetID());
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

	public void Draw(MotionEditor editor) {
		if(Visualise && editor.Visualise) {
			DerivedDraw(editor);
		}
	}

	public abstract ID GetID();
	public abstract Module Initialise(MotionData data);
	public abstract void Slice(Sequence sequence);
	public abstract void Callback(MotionEditor editor);
	protected abstract void DerivedDraw(MotionEditor editor);
	protected abstract void DerivedInspector(MotionEditor editor);

}
#endif