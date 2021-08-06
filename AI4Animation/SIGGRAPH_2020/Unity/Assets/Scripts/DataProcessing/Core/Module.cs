#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;

public abstract class Module : ScriptableObject {

	public enum ID {Root, Style, Phase, Contact, HeightMap, DepthMap, CylinderMap, CuboidMap, SphereMap, DistanceMap, Motion, Alignment, Dribble, Length};
	private static string[] IDs = null;
	public static string[] GetIDs() {
		if(IDs == null) {
			IDs = new string[(int)Module.ID.Length+1];
			for(int i=0; i<IDs.Length-1; i++) {
				IDs[i] = ((Module.ID)i).ToString();
			}
		}
		return IDs;
	}

	public MotionData Data;

	[NonSerialized] public bool Inspect = false;
	[NonSerialized] public bool Visualize = true;

    private Precomputable<ComponentSeries>[] PrecomputedRegularComponentSeries = null;
    private Precomputable<ComponentSeries>[] PrecomputedInverseComponentSeries = null;

	public void ResetPrecomputation() {
		PrecomputedRegularComponentSeries = Data.ResetPrecomputable(PrecomputedRegularComponentSeries);
		PrecomputedInverseComponentSeries = Data.ResetPrecomputable(PrecomputedInverseComponentSeries);
		DerivedResetPrecomputation();
	}

	public ComponentSeries ExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseComponentSeries[index] == null) {
				PrecomputedInverseComponentSeries[index] = new Precomputable<ComponentSeries>(Compute());
			}
			if(!mirrored && PrecomputedRegularComponentSeries[index] == null) {
				PrecomputedRegularComponentSeries[index] = new Precomputable<ComponentSeries>(Compute());
			}
			return mirrored ? PrecomputedInverseComponentSeries[index].Value : PrecomputedRegularComponentSeries[index].Value;
		}

		return Compute();
		ComponentSeries Compute() {
        	return DerivedExtractSeries(global, timestamp, mirrored);
		}
	}

	public Module Initialize(MotionData data) {
		Data = data;
		ResetPrecomputation();
		DerivedInitialize();
		return this;
	}

	public void Load(MotionEditor editor) {
		DerivedLoad(editor);
	}

	public void Callback(MotionEditor editor) {
		DerivedCallback(editor);
	}

	public void GUI(MotionEditor editor) {
		if(Visualize) {
			ExtractSeries(editor.GetTimeSeries(), editor.GetCurrentFrame().Timestamp, editor.Mirror).GUI();
			DerivedGUI(editor);
		}
	}

	public void Draw(MotionEditor editor) {
		if(Visualize) {
			ExtractSeries(editor.GetTimeSeries(), editor.GetCurrentFrame().Timestamp, editor.Mirror).Draw();
			DerivedDraw(editor);
		}
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
				EditorGUILayout.LabelField(GetID().ToString() + " Module");
				GUILayout.FlexibleSpace();
				if(Utility.GUIButton("Visualize", Visualize ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 75f, 20f)) {
					Visualize = !Visualize;
				}
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

	public abstract ID GetID();
	public abstract void DerivedResetPrecomputation();
	public abstract ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored);

	protected abstract void DerivedInitialize();
	protected abstract void DerivedLoad(MotionEditor editor);
	protected abstract void DerivedCallback(MotionEditor editor);
	protected abstract void DerivedGUI(MotionEditor editor);
	protected abstract void DerivedDraw(MotionEditor editor);
	protected abstract void DerivedInspector(MotionEditor editor);
	
}
#endif