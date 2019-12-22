#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class TimeSeriesModule : Module {

	public int PastKeys = 6;
	public int FutureKeys = 6;
	public float PastWindow = 1f;
	public float FutureWindow = 1f;
	public int Resolution = 1;

	public override ID GetID() {
		return ID.TimeSeries;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		
	}


	protected override void DerivedDraw(MotionEditor editor) {
		TimeSeries timeSeries = GetTimeSeries(editor.GetCurrentFrame(), editor.Mirror, 1f/editor.TargetFramerate);
		foreach(TimeSeries.Series series in timeSeries.Data) {
			if(series is TimeSeries.Root && Data.GetModule(ID.Root).Visualise) {
				((TimeSeries.Root)series).Draw();
			}
			if(series is TimeSeries.Style && Data.GetModule(ID.Style).Visualise) {
				((TimeSeries.Style)series).Draw();
			}
			if(series is TimeSeries.Goal && Data.GetModule(ID.Goal).Visualise) {
				((TimeSeries.Goal)series).Draw();
			}
			if(series is TimeSeries.Contact && Data.GetModule(ID.Contact).Visualise) {
				((TimeSeries.Contact)series).Draw();
			}
			if(series is TimeSeries.Phase && Data.GetModule(ID.Phase).Visualise) {
				((TimeSeries.Phase)series).Draw();
			}
		}
	}

	protected override void DerivedInspector(MotionEditor editor) {
		EditorGUILayout.BeginHorizontal();
		GUILayout.FlexibleSpace();
		EditorGUILayout.LabelField("Past Keys", GUILayout.Width(100f));
		PastKeys = EditorGUILayout.IntField(PastKeys, GUILayout.Width(50f));
		EditorGUILayout.LabelField("Future Keys", GUILayout.Width(100f));
		FutureKeys = EditorGUILayout.IntField(FutureKeys, GUILayout.Width(50f));
		EditorGUILayout.LabelField("Past Window", GUILayout.Width(100f));
		PastWindow = EditorGUILayout.FloatField(PastWindow ,GUILayout.Width(50f));
		EditorGUILayout.LabelField("Future Window", GUILayout.Width(100f));
		FutureWindow = EditorGUILayout.FloatField(FutureWindow, GUILayout.Width(50f));
		EditorGUILayout.LabelField("Resolution", GUILayout.Width(100f));
		Resolution = Mathf.Max(EditorGUILayout.IntField(Resolution, GUILayout.Width(50f)), 1);
		GUILayout.FlexibleSpace();
		EditorGUILayout.EndHorizontal();
	}

	public TimeSeries GetTimeSeries(Frame frame, bool mirrored, float delta) {
		return GetTimeSeries(frame, mirrored, PastKeys, FutureKeys, PastWindow, FutureWindow, Resolution, delta);
	}

	public TimeSeries GetTimeSeries(Frame frame, bool mirrored, int pastKeys, int futureKeys, float pastWindow, float futureWindow, int resolution, float delta) {
		TimeSeries timeSeries = new TimeSeries(pastKeys, futureKeys, pastWindow, futureWindow, resolution);
		foreach(Module module in Data.Modules) {
			if(module is RootModule) {
				RootModule m = (RootModule)module;
				TimeSeries.Root series = new TimeSeries.Root(timeSeries);
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					if(t < 0f || t > Data.GetTotalTime()) {
						series.Transformations[i] = m.GetEstimatedRootTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored);
						series.Velocities[i] = m.GetEstimatedRootVelocity(frame, timeSeries.Samples[i].Timestamp, mirrored, delta);
					} else {
						series.Transformations[i] = m.GetRootTransformation(Data.GetFrame(t), mirrored);
						series.Velocities[i] = m.GetRootVelocity(Data.GetFrame(t), mirrored, delta);
					}
				}
			}
			if(module is StyleModule) {
				StyleModule m = (StyleModule)module;
				TimeSeries.Style series = new TimeSeries.Style(timeSeries, m.GetNames());
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetStyles(Data.GetFrame(t));
				}
			}
			if(module is GoalModule) {
				GoalModule m = (GoalModule)module;
				TimeSeries.Goal series = new TimeSeries.Goal(timeSeries, m.GetNames());
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;		
					series.Transformations[i] = m.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, delta);
					series.Values[i] = m.GetActions(Data.GetFrame(t), delta);
					
				}
			}
			if(module is ContactModule) {
				ContactModule m = (ContactModule)module;
				TimeSeries.Contact series = new TimeSeries.Contact(timeSeries, m.GetNames());
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetContacts(Data.GetFrame(t), mirrored);
				}
			}
			if(module is PhaseModule) {
				PhaseModule m = (PhaseModule)module;
				TimeSeries.Phase series = new TimeSeries.Phase(timeSeries);
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetPhase(Data.GetFrame(t), mirrored);
				}
			}
		}
		return timeSeries;
	}

}
#endif
