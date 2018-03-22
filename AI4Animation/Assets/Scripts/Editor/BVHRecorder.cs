using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Collections;

public class BVHRecorder : EditorWindow {

	public static EditorWindow Window;

	public BioAnimation Animation;
	public string Name = "Animation";
	public float FrameTime = 1f/60f;

	private BVHAnimation Data;
	private bool Recording = false;

	[MenuItem ("Addons/BVH Recorder")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHRecorder));
	}

	void Update() {
		if(!Application.isPlaying) {
			Data = null;
			Recording = false;
		}
	}

	private IEnumerator Record() {
		Data = ScriptableObject.CreateInstance<BVHAnimation>();
		Data.Character = Animation.Character;
		Data.FrameTime = FrameTime;

		Data.Trajectory = new Trajectory(0, Animation.Controller.Styles.Length);
		Data.PhaseFunction = new BVHAnimation.BVHPhaseFunction(Data);
		Data.MirroredPhaseFunction = new BVHAnimation.BVHPhaseFunction(Data);
		Data.StyleFunction = new BVHAnimation.BVHStyleFunction(Data);
		for(int i=0; i<Animation.Controller.Styles.Length; i++) {
			Data.StyleFunction.AddStyle(Animation.Controller.Styles[i].Name);
		}
		//Data.StyleFunction.SetStyle(BVHAnimation.BVHStyleFunction.STYLE.Quadruped);
		
		int index = 0;

		while(Recording && Application.isPlaying) {
			yield return new WaitForEndOfFrame();
			//Frames
			BVHAnimation.BVHFrame frame = new BVHAnimation.BVHFrame(Data, Data.GetTotalFrames()+1, Data.GetTotalFrames()*FrameTime);
			frame.Local = Data.Character.GetLocalTransformations();
			frame.World = Data.Character.GetWorldTransformations();
			ArrayExtensions.Add(ref Data.Frames, frame);
			
			//Trajectory
			Trajectory.Point point = new Trajectory.Point(Data.Trajectory.Points.Length, Animation.Controller.Styles.Length);
			point.SetTransformation(Animation.GetTrajectory().Points[60].GetTransformation());
			point.SetLeftsample(Animation.GetTrajectory().Points[60].GetLeftSample());
			point.SetRightSample(Animation.GetTrajectory().Points[60].GetRightSample());
			point.SetSlope(Animation.GetTrajectory().Points[60].GetSlope());
			for(int i=0; i<Animation.Controller.Styles.Length; i++) {
				point.Styles[i] = Animation.GetTrajectory().Points[60].Styles[i];
			}
			ArrayExtensions.Add(ref Data.Trajectory.Points, point);

			//Phase Function
			/*
			ArrayExtensions.Add(ref Data.PhaseFunction.Phase, Mathf.Repeat(Animation.GetPhase() / (2f*Mathf.PI), 1f));
			ArrayExtensions.Add(ref Data.PhaseFunction.Keys, index == 0 ? true : Data.PhaseFunction.Phase[index-1] > Data.PhaseFunction.Phase[index]);
			ArrayExtensions.Add(ref Data.PhaseFunction.Cycle, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.NormalisedCycle, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.Velocities, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.NormalisedVelocities, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.Heights, 0f);
			*/
			ArrayExtensions.Add(ref Data.PhaseFunction.Phase, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.Keys, false);
			ArrayExtensions.Add(ref Data.PhaseFunction.Cycle, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.NormalisedCycle, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.Velocities, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.NormalisedVelocities, 0f);
			ArrayExtensions.Add(ref Data.PhaseFunction.Heights, 0f);

			//Mirrored Phase Function
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.Phase, 0f);
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.Keys, false);
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.Cycle, 0f);
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.NormalisedCycle, 0f);
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.Velocities, 0f);
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.NormalisedVelocities, 0f);
			ArrayExtensions.Add(ref Data.MirroredPhaseFunction.Heights, 0f);

			//Style Function
			bool styleUpdate = false;
			for(int i=0; i<Animation.Controller.Styles.Length; i++) {
				ArrayExtensions.Add(ref Data.StyleFunction.Styles[i].Flags, Animation.Controller.Styles[i].Query());
				ArrayExtensions.Add(ref Data.StyleFunction.Styles[i].Values, Animation.GetTrajectory().Points[60].Styles[i]);
				if(index == 0) {
					styleUpdate = true;
				} else {
					if(Data.StyleFunction.Styles[i].Flags[index-1] != Data.StyleFunction.Styles[i].Flags[index]) {
						styleUpdate = true;
					}
				}
			}
			ArrayExtensions.Add(ref Data.StyleFunction.Keys, styleUpdate);

			index += 1;
		}

		//Setup
		Data.TimeWindow = Data.GetTotalTime();
		Data.Corrections = new Vector3[Animation.Character.Hierarchy.Length];
		Data.DetectSymmetry();

		//Postprocess
		Data.PhaseFunction.Keys[index-1] = true;
		Data.StyleFunction.Keys[index-1] = true;
		Data.PhaseFunction.Recompute();
		Data.StyleFunction.Recompute();

		//Finish
		Recording = false;
	}

	private void Save() {
		if(AssetDatabase.LoadAssetAtPath("Assets/Project/"+Name+".asset", typeof(BVHAnimation)) == null) {
			AssetDatabase.CreateAsset(Data , "Assets/Project/"+Name+".asset");
		} else {
			int i = 1;
			while(AssetDatabase.LoadAssetAtPath("Assets/Project/"+Name+" ("+i+").asset", typeof(BVHAnimation)) != null) {
				i += 1;
			}
			AssetDatabase.CreateAsset(Data, "Assets/Project/"+Name+" ("+i+").asset");
		}
	}

	void OnGUI() {
		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Recorder");
				}

				if(!Application.isPlaying) {
					EditorGUILayout.LabelField("Change into play mode to start recording.");
					return;
				}

				Animation = (BioAnimation)EditorGUILayout.ObjectField("Animation", Animation, typeof(BioAnimation), true);
				Name = EditorGUILayout.TextField("Name", Name);
				FrameTime = EditorGUILayout.FloatField("Frame Time", FrameTime);

				if(Data == null) {
					EditorGUILayout.LabelField("No data recorded.");
				} else {
					EditorGUILayout.LabelField("Frames: " + Data.Frames.Length);
				}

				if(Utility.GUIButton(Recording ? "Stop" : "Start", Recording ? UltiDraw.DarkRed : UltiDraw.DarkGreen, UltiDraw.White)) {
					Recording = !Recording;
					if(Recording) {
						Animation.StartCoroutine(Record());
					}
				}

				if(Utility.GUIButton("Save", UltiDraw.DarkGrey, UltiDraw.White)) {
					Save();
				}

			}
		}
	}

}
