using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHAnimation : ScriptableObject {

	public BVHData Data;
	public Character Character;
	public bool[] Bones;
	public int[] Symmetry;
	public bool MirrorX, MirrorY, MirrorZ;

	public float UnitScale = 100f;
	public Vector3[] Corrections;
	
	public BVHFrame[] Frames = new BVHFrame[0];
	public Trajectory Trajectory;
	public BVHPhaseFunction PhaseFunction;
	public BVHPhaseFunction MirroredPhaseFunction;
	public BVHStyleFunction StyleFunction;

	public BVHSequence[] Sequences = new BVHSequence[0];

	public bool ShowMirrored = false;
	public bool ShowPreview = false;
	public bool ShowVelocities = false;
	public bool ShowTrajectory = false;
	public bool ShowFlow = false;
	public bool ShowZero = false;

	public BVHFrame CurrentFrame = null;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public bool Playing = false;
	public float Timescale = 1f;
	public float TimeWindow = 0f;
	public System.DateTime Timestamp;

	public bool ExportScreenshots = false;
	public bool SkipExportScreenshot = false;

	public void EditorUpdate() {
		if(Playing) {
			PlayTime += Timescale*(float)Utility.GetElapsedTime(Timestamp);
			if(PlayTime > GetTotalTime()) {
				PlayTime -= GetTotalTime();
			}
			LoadFrame(PlayTime);
		}
		Timestamp = Utility.GetTimestamp();

		PhaseFunction.EditorUpdate();
		MirroredPhaseFunction.EditorUpdate();
	}

	public BVHAnimation Create(BVHEditor editor) {
		Load(editor.Path);
		string name = editor.Path.Substring(editor.Path.LastIndexOf("/")+1);
		if(AssetDatabase.LoadAssetAtPath("Assets/Project/"+name+".asset", typeof(BVHAnimation)) == null) {
			AssetDatabase.CreateAsset(this , "Assets/Project/"+name+".asset");
		} else {
			int i = 1;
			while(AssetDatabase.LoadAssetAtPath("Assets/Project/"+name+" ("+i+").asset", typeof(BVHAnimation)) != null) {
				i += 1;
			}
			AssetDatabase.CreateAsset(this, "Assets/Project/"+name+" ("+i+").asset");
		}
		return this;
	}

	public int GetTotalFrames() {
		return Frames.Length;
	}

	public float GetTotalTime() {
		return Frames.Length * FrameTime;
	}

	private void Load(string path) {
		string[] lines = File.ReadAllLines(path);
		char[] whitespace = new char[] {' '};
		int index = 0;

		//Build Hierarchy
		Data = new BVHData();
		string name = string.Empty;
		string parent = string.Empty;
		Vector3 offset = Vector3.zero;
		int[] channels = null;
		for(index = 0; index<lines.Length; index++) {
			if(lines[index] == "MOTION") {
				break;
			}
			string[] entries = lines[index].Split(whitespace);
			for(int entry=0; entry<entries.Length; entry++) {
				if(entries[entry].Contains("ROOT")) {
					parent = "None";
					name = entries[entry+1];
					break;
				} else if(entries[entry].Contains("JOINT")) {
					parent = name;
					name = entries[entry+1];
					break;
				} else if(entries[entry].Contains("End")) {
					parent = name;
					name = name+entries[entry+1];
					string[] subEntries = lines[index+2].Split(whitespace);
					for(int subEntry=0; subEntry<subEntries.Length; subEntry++) {
						if(subEntries[subEntry].Contains("OFFSET")) {
							offset.x = Utility.ReadFloat(subEntries[subEntry+1]);
							offset.y = Utility.ReadFloat(subEntries[subEntry+2]);
							offset.z = Utility.ReadFloat(subEntries[subEntry+3]);
							break;
						}
					}
					Data.AddBone(name, parent, offset, new int[0]);
					index += 2;
					break;
				} else if(entries[entry].Contains("OFFSET")) {
					offset.x = Utility.ReadFloat(entries[entry+1]);
					offset.y = Utility.ReadFloat(entries[entry+2]);
					offset.z = Utility.ReadFloat(entries[entry+3]);
					break;
				} else if(entries[entry].Contains("CHANNELS")) {
					channels = new int[Utility.ReadInt(entries[entry+1])];
					for(int i=0; i<channels.Length; i++) {
						if(entries[entry+2+i] == "Xposition") {
							channels[i] = 1;
						} else if(entries[entry+2+i] == "Yposition") {
							channels[i] = 2;
						} else if(entries[entry+2+i] == "Zposition") {
							channels[i] = 3;
						} else if(entries[entry+2+i] == "Xrotation") {
							channels[i] = 4;
						} else if(entries[entry+2+i] == "Yrotation") {
							channels[i] = 5;
						} else if(entries[entry+2+i] == "Zrotation") {
							channels[i] = 6;
						}
					}
					Data.AddBone(name, parent, offset, channels);
					break;
				} else if(entries[entry].Contains("}")) {
					name = parent;
					parent = name == "None" ? "None" : Data.FindBone(name).Parent;
					break;
				}
			}
		}

		//Read frame count
		index += 1;
		while(lines[index].Length == 0) {
			index += 1;
		}
		System.Array.Resize(ref Frames, Utility.ReadInt(lines[index].Substring(8)));

		//Read frame time
		index += 1;
		FrameTime = Utility.ReadFloat(lines[index].Substring(12));

		//Read motions
		index += 1;
		for(int i=index; i<lines.Length; i++) {
			Data.AddMotion(Utility.ReadArray(lines[i]));
		}

		//Generate character
		GenerateCharacter();

		//Required
		Corrections = new Vector3[Character.Hierarchy.Length];

		//Generate frames
		for(int i=0; i<GetTotalFrames(); i++) {
			Frames[i] = new BVHFrame(this, i+1, i*FrameTime);
		}

		//Initialise variables
		TimeWindow = GetTotalTime();
		PhaseFunction = new BVHPhaseFunction(this);
		MirroredPhaseFunction = new BVHPhaseFunction(this);
		PhaseFunction.Initialise();
		MirroredPhaseFunction.Initialise();
		StyleFunction = new BVHStyleFunction(this);
		Sequences = new BVHSequence[0];

		//Finalise
		DetectSymmetry();
		ComputeFrames();
		ComputeTrajectory();

		//Load First Frame
		LoadFrame(1);
	}

	public void Play() {
		PlayTime = CurrentFrame.Timestamp;
		Timestamp = Utility.GetTimestamp();
		Playing = true;
	}

	public void Stop() {
		Playing = false;
	}

	public void LoadNextFrame() {
		LoadFrame(Mathf.Min(CurrentFrame.Index+1, GetTotalFrames()));
	}

	public void LoadPreviousFrame() {
		LoadFrame(Mathf.Max(CurrentFrame.Index-1, 1));
	}

	public void LoadFrame(BVHFrame frame) {
		CurrentFrame = frame;
	}

	public void LoadFrame(int index) {
		LoadFrame(GetFrame(index));
	}

	public void LoadFrame(float time) {
		LoadFrame(GetFrame(time));
	}

	public BVHFrame GetFrame(int index) {
		if(index < 1 || index > GetTotalFrames()) {
			Debug.Log("Please specify an index between 1 and " + GetTotalFrames() + ".");
			return null;
		}
		return Frames[index-1];
	}

	public BVHFrame GetFrame(float time) {
		if(time < 0f || time > GetTotalTime()) {
			Debug.Log("Please specify a time between 0 and " + GetTotalTime() + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.RoundToInt(time / FrameTime) + 1, GetTotalFrames()));
	}

	public BVHFrame[] GetFrames(int startIndex, int endIndex) {
		int count = endIndex-startIndex+1;
		BVHFrame[] frames = new BVHFrame[count];
		int index = 0;
		for(float i=startIndex; i<=endIndex; i++) {
			frames[index] = GetFrame(i);
			index += 1;
		}
		return frames;
	}

	public BVHFrame[] GetFrames(float startTime, float endTime) {
		List<BVHFrame> frames = new List<BVHFrame>();
		for(float t=startTime; t<=endTime; t+=FrameTime) {
			frames.Add(GetFrame(t));
		}
		return frames.ToArray();
	}

	public void GenerateCharacter() {
		Character = new Character();
		string[] names = new string[Data.Bones.Length];
		string[] parents = new string[Data.Bones.Length];
		for(int i=0; i<Data.Bones.Length; i++) {
			names[i] = Data.Bones[i].Name;
			parents[i] = Data.Bones[i].Parent;
		}
		Character.BuildHierarchy(names, parents);
		Bones = new bool[Character.Hierarchy.Length];
		for(int i=0; i<Bones.Length; i++) {
			Bones[i] = true;
		}
	}

	public void AssignDogCorrections() {
		Corrections = new Vector3[Character.Hierarchy.Length];
		//Only for stupid dog bvh...
		for(int i=0; i<Character.Hierarchy.Length; i++) {
			if(	Character.Hierarchy[i].GetName() == "Head" ||
				Character.Hierarchy[i].GetName() == "HeadSite" ||
				Character.Hierarchy[i].GetName() == "LeftShoulder" ||
				Character.Hierarchy[i].GetName() == "RightShoulder"
				) {
				Corrections[i].x = 90f;
				Corrections[i].y = 90f;
				Corrections[i].z = 90f;
			}
			if(Character.Hierarchy[i].GetName() == "Tail") {
				Corrections[i].x = -45f;
			}
		}
		//
	}

	public void DetectSymmetry() {
		Symmetry = new int[Character.Hierarchy.Length];
		for(int i=0; i<Character.Hierarchy.Length; i++) {
			string name = Character.Hierarchy[i].GetName();
			if(name.Contains("Left")) {
				Character.Segment bone = Character.FindSegment("Right"+name.Substring(4));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.GetIndex();
				}
			} else if(name.Contains("Right")) {
				Character.Segment bone = Character.FindSegment("Left"+name.Substring(5));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.GetIndex();
				}
			} else if(name.StartsWith("L") && char.IsUpper(name[1])) {
				Character.Segment bone = Character.FindSegment("R"+name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.GetIndex();
				}
			} else if(name.StartsWith("R") && char.IsUpper(name[1])) {
				Character.Segment bone = Character.FindSegment("L"+name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.GetIndex();
				}
			} else {
				Symmetry[i] = i;
			}
		}
		MirrorX = false;
		MirrorY = false;
		MirrorZ = true;
	}

	public void ComputeFrames() {
		if(Data.Motions.Length == 0) {
			Debug.Log("No motions available.");
			return;
		}
		for(int i=0; i<Frames.Length; i++) {
			Frames[i].Generate();
		}
	}

	public void ComputeTrajectory() {
		Trajectory = new Trajectory(GetTotalFrames(), 0);
		LayerMask mask = LayerMask.GetMask("Ground");
		for(int i=0; i<GetTotalFrames(); i++) {
			Vector3 rootPos = Utility.ProjectGround(Frames[i].World[0].GetPosition(), mask);
			Trajectory.Points[i].SetPosition(rootPos);

			Vector3 rootDir = Frames[i].World[0].GetRotation() * Vector3.forward;
			rootDir.y = 0f;
			rootDir = rootDir.normalized;
			Trajectory.Points[i].SetDirection(rootDir);

			Vector3 rootVel = (rootPos - Trajectory.Points[Mathf.Clamp(i-1, 0, GetTotalFrames()-1)].GetPosition()) / FrameTime;
			Trajectory.Points[i].SetVelocity(rootVel);
			
			Trajectory.Points[i].Postprocess();

			/*
			//HARDCODED FOR DOG
			int hipIndex = Character.FindSegment("Hips").GetIndex();
			int neckIndex = Character.FindSegment("Neck").GetIndex();
			Vector3 rootDir = Frames[i].World[neckIndex].GetPosition() - Frames[i].World[hipIndex].GetPosition();
			*/
		}
	}

	public Matrix4x4[] ExtractZeroPosture(bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[Character.Hierarchy.Length];
		for(int i=0; i<Character.Hierarchy.Length; i++) {
			BVHData.Bone info = Data.Bones[i];
			Character.Segment parent = Character.Hierarchy[i].GetParent(Character.Hierarchy);
			Matrix4x4 local = Matrix4x4.TRS(
				info.Offset / UnitScale,
				Quaternion.identity,
				Vector3.one
				);
			transformations[i] = parent == null ? local : transformations[parent.GetIndex()] * local;
		}
		for(int i=0; i<Character.Hierarchy.Length; i++) {
			transformations[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Corrections[i]), Vector3.one);
		}
		if(mirrored) {
			for(int i=0; i<Character.Hierarchy.Length; i++) {
				transformations[i] = transformations[i].GetMirror(GetMirrorAxis());
			}
		}
		return transformations;
	}

	public Matrix4x4[] ExtractPosture(BVHFrame frame, bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[Character.Hierarchy.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = mirrored ? frame.World[Symmetry[i]].GetMirror(GetMirrorAxis()) : frame.World[i];
		}
		return transformations;
	}

	/*
	public Vector3[] ExtractVelocities(BVHFrame frame, bool mirrored, float smoothing) {
		Vector3[] velocities = new Vector3[Character.Hierarchy.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = mirrored ? frame.ComputeVelocity(Symmetry[i], smoothing).GetMirror(GetMirrorAxis()) : frame.ComputeVelocity(i, smoothing);
		}
		return velocities;
	}
	*/

	public Vector3[] ExtractBoneVelocities(BVHFrame frame, bool mirrored) {
		Vector3[] velocities = new Vector3[Character.Hierarchy.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = mirrored ? frame.ComputeBoneVelocity(Symmetry[i]).GetMirror(GetMirrorAxis()) : frame.ComputeBoneVelocity(i);
		}
		return velocities;
	}

	public Trajectory ExtractTrajectory(BVHFrame frame, bool mirrored) {
		Trajectory trajectory = new Trajectory(12, StyleFunction.Styles.Length);
		//Past
		for(int i=0; i<6; i++) {
			float timestamp = Mathf.Clamp(frame.Timestamp - 1f + (float)i/6f, 0f, GetTotalTime());
			int index = GetFrame(timestamp).Index;
			trajectory.Points[i].SetIndex(Trajectory.Points[index-1].GetIndex());
			trajectory.Points[i].SetPosition(Trajectory.Points[index-1].GetPosition());
			trajectory.Points[i].SetDirection(Trajectory.Points[index-1].GetDirection());
			trajectory.Points[i].SetVelocity(Trajectory.Points[index-1].GetVelocity());
			trajectory.Points[i].SetLeftsample(Trajectory.Points[index-1].GetLeftSample());
			trajectory.Points[i].SetRightSample(Trajectory.Points[index-1].GetRightSample());
			trajectory.Points[i].SetSlope(Trajectory.Points[index-1].GetSlope());
			for(int j=0; j<StyleFunction.Styles.Length; j++) {
				trajectory.Points[i].Styles[j] = StyleFunction.Styles[j].Values[index-1];
			}
		}
		//Current
		trajectory.Points[6].SetIndex(Trajectory.Points[frame.Index-1].GetIndex());
		trajectory.Points[6].SetPosition(Trajectory.Points[frame.Index-1].GetPosition());
		trajectory.Points[6].SetDirection(Trajectory.Points[frame.Index-1].GetDirection());
		trajectory.Points[6].SetVelocity(Trajectory.Points[frame.Index-1].GetVelocity());
		trajectory.Points[6].SetLeftsample(Trajectory.Points[frame.Index-1].GetLeftSample());
		trajectory.Points[6].SetRightSample(Trajectory.Points[frame.Index-1].GetRightSample());
		trajectory.Points[6].SetSlope(Trajectory.Points[frame.Index-1].GetSlope());
		for(int j=0; j<StyleFunction.Styles.Length; j++) {
			trajectory.Points[6].Styles[j] = StyleFunction.Styles[j].Values[frame.Index-1];
		}
		//Future
		for(int i=7; i<12; i++) {
			float timestamp = Mathf.Clamp(frame.Timestamp + (float)(i-6)/5f, 0f, GetTotalTime());
			int index = GetFrame(timestamp).Index;
			trajectory.Points[i].SetIndex(Trajectory.Points[index-1].GetIndex());
			trajectory.Points[i].SetPosition(Trajectory.Points[index-1].GetPosition());
			trajectory.Points[i].SetDirection(Trajectory.Points[index-1].GetDirection());
			trajectory.Points[i].SetVelocity(Trajectory.Points[index-1].GetVelocity());
			trajectory.Points[i].SetLeftsample(Trajectory.Points[index-1].GetLeftSample());
			trajectory.Points[i].SetRightSample(Trajectory.Points[index-1].GetRightSample());
			trajectory.Points[i].SetSlope(Trajectory.Points[index-1].GetSlope());
			for(int j=0; j<StyleFunction.Styles.Length; j++) {
				trajectory.Points[i].Styles[j] = StyleFunction.Styles[j].Values[index-1];
			}
		}

		if(mirrored) {
			for(int i=0; i<12; i++) {
				trajectory.Points[i].SetPosition(trajectory.Points[i].GetPosition().GetMirror(GetMirrorAxis()));
				trajectory.Points[i].SetDirection(trajectory.Points[i].GetDirection().GetMirror(GetMirrorAxis()));
				trajectory.Points[i].SetVelocity(trajectory.Points[i].GetVelocity().GetMirror(GetMirrorAxis()));
				trajectory.Points[i].SetLeftsample(trajectory.Points[i].GetLeftSample().GetMirror(GetMirrorAxis()));
				trajectory.Points[i].SetRightSample(trajectory.Points[i].GetRightSample().GetMirror(GetMirrorAxis()));
			}
		}

		return trajectory;
	}

	public Vector3 GetMirrorAxis() {
		if(MirrorX) {
			return Vector3.right;
		}
		if(MirrorY) {
			return Vector3.up;
		}
		if(MirrorZ) {
			return Vector3.forward;
		}
		return Vector3.zero;
	}

	private void SetUnitScale(float unitScale) {
		if(UnitScale != unitScale) {
			UnitScale = unitScale;
			ComputeFrames();
			ComputeTrajectory();
		}
	}

	private void SetCorrection(int index, Vector3 correction) {
		if(Corrections[index] != correction) {
			Corrections[index] = correction;
			ComputeFrames();
		}
	}

	private void AddSequence() {
		System.Array.Resize(ref Sequences, Sequences.Length+1);
		Sequences[Sequences.Length-1] = new BVHSequence(this);
	}

	private void RemoveSequence() {
		if(Sequences.Length > 0) {
			System.Array.Resize(ref Sequences, Sequences.Length-1);
		}
	}

	public void Inspector() {
		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton(ShowMirrored ? "Mirrored" : "Default", UltiDraw.Cyan, UltiDraw.Black)) {
				ShowMirrored = !ShowMirrored;
			}
			EditorGUILayout.EndHorizontal();
		}

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("Show Velocities", ShowVelocities ? UltiDraw.Green : UltiDraw.Grey, ShowVelocities ? UltiDraw.Black : UltiDraw.LightGrey)) {
				ShowVelocities = !ShowVelocities;
			}
			if(Utility.GUIButton("Show Trajectory", ShowTrajectory ? UltiDraw.Green : UltiDraw.Grey, ShowTrajectory ? UltiDraw.Black : UltiDraw.LightGrey)) {
				ShowTrajectory = !ShowTrajectory;
			}
			if(Utility.GUIButton("Show Preview", ShowPreview ? UltiDraw.Green : UltiDraw.Grey, ShowPreview ? UltiDraw.Black : UltiDraw.LightGrey)) {
				ShowPreview = !ShowPreview;
			}
			if(Utility.GUIButton("Show Flow", ShowFlow ? UltiDraw.Green : UltiDraw.Grey, ShowFlow ? UltiDraw.Black : UltiDraw.LightGrey)) {
				ShowFlow = !ShowFlow;
			}
			if(Utility.GUIButton("Show Zero", ShowZero ? UltiDraw.Green : UltiDraw.Grey, ShowZero ? UltiDraw.Black : UltiDraw.LightGrey)) {
				ShowZero = !ShowZero;
			}
			EditorGUILayout.EndHorizontal();
		}

		
		if(Utility.GUIButton("Recompute Trajectory", UltiDraw.Brown, UltiDraw.White)) {
			ComputeTrajectory();
		}

		Utility.SetGUIColor(UltiDraw.LightGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Frames: " + GetTotalFrames(), GUILayout.Width(100f));
			EditorGUILayout.LabelField("Time: " + GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
			EditorGUILayout.LabelField("Time/Frame: " + FrameTime.ToString("F3") + "s" + " (" + (1f/FrameTime).ToString("F1") + "Hz)", GUILayout.Width(175f));
			EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
			Timescale = EditorGUILayout.FloatField(Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
			EditorGUILayout.EndHorizontal();

			//for(int i=0; i<Frames.Length; i++) {
			//	if(StyleFunction.Keys[i] && StyleFunction.Styles[2].Flags[i]) {
			//		EditorGUILayout.LabelField("Jumping at frame " + (i+1));
			//	}
			//}
		}

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			EditorGUILayout.BeginHorizontal();
			if(Playing) {
				if(Utility.GUIButton("||", Color.red, Color.black, 20f, 20f)) {
					Stop();
				}
			} else {
				if(Utility.GUIButton("|>", Color.green, Color.black, 20f, 20f)) {
					Play();
				}
			}
			if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
				LoadPreviousFrame();
			}
			if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
				LoadNextFrame();
			}
			BVHAnimation.BVHFrame frame = GetFrame(EditorGUILayout.IntSlider(CurrentFrame.Index, 1, GetTotalFrames(), GUILayout.Width(440f)));
			if(CurrentFrame != frame) {
				PlayTime = frame.Timestamp;
				LoadFrame(frame);
			}
			EditorGUILayout.LabelField(CurrentFrame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
			EditorGUILayout.EndHorizontal();
		}

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("-1s", UltiDraw.Grey, UltiDraw.White, 65, 20f)) {
				LoadFrame(Mathf.Max(CurrentFrame.Timestamp - 1f, 0f));
			}
			TimeWindow = EditorGUILayout.Slider(TimeWindow, 2f*FrameTime, GetTotalTime(), GUILayout.Width(440f));
			if(Utility.GUIButton("+1s", UltiDraw.Grey, UltiDraw.White, 65, 20f)) {
				LoadFrame(Mathf.Min(CurrentFrame.Timestamp + 1f, GetTotalTime()));
			}
			EditorGUILayout.EndHorizontal();
		}

		if(ShowMirrored) {
			MirroredPhaseFunction.Inspector();
		} else {
			PhaseFunction.Inspector();
		}

		StyleFunction.Inspector();

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.LabelField("Sequences");
			}

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				for(int i=0; i<Sequences.Length; i++) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Start", GUILayout.Width(67f));
					Sequences[i].Start = EditorGUILayout.IntSlider(Sequences[i].Start, 1, GetTotalFrames(), GUILayout.Width(182f));
					EditorGUILayout.LabelField("End", GUILayout.Width(67f));
					Sequences[i].End = EditorGUILayout.IntSlider(Sequences[i].End, 1, GetTotalFrames(), GUILayout.Width(182f));
					EditorGUILayout.LabelField("Export", GUILayout.Width(67f));
					Sequences[i].Export = Mathf.Max(1, EditorGUILayout.IntField(Sequences[i].Export, GUILayout.Width(182f)));
					if(Utility.GUIButton("Auto", UltiDraw.DarkGrey, UltiDraw.White)) {
						Sequences[i].Auto();
					}
					EditorGUILayout.EndHorizontal();
				}

				if(Utility.GUIButton("Add Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
					AddSequence();
				}
				if(Utility.GUIButton("Remove Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
					RemoveSequence();
				}
			}
		}

		
		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.LabelField("Armature");
			}

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Character.Inspector();

				if(Utility.GUIButton("Export Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
					ExportSkeleton();
				}

				SetUnitScale(EditorGUILayout.FloatField("Unit Scale", UnitScale));
			}
		}

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.LabelField("Setup");
			}

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Mirror X", MirrorX ? UltiDraw.Cyan : UltiDraw.Grey, MirrorX ? UltiDraw.Black : UltiDraw.LightGrey)) {
						MirrorX = !MirrorX;
						MirrorY = false;
						MirrorZ = false;
					}
					if(Utility.GUIButton("Mirror Y", MirrorY ? UltiDraw.Cyan : UltiDraw.Grey, MirrorY ? UltiDraw.Black : UltiDraw.LightGrey)) {
						MirrorY = !MirrorY;
						MirrorX = false;
						MirrorZ = false;
					}
					if(Utility.GUIButton("Mirror Z", MirrorZ ? UltiDraw.Cyan : UltiDraw.Grey, MirrorZ ? UltiDraw.Black : UltiDraw.LightGrey)) {
						MirrorZ = !MirrorZ;
						MirrorX = false;
						MirrorY = false;
					}
					EditorGUILayout.EndHorizontal();
				}
				if(Utility.GUIButton("Auto Detect", UltiDraw.DarkGrey, UltiDraw.White)) {
					DetectSymmetry();
				}
				string[] names = new string[Character.Hierarchy.Length];
				for(int i=0; i<Character.Hierarchy.Length; i++) {
					names[i] = (i+1) + ": " + Character.Hierarchy[i].GetName();
				}
				for(int i=0; i<Character.Hierarchy.Length; i++) {
					EditorGUILayout.BeginHorizontal();
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.TextField(names[i]);
					EditorGUI.EndDisabledGroup();

					if(Bones == null) {
						Bones = new bool[Character.Hierarchy.Length];
						for(int j=0; j<Bones.Length; j++) {
							Bones[j] = true;
						}
					}
					if(Bones.Length != Character.Hierarchy.Length) {
						Bones = new bool[Character.Hierarchy.Length];
						for(int j=0; j<Bones.Length; j++) {
							Bones[j] = true;
						}
					}

					Bones[i] = EditorGUILayout.Toggle(Bones[i]);
					SetCorrection(i, EditorGUILayout.Vector3Field("", Corrections[i]));
					Symmetry[i] = EditorGUILayout.Popup(Symmetry[i], names);
					EditorGUILayout.EndHorizontal();
				}
			}
		}
	}

	private void ExportSkeleton() {
		int active = 0;
		Transform skeleton = ExportSkeleton(Character.GetRoot(), null, ref active);
		Transform root = new GameObject("Skeleton").transform;
		root.position = new Vector3(skeleton.position.x, 0f, skeleton.position.z);
		root.rotation = skeleton.rotation;
		skeleton.SetParent(root.transform);

		//BioAnimation animation = root.gameObject.AddComponent<BioAnimation>();
		/*
		animation.Joints = new Transform[active];
		int index = 0;
		AssignJoints(skeleton, ref animation.Joints, ref index);
		*/
	}

	private Transform ExportSkeleton(Character.Segment bone, Transform parent, ref int active) {
		Transform instance = parent;
		if(Bones[bone.GetIndex()]) {
			active += 1;
			instance = new GameObject(bone.GetName()).transform;
			instance.SetParent(parent);
			instance.position = bone.GetTransformation().GetPosition();
			instance.rotation = bone.GetTransformation().GetRotation();
			for(int i=0; i<bone.GetChildCount(); i++) {
				ExportSkeleton(bone.GetChild(Character.Hierarchy, i), instance, ref active);
			}
		} else {
			for(int i=0; i<bone.GetChildCount(); i++) {
				ExportSkeleton(bone.GetChild(Character.Hierarchy, i), parent, ref active);
			}
		}
		return instance.root;
	}

	private void AssignJoints(Transform t, ref Transform[] joints, ref int index) {
		joints[index] = t;
		index += 1;
		for(int i=0; i<t.childCount; i++) {
			AssignJoints(t.GetChild(i), ref joints, ref index);
		}
	}

	public void Draw() {
		if(ShowFlow) {
			Matrix4x4[] matrices = ShowZero ? ExtractZeroPosture(ShowMirrored) : ExtractPosture(CurrentFrame, ShowMirrored);
			for(int i=0; i<Character.Hierarchy.Length; i++) {
				Character.Hierarchy[i].SetTransformation(matrices[i]);
			}
			Character.Draw(UltiDraw.Blue, UltiDraw.Yellow, 1f);

			int steps = 5;
			float timespan = 0.1f;

			//Past
			for(int i=1; i<=steps; i++) {
				float timestamp = CurrentFrame.Timestamp - i*timespan/steps;
				if(timestamp >= 0) {
					Matrix4x4[] t = ExtractPosture(GetFrame(timestamp), ShowMirrored);
					for(int j=0; j<Character.Hierarchy.Length; j++) {
						Character.Hierarchy[j].SetTransformation(t[j]);
					}
					Character.Draw(
						Color.Lerp(UltiDraw.Blue, UltiDraw.Red, (float)i / (float)steps),
						UltiDraw.Yellow,
						1f - (float)i / (float)(steps+1)
						);
				}
			}

			//Future
			for(int i=1; i<=steps; i++) {
				float timestamp = CurrentFrame.Timestamp + i*timespan/steps;
				if(timestamp <= GetTotalTime()) {
					Matrix4x4[] t = ExtractPosture(GetFrame(timestamp), ShowMirrored);
					for(int j=0; j<Character.Hierarchy.Length; j++) {
						Character.Hierarchy[j].SetTransformation(t[j]);
					}
					Character.Draw(
						Color.Lerp(UltiDraw.Blue, UltiDraw.Green, (float)i / (float)steps),
						UltiDraw.Yellow,
						1f - (float)i / (float)(steps+1));
				}
			}

			if(ExportScreenshots) {
				bool export = false;
				for(int i=0; i<Sequences.Length; i++) {
					export = export || CurrentFrame.Index >= Sequences[i].Start && CurrentFrame.Index <= Sequences[i].End;
				}

				if(export) {
					if(!SkipExportScreenshot) {
						Debug.Log("Exporting screenshot " + CurrentFrame.Index);
						string folder = Application.dataPath.Substring(0, Application.dataPath.Length-6) + "Screenshots/" + name;
						bool exists = System.IO.Directory.Exists(folder);
						if(!exists) {
							System.IO.Directory.CreateDirectory(folder);
						}

						float size = 1f;
						Utility.Screenshot(
							folder + "/" + CurrentFrame.Index.ToString(),
							Screen.width/2 - Mathf.RoundToInt(Screen.width*size/2),
							Screen.height/2 - Mathf.RoundToInt(Screen.width*size/2),
							Mathf.RoundToInt(Screen.width*size),
							Mathf.RoundToInt(Screen.width*size)
							);

						if(CurrentFrame.Index == GetTotalFrames()) {
							Debug.Log("Finished exporting screenshots.");
							ExportScreenshots = false;
							SkipExportScreenshot = false;
						} else {
							SkipExportScreenshot = true;
							LoadFrame(CurrentFrame.Index + 1);
						}
					} else {
						SkipExportScreenshot = false;
					}
				} else {
					if(CurrentFrame.Index == GetTotalFrames()) {
						Debug.Log("Finished exporting screenshots.");
						ExportScreenshots = false;
						SkipExportScreenshot = false;
					} else {
						LoadFrame(CurrentFrame.Index + 1);
					}
				}
			}

			return;
		}

		if(ShowPreview) {
			UltiDraw.Begin();
			for(int i=1; i<GetTotalFrames(); i++) {
				Matrix4x4[] prevTransformations = ExtractPosture(Frames[i-1], ShowMirrored);
				Matrix4x4[] currTransformations = ExtractPosture(Frames[i], ShowMirrored);
				UltiDraw.DrawLine(prevTransformations[0].GetPosition(), currTransformations[0].GetPosition(), UltiDraw.Magenta);
			}
			UltiDraw.End();
			float step = 1f;
			for(float i=0f; i<=GetTotalTime(); i+=step) {
				Matrix4x4[] t = ExtractPosture(GetFrame(i), ShowMirrored);
				for(int j=0; j<Character.Hierarchy.Length; j++) {
					Character.Hierarchy[j].SetTransformation(t[j]);
				}
				Character.DrawSimple(UltiDraw.Grey);
			}
		}

		if(ShowTrajectory) {
			ExtractTrajectory(CurrentFrame, ShowMirrored).Draw();
		}

		if(ShowFlow) {
			//Previous postures
			for(int t=0; t<6; t++) {
				float timestamp = Mathf.Clamp(CurrentFrame.Timestamp - 1f + (float)t/6f, 0f, GetTotalTime());
				Matrix4x4[] previousPosture = ExtractPosture(GetFrame(timestamp), ShowMirrored);
				for(int j=0; j<Character.Hierarchy.Length; j++) {
					Character.Hierarchy[j].SetTransformation(previousPosture[j]);
				}
				Character.DrawSimple(Color.Lerp(UltiDraw.Blue, UltiDraw.Cyan, 1f - (float)(t+1)/6f).Transparent(0.75f));
			}
			//
		}

		//Current posture
		Matrix4x4[] posture = ShowZero ? ExtractZeroPosture(ShowMirrored) : ExtractPosture(CurrentFrame, ShowMirrored);
		for(int i=0; i<Character.Hierarchy.Length; i++) {
			Character.Hierarchy[i].SetTransformation(posture[i]);
		}
		Character.Draw();
		//

		if(ShowFlow) {
			//Future postures
			for(int t=1; t<6; t++) {
				float timestamp = Mathf.Clamp(CurrentFrame.Timestamp + (float)t/5f, 0f, GetTotalTime());
				Matrix4x4[] futurePosture = ExtractPosture(GetFrame(timestamp), ShowMirrored);
				for(int j=0; j<Character.Hierarchy.Length; j++) {
					Character.Hierarchy[j].SetTransformation(futurePosture[j]);
				}
				Character.DrawSimple(Color.Lerp(UltiDraw.Red, UltiDraw.Orange, (float)(t+1)/5f).Transparent(0.75f));
			}
			//
		}

		UltiDraw.Begin();
		BVHPhaseFunction function = ShowMirrored ? MirroredPhaseFunction : PhaseFunction;
		for(int i=0; i<function.Variables.Length; i++) {
			if(function.Variables[i]) {
				Color red = UltiDraw.Red;
				red.a = 0.25f;
				Color green = UltiDraw.Green;
				green.a = 0.25f;
				UltiDraw.DrawCircle(ShowMirrored ? posture[Symmetry[i]].GetPosition() : posture[i].GetPosition(), Character.BoneSize*1.25f, green);
				UltiDraw.DrawCircle(ShowMirrored ? posture[i].GetPosition() : posture[Symmetry[i]].GetPosition(), Character.BoneSize*1.25f, red);
			}
		}
		UltiDraw.End();
		
		if(ShowVelocities) {
			//Vector3[] velocities = ExtractVelocities(CurrentFrame, ShowMirrored, 0.1f);
			Vector3[] velocities = ExtractBoneVelocities(CurrentFrame, ShowMirrored);
			UltiDraw.Begin();
			for(int i=0; i<Character.Hierarchy.Length; i++) {
				UltiDraw.DrawArrow(
					posture[i].GetPosition(),
					posture[i].GetPosition() + velocities[i],
					0.75f,
					0.0075f,
					0.05f,
					UltiDraw.Purple.Transparent(0.5f)
				);				
			}
			UltiDraw.End();
		}
	}

	[System.Serializable]
	public class BVHSequence {
		public BVHAnimation Animation;
		public int Start = 1;
		public int End = 1;
		public int Export = 1;
		public BVHSequence(BVHAnimation animation) {
			Animation = animation;
			Start = 1;
			End = 1;
			Export = 1;
		}
		public void Auto() {
			int index = System.Array.FindIndex(Animation.Sequences, x => x == this);
			BVHSequence prev = Animation.Sequences[Mathf.Max(0, index-1)];
			BVHSequence next = Animation.Sequences[Mathf.Min(Animation.Sequences.Length-1, index+1)];
			if(prev != this) {
				Start = prev.End + 60;
			} else {
				Start = 61;
			}
			if(next != this) {
				End = next.Start - 60;
			} else {
				End = Animation.GetTotalFrames()-60;
			}
		}
		public int GetLength() {
			return End-Start+1;
		}
	}

	[System.Serializable]
	public class BVHData {
		public Bone[] Bones;
		public Motion[] Motions;

		public BVHData() {
			Bones = new Bone[0];
			Motions = new Motion[0];
		}

		public void AddBone(string name, string parent, Vector3 offset, int[] channels) {
			System.Array.Resize(ref Bones, Bones.Length+1);
			Bones[Bones.Length-1] = new Bone(name, parent, offset, channels);
		}

		public Bone FindBone(string name) {
			return System.Array.Find(Bones, x => x.Name == name);
		}

		public void AddMotion(float[] values) {
			System.Array.Resize(ref Motions, Motions.Length+1);
			Motions[Motions.Length-1] = new Motion(values);
		}

		[System.Serializable]
		public class Bone {
			public string Name;
			public string Parent;
			public Vector3 Offset;
			public int[] Channels;
			public Bone(string name, string parent, Vector3 offset, int[] channels) {
				Name = name;
				Parent = parent;
				Offset = offset;
				Channels = channels;
			}
		}

		[System.Serializable]
		public class Motion {
			public float[] Values;
			public Motion(float[] values) {
				Values = values;
			}
		}
	}

	[System.Serializable]
	public class BVHFrame {
		public BVHAnimation Animation;

		public int Index;
		public float Timestamp;

		public Matrix4x4[] Local;
		public Matrix4x4[] World;

		public BVHFrame(BVHAnimation animation, int index, float timestamp) {
			Animation = animation;
			Index = index;
			Timestamp = timestamp;

			Local = new Matrix4x4[Animation.Character.Hierarchy.Length];
			World = new Matrix4x4[Animation.Character.Hierarchy.Length];
		}

		public void Generate() {
			int channel = 0;
			BVHData.Motion motion = Animation.Data.Motions[Index-1];
			for(int i=0; i<Animation.Character.Hierarchy.Length; i++) {
				BVHData.Bone info = Animation.Data.Bones[i];
				Vector3 position = Vector3.zero;
				Quaternion rotation = Quaternion.identity;
				for(int j=0; j<info.Channels.Length; j++) {
					if(info.Channels[j] == 1) {
						position.x = motion.Values[channel]; channel += 1;
					}
					if(info.Channels[j] == 2) {
						position.y = motion.Values[channel]; channel += 1;
					}
					if(info.Channels[j] == 3) {
						position.z = motion.Values[channel]; channel += 1;
					}
					if(info.Channels[j] == 4) {
						rotation *= Quaternion.AngleAxis(motion.Values[channel], Vector3.right); channel += 1;
					}
					if(info.Channels[j] == 5) {
						rotation *= Quaternion.AngleAxis(motion.Values[channel], Vector3.up); channel += 1;
					}
					if(info.Channels[j] == 6) {
						rotation *= Quaternion.AngleAxis(motion.Values[channel], Vector3.forward); channel += 1;
					}
				}

				position = (position == Vector3.zero ? info.Offset : position) / Animation.UnitScale;
				Character.Segment parent = Animation.Character.Hierarchy[i].GetParent(Animation.Character.Hierarchy);
				Local[i] = Matrix4x4.TRS(position, rotation, Vector3.one);
				World[i] = parent == null ? Local[i] : World[parent.GetIndex()] * Local[i];
			}
			for(int i=0; i<Animation.Character.Hierarchy.Length; i++) {
				Local[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Animation.Corrections[i]), Vector3.one);
				World[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Animation.Corrections[i]), Vector3.one);
			}
		}

		/*
		public Vector3 ComputeVelocity(int index, float smoothing) {
			if(smoothing == 0f) {
				return (World[index].GetPosition() - Animation.GetFrame(Mathf.Max(1, Index-1)).World[index].GetPosition()) / Animation.FrameTime;
			}
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.GetTotalTime(), Timestamp+smoothing/2f));
			Vector3 velocity = Vector3.zero;
			for(int i=1; i<frames.Length; i++) {
				velocity += (frames[i].World[index].GetPosition() - frames[i-1].World[index].GetPosition()) / Animation.FrameTime;
			}
			velocity /= frames.Length;
			return velocity;
		}
		*/

		public Vector3 ComputeBoneVelocity(int bone) {
			return (World[bone].GetPosition() - Animation.GetFrame(Mathf.Max(1, Index-1)).World[bone].GetPosition()) / Animation.FrameTime;
		}
	}

	[System.Serializable]
	public class BVHPhaseFunction {
		public BVHAnimation Animation;

		public float[] Phase;

		public bool[] Keys;
		
		public bool ShowCycle;
		public float[] Cycle;
		public float[] NormalisedCycle;

		public Vector2 VariablesScroll;
		public bool[] Variables;

		public float VelocitySmoothing;
		public float VelocityThreshold;
		public float[] Velocities;
		public float[] NormalisedVelocities;

		public float HeightThreshold;
		public float[] Heights;

		private BVHEvolution Optimiser;
		private bool Optimising;

		public BVHPhaseFunction(BVHAnimation animation) {
			Animation = animation;

			Phase = new float[Animation.GetTotalFrames()];

			Keys = new bool[Animation.GetTotalFrames()];

			Cycle = new float[Animation.GetTotalFrames()];
			NormalisedCycle = new float[Animation.GetTotalFrames()];

			Variables = new bool[Animation.Character.Hierarchy.Length];

			Velocities = new float[Animation.GetTotalFrames()];
			NormalisedVelocities = new float[Animation.GetTotalFrames()];

			Heights = new float[Animation.GetTotalFrames()];
		}

		public void Initialise() {
			Keys[0] = true;
			Keys[Animation.GetTotalFrames()-1] = true;
			Phase[0] = 0f;
			Phase[Animation.GetTotalFrames()-1] = 1f;
			VelocitySmoothing = 0f;
			VelocityThreshold = 0.1f;
			HeightThreshold = 0f;
			ComputeValues();
		}

		public void SetKey(BVHFrame frame, bool value) {
			if(value) {
				if(IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = true;
				Phase[frame.Index-1] = 1f;
				Interpolate(frame);
			} else {
				if(!IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = false;
				Phase[frame.Index-1] = 0f;
				Interpolate(frame);
			}
		}

		public bool IsKey(BVHFrame frame) {
			return Keys[frame.Index-1];
		}

		public void SetPhase(BVHFrame frame, float value) {
			if(Phase[frame.Index-1] != value) {
				Phase[frame.Index-1] = value;
				Interpolate(frame);
			}
		}

		public float GetPhase(BVHFrame frame) {
			return Phase[frame.Index-1];
		}

		public void SetVelocitySmoothing(float value) {
			value = Mathf.Max(0f, value);
			if(VelocitySmoothing != value) {
				Animation.PhaseFunction.VelocitySmoothing = value;
				Animation.MirroredPhaseFunction.VelocitySmoothing = value;
				Animation.PhaseFunction.ComputeValues();
				Animation.MirroredPhaseFunction.ComputeValues();
			}
		}

		public void SetVelocityThreshold(float value) {
			value = Mathf.Max(0f, value);
			if(VelocityThreshold != value) {
				Animation.PhaseFunction.VelocityThreshold = value;
				Animation.MirroredPhaseFunction.VelocityThreshold = value;
				Animation.PhaseFunction.ComputeValues();
				Animation.MirroredPhaseFunction.ComputeValues();
			}
		}

		public void SetHeightThreshold(float value) {
			value = Mathf.Max(0f, value);
			if(HeightThreshold != value) {
				Animation.PhaseFunction.HeightThreshold = value;
				Animation.MirroredPhaseFunction.HeightThreshold = value;
				Animation.PhaseFunction.ComputeValues();
				Animation.MirroredPhaseFunction.ComputeValues();
			}
		}

		public void ToggleVariable(int index) {
			Variables[index] = !Variables[index];
			if(Animation.ShowMirrored) {
				for(int i=0; i<Animation.PhaseFunction.Variables.Length; i++) {
					Animation.PhaseFunction.Variables[Animation.Symmetry[index]] = Variables[index];
				}
			} else {
				for(int i=0; i<Animation.MirroredPhaseFunction.Variables.Length; i++) {
					Animation.MirroredPhaseFunction.Variables[Animation.Symmetry[index]] = Variables[index];
				}
			}
			Animation.PhaseFunction.ComputeValues();
			Animation.MirroredPhaseFunction.ComputeValues();
		}

		public BVHFrame GetPreviousKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index-1; i>=1; i--) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return Animation.Frames[0];
		}

		public BVHFrame GetNextKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index+1; i<=Animation.GetTotalFrames(); i++) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return Animation.Frames[Animation.GetTotalFrames()-1];
		}

		public void Recompute() {
			for(int i=0; i<Animation.Frames.Length; i++) {
				if(IsKey(Animation.Frames[i])) {
					Phase[i] = 1f;
				}
			}
			BVHFrame A = Animation.Frames[0];
			BVHFrame B = GetNextKey(A);
			while(A != B) {
				Interpolate(A, B);
				A = B;
				B = GetNextKey(A);
			}
		}

		private void Interpolate(BVHFrame frame) {
			if(IsKey(frame)) {
				Interpolate(GetPreviousKey(frame), frame);
				Interpolate(frame, GetNextKey(frame));
			} else {
				Interpolate(GetPreviousKey(frame), GetNextKey(frame));
			}
		}

		private void Interpolate(BVHFrame a, BVHFrame b) {
			if(a == null || b == null) {
				Debug.Log("A given frame was null.");
				return;
			}
			int dist = b.Index - a.Index;
			if(dist >= 2) {
				for(int i=a.Index+1; i<b.Index; i++) {
					float rateA = (float)((float)i-(float)a.Index)/(float)dist;
					float rateB = (float)((float)b.Index-(float)i)/(float)dist;
					Phase[i-1] = rateB*Mathf.Repeat(Phase[a.Index-1], 1f) + rateA*Phase[b.Index-1];
				}
			}

			if(a.Index == 1) {
				BVHFrame first = Animation.Frames[0];
				BVHFrame next1 = GetNextKey(first);
				BVHFrame next2 = GetNextKey(next1);
				Keys[0] = true;
				float xFirst = next1.Timestamp - first.Timestamp;
				float mFirst = next2.Timestamp - next1.Timestamp;
				SetPhase(first, Mathf.Clamp(1f - xFirst / mFirst, 0f, 1f));
			}
			if(b.Index == Animation.GetTotalFrames()) {
				BVHFrame last = Animation.Frames[Animation.GetTotalFrames()-1];
				BVHFrame previous1 = GetPreviousKey(last);
				BVHFrame previous2 = GetPreviousKey(previous1);
				Keys[Animation.GetTotalFrames()-1] = true;
				float xLast = last.Timestamp - previous1.Timestamp;
				float mLast = previous1.Timestamp - previous2.Timestamp;
				SetPhase(last, Mathf.Clamp(xLast / mLast, 0f, 1f));
			}
		}

		public void ComputeValues() {
			for(int i=0; i<Animation.GetTotalFrames(); i++) {
				Heights[i] = 0f;
				Velocities[i] = 0f;
				NormalisedVelocities[i] = 0f;
			}
			float min, max;
			
			LayerMask mask = LayerMask.GetMask("Ground");
			min = float.MaxValue;
			max = float.MinValue;
			for(int i=0; i<Animation.GetTotalFrames(); i++) {
				for(int j=0; j<Animation.Character.Hierarchy.Length; j++) {
					if(Variables[j]) {
						float offset = Mathf.Max(0f, Animation.Frames[i].World[j].GetPosition().y - Utility.ProjectGround(Animation.Frames[i].World[j].GetPosition(), mask).y);
						Heights[i] += offset < HeightThreshold ? 0f : offset;
					}
				}
				if(Heights[i] < min) {
					min = Heights[i];
				}
				if(Heights[i] > max) {
					max = Heights[i];
				}
			}
			for(int i=0; i<Heights.Length; i++) {
				Heights[i] = Utility.Normalise(Heights[i], min, max, 0f, 1f);
			}

			min = float.MaxValue;
			max = float.MinValue;
			for(int i=0; i<Animation.GetTotalFrames(); i++) {
				for(int j=0; j<Animation.Character.Hierarchy.Length; j++) {
					if(Variables[j]) {
						//float boneVelocity = Animation.Frames[i].ComputeVelocity(j, VelocitySmoothing).magnitude;
						float boneVelocity = Animation.Frames[i].ComputeBoneVelocity(j).magnitude;
						Velocities[i] += boneVelocity;
					}
				}
				if(Velocities[i] < VelocityThreshold || Heights[i] == 0f) {
					Velocities[i] = 0f;
				}
				if(Velocities[i] < min) {
					min = Velocities[i];
				}
				if(Velocities[i] > max) {
					max = Velocities[i];
				}
			}
			for(int i=0; i<Velocities.Length; i++) {
				NormalisedVelocities[i] = Utility.Normalise(Velocities[i], min, max, 0f, 1f);
			}
		}

		public void EditorUpdate() {
			if(Optimising) {
				if(Cycle == null) {
					Cycle = new float[Animation.GetTotalFrames()];
				} else if(Cycle.Length != Animation.GetTotalFrames()) {
					Cycle = new float[Animation.GetTotalFrames()];
				}
				if(Optimiser == null) {
					Optimiser = new BVHEvolution(Animation, this);
				}
				Optimiser.Optimise();
			}
		}

		public void Inspector() {
			UltiDraw.Begin();

			Utility.SetGUIColor(UltiDraw.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Phase Function");
				}

				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Optimising) {
						if(Utility.GUIButton("Stop Optimisation", UltiDraw.LightGrey, UltiDraw.Black)) {
							Optimising = !Optimising;
						}
					} else {
						if(Utility.GUIButton("Start Optimisation", UltiDraw.DarkGrey, UltiDraw.White)) {
							Optimising = !Optimising;
						}
					}
					if(Optimiser != null) {
						if(Utility.GUIButton("Restart", UltiDraw.Brown, UltiDraw.White)) {
							Optimiser.Initialise();
						}
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Fitness: " + Optimiser.GetFitness(), GUILayout.Width(150f));
						float[] configuration = Optimiser.GetPeakConfiguration();
						EditorGUILayout.LabelField("Peak: " + configuration[0] + " | " + configuration[1] + " | " + configuration[2] + " | " + configuration[3] + " | " + configuration[4]);
						EditorGUILayout.EndHorizontal();
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Exploration", GUILayout.Width(100f));
						GUILayout.FlexibleSpace();
						Optimiser.Behaviour = EditorGUILayout.Slider(Optimiser.Behaviour, 0f, 1f);
						GUILayout.FlexibleSpace();
						EditorGUILayout.LabelField("Exploitation", GUILayout.Width(100f));
						EditorGUILayout.EndHorizontal();
						Optimiser.SetAmplitude(EditorGUILayout.Slider("Amplitude", Optimiser.Amplitude, 0, BVHEvolution.AMPLITUDE));
						Optimiser.SetFrequency(EditorGUILayout.Slider("Frequency", Optimiser.Frequency, 0f, BVHEvolution.FREQUENCY));
						Optimiser.SetShift(EditorGUILayout.Slider("Shift", Optimiser.Shift, 0, BVHEvolution.SHIFT));
						Optimiser.SetOffset(EditorGUILayout.Slider("Offset", Optimiser.Offset, 0, BVHEvolution.OFFSET));
						Optimiser.SetSlope(EditorGUILayout.Slider("Slope", Optimiser.Slope, 0, BVHEvolution.SLOPE));
						Optimiser.SetWindow(EditorGUILayout.Slider("Window", Optimiser.Window, 0.1f, BVHEvolution.WINDOW));
						Optimiser.Blending = EditorGUILayout.Slider("Blending", Optimiser.Blending, 0f, 1f);
					} else {
						EditorGUILayout.LabelField("No optimiser available.");
					}
				}

				VariablesScroll = EditorGUILayout.BeginScrollView(VariablesScroll, GUILayout.Height(100f));
				for(int i=0; i<Animation.Character.Hierarchy.Length; i++) {
					if(Variables[i]) {
						if(Utility.GUIButton(Animation.Character.Hierarchy[i].GetName(), UltiDraw.DarkGreen, UltiDraw.White)) {
							ToggleVariable(i);
						}
					} else {
						if(Utility.GUIButton(Animation.Character.Hierarchy[i].GetName(), UltiDraw.DarkRed, UltiDraw.White)) {
							ToggleVariable(i);
						}
					}
				}
				EditorGUILayout.EndScrollView();

				SetVelocitySmoothing(EditorGUILayout.FloatField("Velocity Smoothing", VelocitySmoothing));
				SetVelocityThreshold(EditorGUILayout.FloatField("Velocity Threshold", VelocityThreshold));
				SetHeightThreshold(EditorGUILayout.FloatField("Height Threshold", HeightThreshold));

				if(IsKey(Animation.CurrentFrame)) {
					SetPhase(Animation.CurrentFrame, EditorGUILayout.Slider("Phase", GetPhase(Animation.CurrentFrame), 0f, 1f));
				} else {
					EditorGUI.BeginDisabledGroup(true);
					SetPhase(Animation.CurrentFrame, EditorGUILayout.Slider("Phase", GetPhase(Animation.CurrentFrame), 0f, 1f));
					EditorGUI.EndDisabledGroup();
				}

				ShowCycle = EditorGUILayout.Toggle("Show Cycle", ShowCycle);

				if(IsKey(Animation.CurrentFrame)) {
					if(Utility.GUIButton("Unset Key", UltiDraw.Grey, UltiDraw.White)) {
						SetKey(Animation.CurrentFrame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", UltiDraw.DarkGrey, UltiDraw.White)) {
						SetKey(Animation.CurrentFrame, true);
					}
				}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = Animation.CurrentFrame.Timestamp-Animation.TimeWindow/2f;
				float endTime = Animation.CurrentFrame.Timestamp+Animation.TimeWindow/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > Animation.GetTotalTime()) {
					startTime -= endTime-Animation.GetTotalTime();
					endTime = Animation.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(Animation.GetTotalTime(), endTime);
				int start = Animation.GetFrame(startTime).Index;
				int end = Animation.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 prevPos = Vector3.zero;
				Vector3 newPos = Vector3.zero;
				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

				//Velocities
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Animation.PhaseFunction.NormalisedVelocities[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Animation.PhaseFunction.NormalisedVelocities[i+start] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, this == Animation.PhaseFunction ? UltiDraw.Green : UltiDraw.Red);
				}

				//Mirrored Velocities
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Animation.MirroredPhaseFunction.NormalisedVelocities[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Animation.MirroredPhaseFunction.NormalisedVelocities[i+start] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, this == Animation.PhaseFunction ? UltiDraw.Red : UltiDraw.Green);
				}

				//Heights
				/*
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Heights[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Heights[i+start] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Red);
				}
				*/
				
				//Cycle
				if(ShowCycle) {
					for(int i=1; i<elements; i++) {
						prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
						prevPos.y = rect.yMax - NormalisedCycle[i+start-1] * rect.height;
						newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
						newPos.y = rect.yMax - NormalisedCycle[i+start] * rect.height;
						UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Yellow);
					}
				}

				//Phase
				/*
				for(int i=1; i<Animation.Frames.Length; i++) {
					BVHFrame A = Animation.Frames[i-1];
					BVHFrame B = Animation.Frames[i];
					prevPos.x = rect.xMin + (float)(A.Index-start)/elements * rect.width;
					prevPos.y = rect.yMax - Mathf.Repeat(Phase[A.Index-1], 1f) * rect.height;
					newPos.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					newPos.y = rect.yMax - Phase[B.Index-1] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
				}
				*/
				BVHFrame A = Animation.GetFrame(start);
				if(A.Index == 1) {
					bottom.x = rect.xMin;
					top.x = rect.xMin;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta);
				}
				BVHFrame B = GetNextKey(A);
				while(A != B) {
					prevPos.x = rect.xMin + (float)(A.Index-start)/elements * rect.width;
					prevPos.y = rect.yMax - Mathf.Repeat(Phase[A.Index-1], 1f) * rect.height;
					newPos.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					newPos.y = rect.yMax - Phase[B.Index-1] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta);
					A = B;
					B = GetNextKey(A);
					if(B.Index > end) {
						break;
					}
				}

				//Seconds
				float timestamp = startTime;
				while(timestamp <= endTime) {
					float floor = Mathf.FloorToInt(timestamp);
					if(floor >= startTime && floor <= endTime) {
						top.x = rect.xMin + (float)(Animation.GetFrame(floor).Index-start)/elements * rect.width;
						UltiDraw.DrawCircle(top, 5f, UltiDraw.White);
					}
					timestamp += 1f;
				}
				//

				//Sequences
				for(int i=0; i<Animation.Sequences.Length; i++) {
					top.x = rect.xMin + (float)(Animation.Sequences[i].Start-start)/elements * rect.width;
					bottom.x = rect.xMin + (float)(Animation.Sequences[i].Start-start)/elements * rect.width;
					Vector3 a = top;
					Vector3 b = bottom;
					top.x = rect.xMin + (float)(Animation.Sequences[i].End-start)/elements * rect.width;
					bottom.x = rect.xMin + (float)(Animation.Sequences[i].End-start)/elements * rect.width;
					Vector3 c = top;
					Vector3 d = bottom;

					Color yellow = UltiDraw.Yellow;
					yellow.a = 0.25f;
					UltiDraw.DrawTriangle(a, b, c, yellow);
					UltiDraw.DrawTriangle(b, d, c, yellow);
				}

				//Current Pivot
				top.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
				UltiDraw.DrawCircle(top, 3f, UltiDraw.Green);
				UltiDraw.DrawCircle(bottom, 3f, UltiDraw.Green);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();
			}

			UltiDraw.End();
		}
	}

	[System.Serializable]
	public class BVHStyleFunction {
		public BVHAnimation Animation;
		
		public enum STYLE {Custom, Biped, Quadruped, Count}
		public STYLE Style = STYLE.Custom;

		public float Transition;
		public bool[] Keys;
		public BVHStyle[] Styles;

		public BVHStyleFunction(BVHAnimation animation) {
			Animation = animation;
			Reset();
		}

		public void Reset() {
			Transition = 0.1f;
			Keys = new bool[Animation.GetTotalFrames()];
			Styles = new BVHStyle[0];
		}

		public void SetStyle(STYLE style) {
			if(Style != style) {
				Style = style;
				Reset();
				switch(Style) {
					case STYLE.Custom:
					break;

					case STYLE.Biped:
					AddStyle("Idle");
					AddStyle("Walk");
					AddStyle("Run");
					AddStyle("Crouch");
					AddStyle("Jump");
					break;

					case STYLE.Quadruped:
					AddStyle("Idle");
					AddStyle("Move");
					AddStyle("Jump");
					AddStyle("Sit");
					AddStyle("Lie");
					AddStyle("Stand");
					break;
				}
			}
		}

		public void AddStyle(string name = "Style") {
			ArrayExtensions.Add(ref Styles, new BVHStyle(name, Animation.GetTotalFrames()));
		}

		public void RemoveStyle() {
			ArrayExtensions.Shrink(ref Styles);
		}

		public void RemoveStyle(string name) {
			ArrayExtensions.RemoveAt(ref Styles, System.Array.FindIndex(Styles, x => x.Name == name));
		}

		public BVHStyle GetStyle(string name) {
			return System.Array.Find(Styles, x => x.Name == name);
		}

		public void SetFlag(BVHFrame frame, int dimension, bool value) {
			if(GetFlag(frame, dimension) == value) {
				return;
			}
			Styles[dimension].Flags[frame.Index-1] = value;
			Interpolate(frame, dimension);
		}

		public bool GetFlag(BVHFrame frame, int dimension) {
			return Styles[dimension].Flags[frame.Index-1];
		}

		public void SetKey(BVHFrame frame, bool value) {
			if(value) {
				if(IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = true;
				Recompute();
			} else {
				if(!IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = false;
				Recompute();
			}
		}

		public bool IsKey(BVHFrame frame) {
			return Keys[frame.Index-1];
		}

		public BVHFrame GetPreviousKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index-1; i>=1; i--) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return Animation.Frames[0];
		}

		public BVHFrame GetNextKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index+1; i<=Animation.GetTotalFrames(); i++) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return Animation.Frames[Animation.GetTotalFrames()-1];
		}

		public void Recompute() {
			for(int i=0; i<Animation.GetTotalFrames(); i++) {
				if(Keys[i]) {
					for(int j=0; j<Styles.Length; j++) {
						Interpolate(Animation.Frames[i], j);
					}
				}
			}
		}

		public void SetTransition(float value) {
			value = Mathf.Max(value, 0f);
			if(Transition == value) {
				return;
			}
			Transition = value;
			Recompute();
		}

		private void Interpolate(BVHFrame frame, int dimension) {
			BVHFrame prev = GetPreviousKey(frame);
			BVHFrame next = GetNextKey(frame);
			Styles[dimension].Values[frame.Index-1] = GetFlag(frame, dimension) ? 1f : 0f;
			if(IsKey(frame)) {
				MakeConstant(dimension, prev, frame);
				MakeConstant(dimension, frame, next);
			} else {
				MakeConstant(dimension, prev, next);
			}
			MakeTransition(dimension, prev);
			MakeTransition(dimension, frame);
			MakeTransition(dimension, next);
		}

		private void MakeConstant(int dimension, BVHFrame previous, BVHFrame next) {
			int start = previous == null ? 1 : previous.Index;
			int end = next == null ? Animation.GetTotalFrames() : next.Index-1;
			//int start = previous.Index;
			//int end = next.Index;
			for(int i=start; i<end; i++) {
				Styles[dimension].Flags[i] = Styles[dimension].Flags[start-1];
				Styles[dimension].Values[i] = Styles[dimension].Flags[start-1] ? 1f : 0f;
			}
		}

		private void MakeTransition(int dimension, BVHFrame frame) {
			if(frame == null) {
				return;
			}
			//Get window
			float window = GetWindow(frame);
			
			//Interpolate
			BVHFrame a = Animation.GetFrame(frame.Timestamp - 0.5f*window);
			BVHFrame b = Animation.GetFrame(frame.Timestamp + 0.5f*window);
			int dist = b.Index - a.Index;
			if(dist >= 2) {
				for(int i=a.Index+1; i<b.Index; i++) {
					float rateA = (float)((float)i-(float)a.Index)/(float)dist;
					float rateB = (float)((float)b.Index-(float)i)/(float)dist;
					rateA = rateA * rateA;
					rateB = rateB * rateB;
					float valueA = Styles[dimension].Flags[a.Index-1] ? 1f : 0f;
					float valueB = Styles[dimension].Flags[b.Index-1] ? 1f : 0f;
					Styles[dimension].Values[i-1] = rateB/(rateA+rateB)*valueA + rateA/(rateA+rateB)*valueB;
				}
			}
		}

		public float GetWindow(BVHFrame frame) {
			BVHFrame prev = GetPreviousKey(frame);
			BVHFrame next = GetNextKey(frame);
			float prevTS = prev == null ? 0f : prev.Timestamp;
			float nextTS = next == null ? Animation.GetTotalTime() : next.Timestamp;
			float prevTime = Mathf.Abs(frame.Timestamp - prevTS);
			float nextTime = Mathf.Abs(frame.Timestamp - nextTS);
			float window = Mathf.Min(prevTime, nextTime, Transition);
			return window;
		}

		public void Inspector() {
			UltiDraw.Begin();

			Utility.SetGUIColor(UltiDraw.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Style Function");
				}

				SetTransition(EditorGUILayout.FloatField("Transition", Transition));

				string[] names = new string[(int)STYLE.Count];
				for(int i=0; i<names.Length; i++) {
					names[i] = ((STYLE)i).ToString();
				}
				SetStyle((STYLE)EditorGUILayout.Popup((int)Style, names));

				for(int i=0; i<Styles.Length; i++) {
					EditorGUILayout.BeginHorizontal();
					Styles[i].Name = EditorGUILayout.TextField(Styles[i].Name, GUILayout.Width(75f));
					if(IsKey(Animation.CurrentFrame)) {
						if(GetFlag(Animation.CurrentFrame, i)) {
							if(Utility.GUIButton("On", UltiDraw.DarkGreen, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, false);
							}
						} else {
							if(Utility.GUIButton("Off", UltiDraw.DarkRed, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, true);
							}
						}
					} else {
						EditorGUI.BeginDisabledGroup(true);
						if(GetFlag(Animation.CurrentFrame, i)) {
							if(Utility.GUIButton("On", UltiDraw.DarkGreen, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, false);
							}
						} else {
							if(Utility.GUIButton("Off", UltiDraw.DarkRed, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, true);
							}
						}
						EditorGUI.EndDisabledGroup();
					}
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.Slider(Styles[i].Values[Animation.CurrentFrame.Index-1], 0f, 1f);
					EditorGUI.EndDisabledGroup();
					EditorGUILayout.EndHorizontal();
				}
				
				if(Style == STYLE.Custom) {
				EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Add Style", UltiDraw.DarkGrey, UltiDraw.White)) {
						AddStyle();
					}
					if(Utility.GUIButton("Remove Style", UltiDraw.DarkGrey, UltiDraw.White)) {
						RemoveStyle();
					}
					EditorGUILayout.EndHorizontal();
				}

				if(IsKey(Animation.CurrentFrame)) {
					if(Utility.GUIButton("Unset Key", UltiDraw.Grey, UltiDraw.White)) {
						SetKey(Animation.CurrentFrame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", UltiDraw.DarkGrey, UltiDraw.White)) {
						SetKey(Animation.CurrentFrame, true);
					}
				}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = Animation.CurrentFrame.Timestamp-Animation.TimeWindow/2f;
				float endTime = Animation.CurrentFrame.Timestamp+Animation.TimeWindow/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > Animation.GetTotalTime()) {
					startTime -= (endTime-Animation.GetTotalTime());
					endTime = Animation.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(Animation.GetTotalTime(), endTime);
				int start = Animation.GetFrame(startTime).Index;
				int end = Animation.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 prevPos = Vector3.zero;
				Vector3 newPos = Vector3.zero;
				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);
				
				Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);
				BVHFrame A = Animation.GetFrame(start);
				if(IsKey(A)) {
					bottom.x = rect.xMin;
					top.x = rect.xMin;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta);
				}
				
				BVHFrame B = GetNextKey(A);
				while(A != B && A != null && B != null) {
					float window = GetWindow(B);
					BVHFrame left = Animation.GetFrame(B.Timestamp - window/2f);
					BVHFrame right = Animation.GetFrame(B.Timestamp + window/2f);
					for(int f=left.Index; f<right.Index; f++) {
						prevPos.x = rect.xMin + (float)(f-start)/elements * rect.width;
						newPos.x = rect.xMin + (float)(f+1-start)/elements * rect.width;
						for(int i=0; i<Styles.Length; i++) {
							prevPos.y = rect.yMax - Styles[i].Values[f-1] * rect.height;
							newPos.y = rect.yMax - Styles[i].Values[f] * rect.height;
							UltiDraw.DrawLine(prevPos, newPos, colors[i]);
						}
					}
					
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta);
					
					A = B;
					B = GetNextKey(A);
					if(B == null) {
						break;
					}
					if(B.Index > end) {
						break;
					}
				}

				//Seconds
				float timestamp = startTime;
				while(timestamp <= endTime) {
					float floor = Mathf.FloorToInt(timestamp);
					if(floor >= startTime && floor <= endTime) {
						top.x = rect.xMin + (float)(Animation.GetFrame(floor).Index-start)/elements * rect.width;
						UltiDraw.DrawCircle(top, 5f, UltiDraw.White);
					}
					timestamp += 1f;
				}
				//

				//Sequences
				for(int i=0; i<Animation.Sequences.Length; i++) {
					top.x = rect.xMin + (float)(Animation.Sequences[i].Start-start)/elements * rect.width;
					bottom.x = rect.xMin + (float)(Animation.Sequences[i].Start-start)/elements * rect.width;
					Vector3 a = top;
					Vector3 b = bottom;
					top.x = rect.xMin + (float)(Animation.Sequences[i].End-start)/elements * rect.width;
					bottom.x = rect.xMin + (float)(Animation.Sequences[i].End-start)/elements * rect.width;
					Vector3 c = top;
					Vector3 d = bottom;

					Color yellow = UltiDraw.Yellow;
					yellow.a = 0.25f;
					UltiDraw.DrawTriangle(a, b, c, yellow);
					UltiDraw.DrawTriangle(b, d, c, yellow);
				}

				//Current Pivot
				top.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
				UltiDraw.DrawCircle(top, 3f, UltiDraw.Green);
				UltiDraw.DrawCircle(bottom, 3f, UltiDraw.Green);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();

			}

			UltiDraw.End();
		}
	}

	[System.Serializable]
	public class BVHStyle {
		public string Name;
		public bool[] Flags;
		public float[] Values;

		public BVHStyle(string name, int length) {
			Name = name;
			Flags = new bool[length];
			Values = new float[length];
		}
	}

	public class BVHEvolution {
		public static float AMPLITUDE = 10f;
		public static float FREQUENCY = 2.5f;
		public static float SHIFT = Mathf.PI;
		public static float OFFSET = 10f;
		public static float SLOPE = 5f;
		public static float WINDOW = 5f;
		
		public BVHAnimation Animation;
		public BVHPhaseFunction Function;

		public Population[] Populations;

		public float[] LowerBounds;
		public float[] UpperBounds;

		public float Amplitude = AMPLITUDE;
		public float Frequency = FREQUENCY;
		public float Shift = SHIFT;
		public float Offset = OFFSET;
		public float Slope = SLOPE;

		public float Behaviour = 1f;

		public float Window = 1f;
		public float Blending = 1f;

		public BVHEvolution(BVHAnimation animation, BVHPhaseFunction function) {
			Animation = animation;
			Function = function;

			LowerBounds = new float[5];
			UpperBounds = new float[5];

			SetAmplitude(Amplitude);
			SetFrequency(Frequency);
			SetShift(Shift);
			SetOffset(Offset);
			SetSlope(Slope);

			Initialise();
		}

		public void SetAmplitude(float value) {
			Amplitude = value;
			LowerBounds[0] = -value;
			UpperBounds[0] = value;
		}

		public void SetFrequency(float value) {
			Frequency = value;
			LowerBounds[1] = 0f;
			UpperBounds[1] = value;
		}

		public void SetShift(float value) {
			Shift = value;
			LowerBounds[2] = -value;
			UpperBounds[2] = value;
		}

		public void SetOffset(float value) {
			Offset = value;
			LowerBounds[3] = -value;
			UpperBounds[3] = value;
		}

		public void SetSlope(float value) {
			Slope = value;
			LowerBounds[4] = -value;
			UpperBounds[4] = value;
		}

		public void SetWindow(float value) {
			if(Window != value) {
				Window = value;
				Initialise();
			}
		}

		public void Initialise() {
			Interval[] intervals = new Interval[Mathf.FloorToInt(Animation.GetTotalTime() / Window) + 1];
			for(int i=0; i<intervals.Length; i++) {
				int start = Animation.GetFrame(i*Window).Index-1;
				int end = Animation.GetFrame(Mathf.Min(Animation.GetTotalTime(), (i+1)*Window)).Index-2;
				if(end == Animation.GetTotalFrames()-2) {
					end += 1;
				}
				intervals[i] = new Interval(start, end);
			}
			Populations = new Population[intervals.Length];
			for(int i=0; i<Populations.Length; i++) {
				Populations[i] = new Population(this, 50, 5, intervals[i]);
			}
			Assign();
		}

		public void Optimise() {
			for(int i=0; i<Populations.Length; i++) {
				Populations[i].Active = IsActive(i);
			}
			for(int i=0; i<Populations.Length; i++) {
				Populations[i].Evolve(GetPreviousPopulation(i), GetNextPopulation(i), GetPreviousPivotPopulation(i), GetNextPivotPopulation(i));
			}
			Assign();
		}

		
		public void Assign() {
			for(int i=0; i<Animation.GetTotalFrames(); i++) {
				Function.Keys[i] = false;
				Function.Phase[i] = 0f;
				Function.Cycle[i] = 0f;
				Function.NormalisedCycle[i] = 0f;
			}

			//Compute cycle
			float min = float.MaxValue;
			float max = float.MinValue;
			for(int i=0; i<Populations.Length; i++) {
				for(int j=Populations[i].Interval.Start; j<=Populations[i].Interval.End; j++) {
					Function.Cycle[j] = Interpolate(i, j);
					min = Mathf.Min(min, Function.Cycle[j]);
					max = Mathf.Max(max, Function.Cycle[j]);
				}
			}
			for(int i=0; i<Populations.Length; i++) {
				for(int j=Populations[i].Interval.Start; j<=Populations[i].Interval.End; j++) {
					Function.NormalisedCycle[j] = Utility.Normalise(Function.Cycle[j], min, max, 0f, 1f);
				}
			}

			//Fill with frequency negative turning points
			for(int i=0; i<Populations.Length; i++) {
				for(int j=Populations[i].Interval.Start; j<=Populations[i].Interval.End; j++) {
					if(InterpolateD2(i, j) <= 0f && InterpolateD2(i, j+1) >= 0f) {
						Function.Keys[j] = true;
					}
				}
			}

			//Compute phase
			for(int i=0; i<Function.Keys.Length; i++) {
				if(Function.Keys[i]) {
					Function.SetPhase(Animation.Frames[i], i == 0 ? 0f : 1f);
				}
			}
		}

		public Population GetPreviousPopulation(int current) {
			return Populations[Mathf.Max(0, current-1)];
		}

		public Population GetPreviousPivotPopulation(int current) {
			for(int i=current-1; i>=0; i--) {
				if(Populations[i].Active) {
					return Populations[i];
				}
			}
			return Populations[0];
		}

		public Population GetNextPopulation(int current) {
			return Populations[Mathf.Min(Populations.Length-1, current+1)];
		}

		public Population GetNextPivotPopulation(int current) {
			for(int i=current+1; i<Populations.Length; i++) {
				if(Populations[i].Active) {
					return Populations[i];
				}
			}
			return Populations[Populations.Length-1];
		}

		public bool IsActive(int interval) {
			float velocity = 0f;
			for(int i=Populations[interval].Interval.Start; i<=Populations[interval].Interval.End; i++) {
				velocity += Function.Velocities[i];
				velocity += Function == Animation.PhaseFunction ? Animation.MirroredPhaseFunction.Velocities[i] : Animation.PhaseFunction.Velocities[i];
			}
			return velocity / Populations[interval].Interval.Length > 0f;
		}

		public float Interpolate(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float InterpolateD1(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype1(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype1(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype1(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float InterpolateD2(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype2(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype2(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype2(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float InterpolateD3(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype3(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype3(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype3(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float GetFitness() {
			float fitness = 0f;
			for(int i=0; i<Populations.Length; i++) {
				fitness += Populations[i].GetFitness();
			}
			return fitness / Populations.Length;
		}

		public float[] GetPeakConfiguration() {
			float[] configuration = new float[5];
			for(int i=0; i<5; i++) {
				configuration[i] = float.MinValue;
			}
			for(int i=0; i<Populations.Length; i++) {
				for(int j=0; j<5; j++) {
					configuration[j] = Mathf.Max(configuration[j], Mathf.Abs(Populations[i].GetWinner().Genes[j]));
				}
			}
			return configuration;
		}

		public class Population {
			public BVHEvolution Evolution;
			public int Size;
			public int Dimensionality;
			public Interval Interval;

			public bool Active;

			public Individual[] Individuals;
			public Individual[] Offspring;
			public float[] RankProbabilities;
			public float RankProbabilitySum;

			public Population(BVHEvolution evolution, int size, int dimensionality, Interval interval) {
				Evolution = evolution;
				Size = size;
				Dimensionality = dimensionality;
				Interval = interval;

				//Create individuals
				Individuals = new Individual[Size];
				Offspring = new Individual[Size];
				for(int i=0; i<Size; i++) {
					Individuals[i] = new Individual(Dimensionality);
					Offspring[i] = new Individual(Dimensionality);
				}

				//Compute rank probabilities
				RankProbabilities = new float[Size];
				float rankSum = (float)(Size*(Size+1)) / 2f;
				for(int i=0; i<Size; i++) {
					RankProbabilities[i] = (float)(Size-i)/(float)rankSum;
				}
				for(int i=0; i<Size; i++) {
					RankProbabilitySum += RankProbabilities[i];
				}

				//Initialise randomly
				for(int i=0; i<Size; i++) {
					Reroll(Individuals[i]);
				}

				//Evaluate fitness
				for(int i=0; i<Size; i++) {
					Individuals[i].Fitness = ComputeFitness(Individuals[i].Genes);
				}

				//Sort
				SortByFitness(Individuals);

				//Evaluate extinctions
				AssignExtinctions(Individuals);
			}

			public void Evolve(Population previous, Population next, Population previousPivot, Population nextPivot) {
				if(Active) {
					//Copy elite
					Copy(Individuals[0], Offspring[0]);

					//Memetic exploitation
					Exploit(Offspring[0]);

					//Remaining individuals
					for(int o=1; o<Size; o++) {
						Individual offspring = Offspring[o];
						if(Random.value <= Evolution.Behaviour) {
							Individual parentA = Select(Individuals);
							Individual parentB = Select(Individuals);
							while(parentB == parentA) {
								parentB = Select(Individuals);
							}
							Individual prototype = Select(Individuals);
							while(prototype == parentA || prototype == parentB) {
								prototype = Select(Individuals);
							}

							float mutationRate = GetMutationProbability(parentA, parentB);
							float mutationStrength = GetMutationStrength(parentA, parentB);

							for(int i=0; i<Dimensionality; i++) {
								float weight;

								//Recombination
								weight = Random.value;
								float momentum = Random.value * parentA.Momentum[i] + Random.value * parentB.Momentum[i];
								if(Random.value < 0.5f) {
									offspring.Genes[i] = parentA.Genes[i] + momentum;
								} else {
									offspring.Genes[i] = parentB.Genes[i] + momentum;
								}

								//Store
								float gene = offspring.Genes[i];

								//Mutation
								if(Random.value <= mutationRate) {
									float span = Evolution.UpperBounds[i] - Evolution.LowerBounds[i];
									offspring.Genes[i] += Random.Range(-mutationStrength*span, mutationStrength*span);
								}
								
								//Adoption
								weight = Random.value;
								offspring.Genes[i] += 
									weight * Random.value * (0.5f * (parentA.Genes[i] + parentB.Genes[i]) - offspring.Genes[i])
									+ (1f-weight) * Random.value * (prototype.Genes[i] - offspring.Genes[i]);

								//Constrain
								offspring.Genes[i] = Mathf.Clamp(offspring.Genes[i], Evolution.LowerBounds[i], Evolution.UpperBounds[i]);

								//Momentum
								offspring.Momentum[i] = Random.value * momentum + (offspring.Genes[i] - gene);
							}
						} else {
							Reroll(offspring);
						}
					}

					//Evaluate fitness
					for(int i=0; i<Size; i++) {
						Offspring[i].Fitness = ComputeFitness(Offspring[i].Genes);
					}

					//Sort
					SortByFitness(Offspring);

					//Evaluate extinctions
					AssignExtinctions(Offspring);

					//Form new population
					for(int i=0; i<Size; i++) {
						Copy(Offspring[i], Individuals[i]);
					}
				} else {
					//Postprocess
					for(int i=0; i<Size; i++) {
						Individuals[i].Genes[0] = 1f;
						Individuals[i].Genes[1] = 1f;
						Individuals[i].Genes[2] = 0.5f * (previousPivot.GetWinner().Genes[2] + nextPivot.GetWinner().Genes[2]);
						Individuals[i].Genes[3] = 0.5f * (previousPivot.GetWinner().Genes[3] + nextPivot.GetWinner().Genes[3]);
						Individuals[i].Genes[4] = 0f;
						for(int j=0; j<5; j++) {
							Individuals[i].Momentum[j] = 0f;
						}
						Individuals[i].Fitness = 0f;
						Individuals[i].Extinction = 0f;
					}
				}
			}

			//Returns the mutation probability from two parents
			private float GetMutationProbability(Individual parentA, Individual parentB) {
				float extinction = 0.5f * (parentA.Extinction + parentB.Extinction);
				float inverse = 1f/(float)Dimensionality;
				return extinction * (1f-inverse) + inverse;
			}

			//Returns the mutation strength from two parents
			private float GetMutationStrength(Individual parentA, Individual parentB) {
				return 0.5f * (parentA.Extinction + parentB.Extinction);
			}

			public Individual GetWinner() {
				return Individuals[0];
			}

			public float GetFitness() {
				return GetWinner().Fitness;
			}

			private void Copy(Individual from, Individual to) {
				for(int i=0; i<Dimensionality; i++) {
					to.Genes[i] = Mathf.Clamp(from.Genes[i], Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
					to.Momentum[i] = from.Momentum[i];
				}
				to.Extinction = from.Extinction;
				to.Fitness = from.Fitness;
			}

			private void Reroll(Individual individual) {
				for(int i=0; i<Dimensionality; i++) {
					individual.Genes[i] = Random.Range(Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
				}
			}

			private void Exploit(Individual individual) {
				individual.Fitness = ComputeFitness(individual.Genes);
				for(int i=0; i<Dimensionality; i++) {
					float gene = individual.Genes[i];

					float span = Evolution.UpperBounds[i] - Evolution.LowerBounds[i];

					float incGene = Mathf.Clamp(gene + Random.value*individual.Fitness*span, Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
					individual.Genes[i] = incGene;
					float incFitness = ComputeFitness(individual.Genes);

					float decGene = Mathf.Clamp(gene - Random.value*individual.Fitness*span, Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
					individual.Genes[i] = decGene;
					float decFitness = ComputeFitness(individual.Genes);

					individual.Genes[i] = gene;

					if(incFitness < individual.Fitness) {
						individual.Genes[i] = incGene;
						individual.Momentum[i] = incGene - gene;
						individual.Fitness = incFitness;
					}

					if(decFitness < individual.Fitness) {
						individual.Genes[i] = decGene;
						individual.Momentum[i] = decGene - gene;
						individual.Fitness = decFitness;
					}
				}
			}

			//Rank-based selection of an individual
			private Individual Select(Individual[] pool) {
				double rVal = Random.value * RankProbabilitySum;
				for(int i=0; i<Size; i++) {
					rVal -= RankProbabilities[i];
					if(rVal <= 0.0) {
						return pool[i];
					}
				}
				return pool[Size-1];
			}

			//Sorts all individuals starting with best (lowest) fitness
			private void SortByFitness(Individual[] individuals) {
				System.Array.Sort(individuals,
					delegate(Individual a, Individual b) {
						return a.Fitness.CompareTo(b.Fitness);
					}
				);
			}

			//Multi-Objective RMSE
			private float ComputeFitness(float[] genes) {
				float fitness = 0f;
				for(int i=Interval.Start; i<=Interval.End; i++) {
					float y1 = Evolution.Function.Velocities[i];
					float y2 = Evolution.Function == Evolution.Animation.PhaseFunction ? Evolution.Animation.MirroredPhaseFunction.Velocities[i] : Evolution.Animation.PhaseFunction.Velocities[i];
					float x = Phenotype(genes, i);
					float error = (y1-x)*(y1-x) + (-y2-x)*(-y2-x);
					float sqrError = error*error;
					fitness += sqrError;
				}
				fitness /= Interval.Length;
				fitness = Mathf.Sqrt(fitness);
				return fitness;
			}
			
			public float Phenotype(float[] genes, int frame) {
				return Utility.LinSin(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]*Evolution.Animation.FrameTime, 
					genes[4], 
					frame*Evolution.Animation.FrameTime
					);
			}

			public float Phenotype1(float[] genes, int frame) {
				return Utility.LinSin1(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]*Evolution.Animation.FrameTime, 
					genes[4], 
					frame*Evolution.Animation.FrameTime
					);
			}

			public float Phenotype2(float[] genes, int frame) {
				return Utility.LinSin2(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]*Evolution.Animation.FrameTime, 
					genes[4], 
					frame*Evolution.Animation.FrameTime
					);
			}

			public float Phenotype3(float[] genes, int frame) {
				return Utility.LinSin3(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]*Evolution.Animation.FrameTime, 
					genes[4], 
					frame*Evolution.Animation.FrameTime
					);
			}

			//Compute extinction values
			private void AssignExtinctions(Individual[] individuals) {
				float min = individuals[0].Fitness;
				float max = individuals[Size-1].Fitness;
				for(int i=0; i<Size; i++) {
					float grading = (float)i/((float)Size-1);
					individuals[i].Extinction = (individuals[i].Fitness + min*(grading-1f)) / max;
				}
			}
		}

		public class Individual {
			public float[] Genes;
			public float[] Momentum;
			public float Extinction;
			public float Fitness;
			public Individual(int dimensionality) {
				Genes = new float[dimensionality];
				Momentum = new float[dimensionality];
			}
		}

		public class Interval {
			public int Start;
			public int End;
			public int Length;
			public Interval(int start, int end) {
				Start = start;
				End = end;
				Length = end-start+1;
			}
		}

	}

}