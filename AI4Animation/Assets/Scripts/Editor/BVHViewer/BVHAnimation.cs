using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHAnimation : ScriptableObject {

	public BVHData Data;
	public Character Character;
	public int[] Symmetry;
	
	public float UnitScale = 100f;
	public Vector3 PositionOffset = Vector3.zero;
	public Vector3 RotationOffset = Vector3.zero;
	public Vector3 Orientation = Vector3.zero;
	
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

	public BVHFrame CurrentFrame = null;
	public int TotalFrames = 0;
	public float TotalTime = 0f;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public bool Playing = false;
	public float Timescale = 1f;
	public float TimeWindow = 0f;
	public System.DateTime Timestamp;

	public void EditorUpdate() {
		if(Playing) {
			PlayTime += Timescale*(float)Utility.GetElapsedTime(Timestamp);
			if(PlayTime > TotalTime) {
				PlayTime -= TotalTime;
			}
			LoadFrame(PlayTime);
		}
		Timestamp = Utility.GetTimestamp();

		PhaseFunction.EditorUpdate();
		MirroredPhaseFunction.EditorUpdate();
	}

	public BVHAnimation Create(BVHViewer viewer) {
		Load(viewer.Path);
		PhaseFunction = new BVHPhaseFunction(this);
		MirroredPhaseFunction = new BVHPhaseFunction(this);
		PhaseFunction.SetVelocitySmoothing(0.1f);
		PhaseFunction.SetVelocityThreshold(0.1f);
		PhaseFunction.SetHeightThreshold(0.1f);
		StyleFunction = new BVHStyleFunction(this);
		Sequences = new BVHSequence[0];
		string name = viewer.Path.Substring(viewer.Path.LastIndexOf("/")+1);
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

	private void Load(string path) {
		string[] lines = File.ReadAllLines(path);
		char[] whitespace = new char[] {' '};
		int index = 0;

		//Build Hierarchy
		Data = new BVHData();
		Character = new Character();
		Character.Bone bone = null;
		string name = string.Empty;
		Vector3 offset = Vector3.zero;
		int[] channels = null;
		for(index = 0; index<lines.Length; index++) {
			if(lines[index] == "MOTION") {
				break;
			}
			string[] entries = lines[index].Split(whitespace);
			for(int entry=0; entry<entries.Length; entry++) {
				if(entries[entry].Contains("ROOT")) {
					name = entries[entry+1];
					bone = Character.AddBone(name, null);
					break;
				} else if(entries[entry].Contains("JOINT")) {
					name = entries[entry+1];
					bone = Character.AddBone(name, bone);
					break;
				} else if(entries[entry].Contains("End")) {
					index += 3;
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
					Data.AddBone(name, offset, channels);
					break;
				} else if(entries[entry].Contains("}")) {
					if(bone != Character.GetRoot()) {
						bone = bone.GetParent(Character);
					}
					break;
				}
			}
		}

		//Read frame count
		index += 1;
		TotalFrames = Utility.ReadInt(lines[index].Substring(8));

		//Read frame time
		index += 1;
		FrameTime = Utility.ReadFloat(lines[index].Substring(12));

		//Compute total time
		TotalTime = TotalFrames * FrameTime;

		//Read motions
		index += 1;
		for(int i=index; i<lines.Length; i++) {
			Data.AddMotion(Utility.ReadArray(lines[i]));
		}

		//Resize Frames
		System.Array.Resize(ref Frames, TotalFrames);
		for(int i=0; i<TotalFrames; i++) {
			Frames[i] = new BVHFrame(this, i+1, i*FrameTime);
		}

		//Create Symmetry Table
		Symmetry = new int[Character.Bones.Length];
		for(int i=0; i<Symmetry.Length; i++) {
			Symmetry[i] = i;
		}

		//Initialise Variables
		TimeWindow = TotalTime;

		//Generate
		ComputeSymmetry();
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
		if(CurrentFrame.Index < TotalFrames) {
			LoadFrame(CurrentFrame.Index+1);
		}
	}

	public void LoadPreviousFrame() {
		if(CurrentFrame.Index > 1) {
			LoadFrame(CurrentFrame.Index-1);
		}
	}

	public void LoadFrame(BVHFrame frame) {
		if(frame == null) {
			return;
		}
		if(CurrentFrame == frame) {
			return;
		}
		CurrentFrame = frame;
	}

	public void LoadFrame(int index) {
		LoadFrame(GetFrame(index));
	}

	public void LoadFrame(float time) {
		LoadFrame(GetFrame(time));
	}

	public BVHFrame GetFrame(int index) {
		if(index < 1 || index > TotalFrames) {
			Debug.Log("Please specify an index between 1 and " + TotalFrames + ".");
			return null;
		}
		return Frames[index-1];
	}

	public BVHFrame GetFrame(float time) {
		if(time < 0f || time > TotalTime) {
			Debug.Log("Please specify a time between 0 and " + TotalTime + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.RoundToInt(time / FrameTime) + 1, TotalFrames));
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

	public void ComputeSymmetry() {
		for(int i=0; i<Character.Bones.Length; i++) {
			string name = Character.Bones[i].GetName();
			if(name.Contains("Left")) {
				Character.Bone bone = Character.FindBone("Right"+name.Substring(4));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.GetIndex();
				}
			} else if(name.Contains("Right")) {
				Character.Bone bone = Character.FindBone("Left"+name.Substring(5));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.GetIndex();
				}
			} else {
				Symmetry[i] = i;
			}
		}
	}

	public void ComputeFrames() {
		for(int i=0; i<Frames.Length; i++) {
			Frames[i].Generate();
		}
	}

	public void ComputeTrajectory() {
		Trajectory = new Trajectory(TotalFrames, 0);
		LayerMask mask = LayerMask.GetMask("Ground");
		for(int i=0; i<TotalFrames; i++) {
			Vector3 rootPos = Utility.ProjectGround(Frames[i].Positions[0], mask);
			//Vector3 rootDir = Frames[i].Rotations[0] * Vector3.forward;
			
			int hipIndex = Character.FindBone("Hips").GetIndex();
			int neckIndex = Character.FindBone("Neck").GetIndex();
			Vector3 rootDir = Frames[i].Positions[neckIndex] - Frames[i].Positions[hipIndex];
			rootDir.y = 0f;
			rootDir = rootDir.normalized;
			
			Trajectory.Points[i].Position = rootPos;
			Trajectory.Points[i].Direction = rootDir;
			Trajectory.Points[i].Postprocess();
		}
	}

	public Vector3[] ExtractPositions(BVHFrame frame, bool mirrored) {
		Vector3[] positions = new Vector3[Character.Bones.Length];
		for(int i=0; i<positions.Length; i++) {
			positions[i] = mirrored ? frame.Positions[Symmetry[i]].MirrorX() : frame.Positions[i];
		}
		return positions;
	}

	public Quaternion[] ExtractRotations(BVHFrame frame, bool mirrored) {
		Quaternion[] rotations = new Quaternion[Character.Bones.Length];
		for(int i=0; i<rotations.Length; i++) {
			rotations[i] = mirrored ? frame.Rotations[Symmetry[i]].MirrorX() : frame.Rotations[i];
		}
		return rotations;
	}

	public Vector3[] ExtractVelocities(BVHFrame frame, bool mirrored, float smoothing=0f) {
		Vector3[] velocities = new Vector3[Character.Bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = mirrored ? frame.ComputeVelocity(Symmetry[i], smoothing).MirrorX() : frame.ComputeVelocity(i, smoothing);
		}
		return velocities;
	}

	public Trajectory ExtractTrajectory(BVHFrame frame, bool mirrored) {
		Trajectory trajectory = new Trajectory(12, StyleFunction.Styles.Length);
		//Past
		for(int i=0; i<6; i++) {
			float timestamp = Mathf.Clamp(frame.Timestamp - 1f + (float)i/6f, 0f, TotalTime);
			int index = GetFrame(timestamp).Index;
			trajectory.Points[i].Index = Trajectory.Points[index-1].Index;
			trajectory.Points[i].Position = Trajectory.Points[index-1].Position;
			trajectory.Points[i].Direction = Trajectory.Points[index-1].Direction;
			trajectory.Points[i].LeftSample = Trajectory.Points[index-1].LeftSample;
			trajectory.Points[i].RightSample = Trajectory.Points[index-1].RightSample;
			trajectory.Points[i].Rise = Trajectory.Points[index-1].Rise;
			for(int j=0; j<StyleFunction.Styles.Length; j++) {
				trajectory.Points[i].Styles[j] = StyleFunction.Styles[j].Values[index-1];
			}
		}
		//Current
		trajectory.Points[6].Index = Trajectory.Points[frame.Index-1].Index;
		trajectory.Points[6].Position = Trajectory.Points[frame.Index-1].Position;
		trajectory.Points[6].Direction = Trajectory.Points[frame.Index-1].Direction;
		trajectory.Points[6].LeftSample = Trajectory.Points[frame.Index-1].LeftSample;
		trajectory.Points[6].RightSample = Trajectory.Points[frame.Index-1].RightSample;
		trajectory.Points[6].Rise = Trajectory.Points[frame.Index-1].Rise;
		//Future
		for(int i=7; i<12; i++) {
			float timestamp = Mathf.Clamp(frame.Timestamp + (float)(i-6)/5f, 0f, TotalTime);
			int index = GetFrame(timestamp).Index;
			trajectory.Points[i].Index = Trajectory.Points[index-1].Index;
			trajectory.Points[i].Position = Trajectory.Points[index-1].Position;
			trajectory.Points[i].Direction = Trajectory.Points[index-1].Direction;
			trajectory.Points[i].LeftSample = Trajectory.Points[index-1].LeftSample;
			trajectory.Points[i].RightSample = Trajectory.Points[index-1].RightSample;
			trajectory.Points[i].Rise = Trajectory.Points[index-1].Rise;
			for(int j=0; j<StyleFunction.Styles.Length; j++) {
				trajectory.Points[i].Styles[j] = StyleFunction.Styles[j].Values[index-1];
			}
		}

		if(mirrored) {
			for(int i=0; i<12; i++) {
				trajectory.Points[i].Position = trajectory.Points[i].Position.MirrorX();
				trajectory.Points[i].Direction = trajectory.Points[i].Direction.MirrorX();
				trajectory.Points[i].LeftSample = trajectory.Points[i].LeftSample.MirrorX();
				trajectory.Points[i].RightSample = trajectory.Points[i].RightSample.MirrorX();
			}
		}

		return trajectory;
	}

	private void SetUnitScale(float unitScale) {
		if(UnitScale != unitScale) {
			UnitScale = unitScale;
			ComputeFrames();
			ComputeTrajectory();
		}
	}

	private void SetPositionOffset(Vector3 value) {
		if(PositionOffset != value) {
			PositionOffset = value;
			ComputeFrames();
			ComputeTrajectory();
		}
	}

	private void SetRotationOffset(Vector3 value) {
		if(RotationOffset != value) {
			RotationOffset = value;
			ComputeFrames();
			ComputeTrajectory();
		}
	}

	private void SetOrientation(Vector3 orientation) {
		if(Orientation != orientation) {
			Orientation = orientation;
			ComputeFrames();
			ComputeTrajectory();
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
		Character.Inspector();

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("Show Mirrored", GUILayout.Width(100f));
		ShowMirrored = EditorGUILayout.Toggle(ShowMirrored, GUILayout.Width(40));
		EditorGUILayout.LabelField("Show Velocities", GUILayout.Width(100f));
		ShowVelocities = EditorGUILayout.Toggle(ShowVelocities, GUILayout.Width(40));
		EditorGUILayout.LabelField("Show Trajectory", GUILayout.Width(100f));
		ShowTrajectory = EditorGUILayout.Toggle(ShowTrajectory, GUILayout.Width(40));
		EditorGUILayout.LabelField("Show Preview", GUILayout.Width(100f));
		ShowPreview = EditorGUILayout.Toggle(ShowPreview, GUILayout.Width(40));
		EditorGUILayout.EndHorizontal();

		SetUnitScale(EditorGUILayout.FloatField("Unit Scale", UnitScale));
		SetPositionOffset(EditorGUILayout.Vector3Field("Position Offset", PositionOffset));
		SetRotationOffset(EditorGUILayout.Vector3Field("Rotation Offset", RotationOffset));
		SetOrientation(EditorGUILayout.Vector3Field("Orientation", Orientation));

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("Frames: " + TotalFrames, GUILayout.Width(100f));
		EditorGUILayout.LabelField("Time: " + TotalTime.ToString("F3") + "s", GUILayout.Width(100f));
		EditorGUILayout.LabelField("Time/Frame: " + FrameTime.ToString("F3") + "s" + " (" + (1f/FrameTime).ToString("F1") + "Hz)", GUILayout.Width(175f));
		EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
		Timescale = EditorGUILayout.FloatField(Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
		EditorGUILayout.EndHorizontal();

		Utility.SetGUIColor(Utility.DarkGrey);
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
			if(Utility.GUIButton("<", Utility.Grey, Utility.White, 20f, 20f)) {
				LoadPreviousFrame();
			}
			if(Utility.GUIButton(">", Utility.Grey, Utility.White, 20f, 20f)) {
				LoadNextFrame();
			}
			BVHAnimation.BVHFrame frame = GetFrame(EditorGUILayout.IntSlider(CurrentFrame.Index, 1, TotalFrames, GUILayout.Width(440f)));
			if(CurrentFrame != frame) {
				PlayTime = frame.Timestamp;
				LoadFrame(frame);
			}
			EditorGUILayout.LabelField(CurrentFrame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
			EditorGUILayout.EndHorizontal();
		}

		Utility.SetGUIColor(Utility.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("-1s", Utility.Grey, Utility.White, 65, 20f)) {
				LoadFrame(Mathf.Max(CurrentFrame.Timestamp - 1f, 0f));
			}
			TimeWindow = EditorGUILayout.Slider(TimeWindow, 2f*FrameTime, TotalTime, GUILayout.Width(440f));
			if(Utility.GUIButton("+1s", Utility.Grey, Utility.White, 65, 20f)) {
				LoadFrame(Mathf.Min(CurrentFrame.Timestamp + 1f, TotalTime));
			}
			EditorGUILayout.EndHorizontal();
		}

		if(ShowMirrored) {
			MirroredPhaseFunction.Inspector();
		} else {
			PhaseFunction.Inspector();
		}

		StyleFunction.Inspector();

		if(Utility.GUIButton("Export Skeleton", Utility.DarkGrey, Utility.White)) {
			ExportSkeleton(Character.GetRoot(), null);
		}

		Utility.SetGUIColor(Utility.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			for(int i=0; i<Sequences.Length; i++) {
				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Start", GUILayout.Width(67f));
				Sequences[i].Start = EditorGUILayout.IntSlider(Sequences[i].Start, 1, TotalFrames, GUILayout.Width(182f));
				EditorGUILayout.LabelField("End", GUILayout.Width(67f));
				Sequences[i].End = EditorGUILayout.IntSlider(Sequences[i].End, 1, TotalFrames, GUILayout.Width(182f));
				if(Utility.GUIButton("Auto", Utility.DarkGrey, Utility.White)) {
					Sequences[i].Auto();
				}
				EditorGUILayout.EndHorizontal();
			}

			if(Utility.GUIButton("Add Sequence", Utility.DarkGrey, Utility.White)) {
				AddSequence();
			}
			if(Utility.GUIButton("Remove Sequence", Utility.DarkGrey, Utility.White)) {
				RemoveSequence();
			}
		}

		Utility.SetGUIColor(Utility.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("Compute Symmetry", Utility.DarkGrey, Utility.White)) {
				ComputeSymmetry();
			}
			string[] names = new string[Character.Bones.Length];
			for(int i=0; i<Character.Bones.Length; i++) {
				names[i] = Character.Bones[i].GetName();
			}
			for(int i=0; i<Character.Bones.Length; i++) {
				EditorGUILayout.BeginHorizontal();
				EditorGUI.BeginDisabledGroup(true);
				EditorGUILayout.TextField(names[i]);
				EditorGUI.EndDisabledGroup();
				Symmetry[i] = EditorGUILayout.Popup(Symmetry[i], names);
				EditorGUILayout.EndHorizontal();
			}
		}

	}

	private void ExportSkeleton(Character.Bone bone, Transform parent) {
		Transform instance = new GameObject(bone.GetName()).transform;
		instance.SetParent(parent);
		instance.position = bone.GetPosition();
		instance.rotation = bone.GetRotation();
		for(int i=0; i<bone.GetChildCount(); i++) {
			ExportSkeleton(bone.GetChild(Character, i), instance);
		}
	}

	public void Draw() {
		if(ShowPreview) {
			float step = 1f;
			UnityGL.Start();
			for(int i=1; i<TotalFrames; i++) {
				UnityGL.DrawLine(Frames[i-1].Positions[0], Frames[i].Positions[0], Utility.Magenta);
			}
			UnityGL.Finish();
			for(float i=0f; i<=TotalTime; i+=step) {
				Vector3[] pos = ExtractPositions(GetFrame(i), ShowMirrored);
				Quaternion[] rot = ExtractRotations(GetFrame(i), ShowMirrored);
				for(int j=0; j<Character.Bones.Length; j++) {
					Character.Bones[j].SetPosition(pos[j]);
					Character.Bones[j].SetRotation(rot[j]);
				}
				Character.DrawSimple();
			}
		}

		if(ShowTrajectory) {
			Trajectory.Draw();
		} else {
			ExtractTrajectory(CurrentFrame, ShowMirrored).Draw();
		}
		
		Vector3[] positions = ExtractPositions(CurrentFrame, ShowMirrored);
		Quaternion[] rotations = ExtractRotations(CurrentFrame, ShowMirrored);
		for(int i=0; i<Character.Bones.Length; i++) {
			Character.Bones[i].SetPosition(positions[i]);
			Character.Bones[i].SetRotation(rotations[i]);
		}
		Character.Draw();

		UnityGL.Start();
		BVHPhaseFunction function = ShowMirrored ? MirroredPhaseFunction : PhaseFunction;
		for(int i=0; i<function.Variables.Length; i++) {
			if(function.Variables[i]) {
				Color red = Utility.Red;
				red.a = 0.25f;
				Color green = Utility.Green;
				green.a = 0.25f;
				UnityGL.DrawCircle(ShowMirrored ? positions[Symmetry[i]] : positions[i], Character.BoneSize*1.25f, green);
				UnityGL.DrawCircle(ShowMirrored ? positions[i] : positions[Symmetry[i]], Character.BoneSize*1.25f, red);
			}
		}
		UnityGL.Finish();
		
		if(ShowVelocities) {
			Vector3[] velocities = ExtractVelocities(CurrentFrame, ShowMirrored, 0.1f);
			UnityGL.Start();
			for(int i=0; i<Character.Bones.Length; i++) {
				UnityGL.DrawArrow(
					positions[i],
					positions[i] + velocities[i]/FrameTime,
					0.75f,
					0.0075f,
					0.05f,
					new Color(0f, 1f, 0f, 0.5f)
				);				
			}
			UnityGL.Finish();
		}
	}

	[System.Serializable]
	public class BVHSequence {
		public BVHAnimation Animation;
		public int Start;
		public int End;
		public BVHSequence(BVHAnimation animation) {
			Animation = animation;
			Start = 1;
			End = 1;
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
				End = Animation.TotalFrames-60;
			}
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

		public void AddBone(string name, Vector3 offset, int[] channels) {
			System.Array.Resize(ref Bones, Bones.Length+1);
			Bones[Bones.Length-1] = new Bone(name, offset, channels);
		}

		public void AddMotion(float[] values) {
			System.Array.Resize(ref Motions, Motions.Length+1);
			Motions[Motions.Length-1] = new Motion(values);
		}

		[System.Serializable]
		public class Bone {
			public string Name;
			public Vector3 Offset;
			public int[] Channels;
			public Bone(string name, Vector3 offset, int[] channels) {
				Name = name;
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

		public Vector3[] LocalPositions;
		public Quaternion[] LocalRotations;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public BVHFrame(BVHAnimation animation, int index, float timestamp) {
			Animation = animation;
			Index = index;
			Timestamp = timestamp;
			
			LocalPositions = new Vector3[Animation.Character.Bones.Length];
			LocalRotations = new Quaternion[Animation.Character.Bones.Length];
			Positions = new Vector3[Animation.Character.Bones.Length];
			Rotations = new Quaternion[Animation.Character.Bones.Length];
		}

		public void Generate() {
			int channel;

			//Original
			channel = 0;
			for(int i=0; i<Animation.Character.Bones.Length; i++) {
				BVHData.Bone info = Animation.Data.Bones[i];
				BVHData.Motion motion = Animation.Data.Motions[Index-1];
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
				LocalPositions[i] = (i == 0 ? Animation.PositionOffset + Quaternion.Euler(Animation.RotationOffset) * position / Animation.UnitScale : info.Offset / Animation.UnitScale);
				LocalRotations[i] = i == 0 ? Quaternion.Euler(Animation.RotationOffset) * Quaternion.Euler(Animation.Orientation) * rotation : rotation;

				Character.Bone bone = Animation.Character.Bones[i];
				Character.Bone parent = bone.GetParent(Animation.Character);

				Vector3 parentPosition = parent == null ? Vector3.zero : parent.GetPosition();
				Quaternion parentRotation = parent == null ? Quaternion.identity : parent.GetRotation();

				bone.SetPosition(parentPosition + parentRotation * LocalPositions[i]);
				bone.SetRotation(parentRotation * LocalRotations[i]);
			}
			for(int i=0; i<Animation.Character.Bones.Length; i++) {
				Positions[i] = Animation.Character.Bones[i].GetPosition();
				Rotations[i] = Animation.Character.Bones[i].GetRotation();
			}
		}

		public Vector3 ComputeVelocity(int index, float smoothing) {
			if(smoothing == 0f) {
				return Positions[index] - Animation.GetFrame(Mathf.Max(1, Index-1)).Positions[index];
			}
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, Timestamp+smoothing/2f));
			Vector3 velocity = Vector3.zero;
			for(int k=1; k<frames.Length; k++) {
				velocity += frames[k].Positions[index] - frames[k-1].Positions[index];
			}
			velocity /= frames.Length;
			return velocity;
		}
	
		public float ComputeTranslationalVelocity(int index, float smoothing) {
			return ComputeVelocity(index, smoothing).magnitude;
		}

		public float ComputeRotationalVelocity(int index, float smoothing) {
			if(smoothing == 0f) {
				return Quaternion.Angle(Animation.GetFrame(Mathf.Max(1, Index-1)).Rotations[index], Rotations[index]);
			}
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, Timestamp+smoothing/2f));
			float velocity = 0f;
			for(int k=1; k<frames.Length; k++) {
				velocity += Quaternion.Angle(frames[k-1].Rotations[index], frames[k].Rotations[index]);
			}
			velocity /= frames.Length;
			return velocity;
		}
	}

	[System.Serializable]
	public class BVHPhaseFunction {
		public BVHAnimation Animation;

		public bool[] Keys;
		public float[] Phase;
		public float[] Cycle;
		public float[] NormalisedCycle;
		
		public Vector2 VariablesScroll;
		public bool[] Variables;
		public float VelocitySmoothing;
		public float VelocityThreshold;
		public float HeightThreshold;

		public float[] Heights;
		public float[] Velocities;
		public float[] NormalisedVelocities;

		private BVHEvolution Optimiser;
		private bool Optimising;

		public BVHPhaseFunction(BVHAnimation animation) {
			Animation = animation;
			Keys = new bool[Animation.TotalFrames];
			Phase = new float[Animation.TotalFrames];
			Cycle = new float[Animation.TotalFrames];
			NormalisedCycle = new float[Animation.TotalFrames];
			Keys[0] = true;
			Keys[Animation.TotalFrames-1] = true;
			Phase[0] = 0f;
			Phase[Animation.TotalFrames-1] = 1f;
			Variables = new bool[Animation.Character.Bones.Length];
			Velocities = new float[Animation.TotalFrames];
			NormalisedVelocities = new float[Animation.TotalFrames];
			Heights = new float[Animation.TotalFrames];
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
				for(int i=frame.Index+1; i<=Animation.TotalFrames; i++) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return Animation.Frames[Animation.TotalFrames-1];
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
			if(b.Index == Animation.TotalFrames) {
				BVHFrame last = Animation.Frames[Animation.TotalFrames-1];
				BVHFrame previous1 = GetPreviousKey(last);
				BVHFrame previous2 = GetPreviousKey(previous1);
				Keys[Animation.TotalFrames-1] = true;
				float xLast = last.Timestamp - previous1.Timestamp;
				float mLast = previous1.Timestamp - previous2.Timestamp;
				SetPhase(last, Mathf.Clamp(xLast / mLast, 0f, 1f));
			}
		}

		private void ComputeValues() {
			for(int i=0; i<Animation.TotalFrames; i++) {
				Heights[i] = 0f;
				Velocities[i] = 0f;
				NormalisedVelocities[i] = 0f;

			}
			float min, max;
			
			LayerMask mask = LayerMask.GetMask("Ground");
			min = float.MaxValue;
			max = float.MinValue;
			for(int i=0; i<Animation.TotalFrames; i++) {
				for(int j=0; j<Animation.Character.Bones.Length; j++) {
					if(Variables[j]) {
						float offset = Mathf.Max(0f, Animation.Frames[i].Positions[j].y - Utility.ProjectGround(Animation.Frames[i].Positions[j], mask).y);
						Heights[i] = Mathf.Max(Heights[i], offset);
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
			for(int i=0; i<Animation.TotalFrames; i++) {
				for(int j=0; j<Animation.Character.Bones.Length; j++) {
					if(Variables[j]) {
						float boneVelocity = Animation.Frames[i].ComputeTranslationalVelocity(j, VelocitySmoothing) / Animation.FrameTime;
						Velocities[i] = Mathf.Max(0f, Velocities[i], boneVelocity);
						if(Velocities[i] < VelocityThreshold || Heights[i] < HeightThreshold) {
							Velocities[i] = 0f;
						}
					}
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
					Cycle = new float[Animation.TotalFrames];
				} else if(Cycle.Length != Animation.TotalFrames) {
					Cycle = new float[Animation.TotalFrames];
				}
				if(Optimiser == null) {
					Optimiser = new BVHEvolution(Animation, this);
				}
				Optimiser.Optimise();
			}
		}

		public void Inspector() {
			UnityGL.Start();

			Utility.SetGUIColor(Utility.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Phase Function");
				}

				Utility.SetGUIColor(Utility.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Optimising) {
						if(Utility.GUIButton("Stop Optimisation", Utility.LightGrey, Utility.Black)) {
							Optimising = !Optimising;
						}
					} else {
						if(Utility.GUIButton("Start Optimisation", Utility.DarkGrey, Utility.White)) {
							Optimising = !Optimising;
						}
					}
					if(Optimiser != null) {
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Fitness: " + Optimiser.GetFitness());
						if(Utility.GUIButton("Restart", Utility.DarkGrey, Utility.White)) {
							Optimiser.Initialise();
						}
						EditorGUILayout.EndHorizontal();

						Optimiser.RecombinationRate = EditorGUILayout.Slider("Recombination Rate", Optimiser.RecombinationRate, 0f, 1f);
						Optimiser.MutationRate = EditorGUILayout.Slider("Mutation Rate", Optimiser.MutationRate, 0f, 1f);
						Optimiser.MutationStrength = EditorGUILayout.Slider("Mutation Strength", Optimiser.MutationStrength, 0f, 1f);

						Optimiser.SetAmplitude(EditorGUILayout.Slider("Amplitude", Optimiser.Amplitude, 0, BVHEvolution.AMPLITUDE));
						Optimiser.SetFrequency(EditorGUILayout.Slider("Frequency", Optimiser.Frequency, 0f, BVHEvolution.FREQUENCY));
						Optimiser.SetShift(EditorGUILayout.Slider("Shift", Optimiser.Shift, 0, BVHEvolution.SHIFT));
						Optimiser.SetOffset(EditorGUILayout.Slider("Offset", Optimiser.Offset, 0, BVHEvolution.OFFSET));
						Optimiser.SetSlope(EditorGUILayout.Slider("Slope", Optimiser.Slope, 0, BVHEvolution.SLOPE));
						Optimiser.SetGradientCutoff(EditorGUILayout.Slider("Gradient Cutoff", Optimiser.GradientCutoff, 0f, BVHEvolution.GRADIENTCUTOFF));
						Optimiser.SetWindow(EditorGUILayout.Slider("Window", Optimiser.Window, 1f, BVHEvolution.WINDOW));
					} else {
						EditorGUILayout.LabelField("No optimiser available.");
					}
				}

				VariablesScroll = EditorGUILayout.BeginScrollView(VariablesScroll, GUILayout.Height(100f));
				for(int i=0; i<Animation.Character.Bones.Length; i++) {
					if(Variables[i]) {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.DarkGreen, Utility.White)) {
							ToggleVariable(i);
						}
					} else {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.DarkRed, Utility.White)) {
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

				if(IsKey(Animation.CurrentFrame)) {
					if(Utility.GUIButton("Unset Key", Utility.Grey, Utility.White)) {
						SetKey(Animation.CurrentFrame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", Utility.DarkGrey, Utility.White)) {
						SetKey(Animation.CurrentFrame, true);
					}
				}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", Utility.DarkGrey, Utility.White, 25f, 50f)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, Utility.Black);

				float startTime = Animation.CurrentFrame.Timestamp-Animation.TimeWindow/2f;
				float endTime = Animation.CurrentFrame.Timestamp+Animation.TimeWindow/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > Animation.TotalTime) {
					startTime -= endTime-Animation.TotalTime;
					endTime = Animation.TotalTime;
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(Animation.TotalTime, endTime);
				int start = Animation.GetFrame(startTime).Index;
				int end = Animation.GetFrame(endTime).Index;
				int elements = end-start;

				//TODO REMOVE LATER
				if(NormalisedVelocities == null) {
					NormalisedVelocities = new float[Animation.TotalFrames];
				} else if(NormalisedVelocities.Length == 0) {
					NormalisedVelocities = new float[Animation.TotalFrames];
				}
				if(NormalisedCycle == null) {
					NormalisedCycle = new float[Animation.TotalFrames];
				} else if(NormalisedCycle.Length == 0) {
					NormalisedCycle = new float[Animation.TotalFrames];
				}
				//

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
					UnityGL.DrawLine(prevPos, newPos, this == Animation.PhaseFunction ? Utility.Green : Utility.Red);
				}

				//Mirrored Velocities
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Animation.MirroredPhaseFunction.NormalisedVelocities[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Animation.MirroredPhaseFunction.NormalisedVelocities[i+start] * rect.height;
					UnityGL.DrawLine(prevPos, newPos, this == Animation.PhaseFunction ? Utility.Red : Utility.Green);
				}

				//Heights
				/*
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Heights[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Heights[i+start] * rect.height;
					UnityGL.DrawLine(prevPos, newPos, Utility.Red);
				}
				*/
				
				//Cycle
				/*
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - NormalisedCycle[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - NormalisedCycle[i+start] * rect.height;
					UnityGL.DrawLine(prevPos, newPos, Utility.Yellow);
				}
				*/

				//Phase
				BVHFrame A = Animation.GetFrame(start);
				if(A.Index == 1) {
					bottom.x = rect.xMin;
					top.x = rect.xMin;
					UnityGL.DrawLine(bottom, top, Utility.Magenta);
				}
				BVHFrame B = GetNextKey(A);
				while(A != B) {
					prevPos.x = rect.xMin + (float)(A.Index-start)/elements * rect.width;
					prevPos.y = rect.yMax - Mathf.Repeat(Phase[A.Index-1], 1f) * rect.height;
					newPos.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					newPos.y = rect.yMax - Phase[B.Index-1] * rect.height;
					UnityGL.DrawLine(prevPos, newPos, Utility.White);
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					UnityGL.DrawLine(bottom, top, Utility.Magenta);
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
						UnityGL.DrawCircle(top, 2.5f, Utility.White);
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

					Color yellow = Utility.Yellow;
					yellow.a = 0.25f;
					UnityGL.DrawTriangle(a, b, c, yellow);
					UnityGL.DrawTriangle(d, b, c, yellow);
				}

				//Current Pivot
				top.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				UnityGL.DrawLine(top, bottom, Utility.Yellow);
				UnityGL.DrawCircle(top, 3f, Utility.Green);
				UnityGL.DrawCircle(bottom, 3f, Utility.Green);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", Utility.DarkGrey, Utility.White, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();
			}

			UnityGL.Finish();
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
			Transition = 0.25f;
			Keys = new bool[Animation.TotalFrames];
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
					AddStyle("Sit");
					break;

					case STYLE.Quadruped:
					AddStyle("Idle");
					AddStyle("Walk");
					AddStyle("Sprint");
					AddStyle("Jump");
					AddStyle("Sit");
					AddStyle("Lie");
					AddStyle("Stand");
					AddStyle("Sleep");
					break;
				}
			}
		}

		public void AddStyle(string name = "Style") {
			System.Array.Resize(ref Styles, Styles.Length+1);
			Styles[Styles.Length-1] = new BVHStyle(name, Animation.TotalFrames);
		}

		public void RemoveStyle() {
			if(Styles.Length == 0) {
				return;
			}
			System.Array.Resize(ref Styles, Styles.Length-1);
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
				Refresh();
			} else {
				if(!IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = false;
				Refresh();
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
			return null;
		}

		public BVHFrame GetNextKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index+1; i<=Animation.TotalFrames; i++) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return null;
		}

		private void SetTransition(float value) {
			value = Mathf.Max(value, 0f);
			if(Transition == value) {
				return;
			}
			Transition = value;
			Refresh();
		}

		private void Refresh() {
			for(int i=0; i<Animation.TotalFrames; i++) {
				if(Keys[i]) {
					for(int j=0; j<Styles.Length; j++) {
						Interpolate(Animation.Frames[i], j);
					}
				}
			}
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
			int end = next == null ? Animation.TotalFrames : next.Index-1;
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
			float nextTS = next == null ? Animation.TotalTime : next.Timestamp;
			float prevTime = Mathf.Abs(frame.Timestamp - prevTS);
			float nextTime = Mathf.Abs(frame.Timestamp - nextTS);
			float window = Mathf.Min(prevTime, nextTime, Transition);
			return window;
		}

		public void Inspector() {
			UnityGL.Start();

			Utility.SetGUIColor(Utility.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
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
							if(Utility.GUIButton("On", Utility.DarkGreen, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, false);
							}
						} else {
							if(Utility.GUIButton("Off", Utility.DarkRed, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, true);
							}
						}
					} else {
						EditorGUI.BeginDisabledGroup(true);
						if(GetFlag(Animation.CurrentFrame, i)) {
							if(Utility.GUIButton("On", Utility.DarkGreen, Color.white)) {
								SetFlag(Animation.CurrentFrame, i, false);
							}
						} else {
							if(Utility.GUIButton("Off", Utility.DarkRed, Color.white)) {
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
					if(Utility.GUIButton("Add Style", Utility.DarkGrey, Utility.White)) {
						AddStyle();
					}
					if(Utility.GUIButton("Remove Style", Utility.DarkGrey, Utility.White)) {
						RemoveStyle();
					}
					EditorGUILayout.EndHorizontal();
				}

				if(IsKey(Animation.CurrentFrame)) {
					if(Utility.GUIButton("Unset Key", Utility.Grey, Utility.White)) {
						SetKey(Animation.CurrentFrame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", Utility.DarkGrey, Utility.White)) {
						SetKey(Animation.CurrentFrame, true);
					}
				}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", Utility.DarkGrey, Utility.White, 25f, 50f)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, Utility.Black);

				float startTime = Animation.CurrentFrame.Timestamp-Animation.TimeWindow/2f;
				float endTime = Animation.CurrentFrame.Timestamp+Animation.TimeWindow/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > Animation.TotalTime) {
					startTime -= (endTime-Animation.TotalTime);
					endTime = Animation.TotalTime;
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(Animation.TotalTime, endTime);
				int start = Animation.GetFrame(startTime).Index;
				int end = Animation.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 prevPos = Vector3.zero;
				Vector3 newPos = Vector3.zero;
				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);
				
				Color[] colors = Utility.GetRainbowColors(Styles.Length);
				BVHFrame A = Animation.GetFrame(start);
				if(IsKey(A)) {
					bottom.x = rect.xMin;
					top.x = rect.xMin;
					UnityGL.DrawLine(bottom, top, Utility.Magenta);
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
							UnityGL.DrawLine(prevPos, newPos, colors[i]);
						}
					}
					
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					UnityGL.DrawLine(bottom, top, Utility.Magenta);
					
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
						UnityGL.DrawCircle(top, 2.5f, Utility.White);
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

					Color yellow = Utility.Yellow;
					yellow.a = 0.25f;
					UnityGL.DrawTriangle(a, b, c, yellow);
					UnityGL.DrawTriangle(d, b, c, yellow);
				}

				//Current Pivot
				top.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				UnityGL.DrawLine(top, bottom, Utility.Yellow);
				UnityGL.DrawCircle(top, 3f, Utility.Green);
				UnityGL.DrawCircle(bottom, 3f, Utility.Green);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", Utility.DarkGrey, Utility.White, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();

			}

			UnityGL.Finish();
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
		public static float FREQUENCY = 5f;
		public static float SHIFT = Mathf.PI;
		public static float OFFSET = 10f;
		public static float SLOPE = 10f;
		public static float GRADIENTCUTOFF = 0.25f;
		public static float WINDOW = 10f;
		
		public BVHAnimation Animation;
		public BVHPhaseFunction Function;

		public Population[] Populations;

		public float[] LowerBounds;
		public float[] UpperBounds;

		public float RecombinationRate = 0.9f;
		public float MutationRate = 0.25f;
		public float MutationStrength = 0.1f;

		public float Amplitude = AMPLITUDE;
		public float Frequency = FREQUENCY/2f;
		public float Shift = SHIFT;
		public float Offset = OFFSET;
		public float Slope = SLOPE;
		public float GradientCutoff = GRADIENTCUTOFF;

		public float Window = 1f;
		public Interval[] Intervals;

		public BVHEvolution(BVHAnimation animation, BVHPhaseFunction function) {
			Animation = animation;
			Function = function;

			LowerBounds = new float[5];
			UpperBounds = new float[5];
			LowerBounds[0] = -Amplitude;
			UpperBounds[0] = Amplitude;
			LowerBounds[1] = -Frequency;
			UpperBounds[1] = Frequency;
			LowerBounds[2] = -Shift;
			UpperBounds[2] = Shift;
			LowerBounds[3] = -Offset;
			UpperBounds[3] = Offset;
			LowerBounds[4] = -Slope;
			UpperBounds[4] = Slope;

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

		public void SetGradientCutoff(float value) {
			if(GradientCutoff != value) {
				GradientCutoff = value;
				Assign();
			}
		}

		public void SetWindow(float value) {
			if(Window != value) {
				Window = value;
				Initialise();
			}
		}

		public void Initialise() {
			Intervals = new Interval[Mathf.FloorToInt(Animation.TotalTime / Window) + 1];
			for(int i=0; i<Intervals.Length; i++) {
				int start = Animation.GetFrame(i*Window).Index-1;
				int end = Animation.GetFrame(Mathf.Min(Animation.TotalTime, (i+1)*Window)).Index-2;
				if(end == Animation.TotalFrames-2) {
					end += 1;
				}
				Intervals[i] = new Interval(start, end);
			}

			Populations = new Population[Intervals.Length];
			for(int i=0; i<Populations.Length; i++) {
				Populations[i] = new Population(this, 50, 5);
			}
			for(int i=0; i<Populations.Length; i++) {
				Populations[i].Initialise(Intervals[i]);
			}

			Assign();
		}

		public void Optimise() {
			for(int i=0; i<Populations.Length; i++) {
				Populations[i].Evolve(Intervals[i]);
			}
			Assign();
		}

		
		public void Assign() {
			for(int i=0; i<Animation.TotalFrames; i++) {
				Function.Keys[i] = false;
				Function.Phase[i] = 0f;
				Function.Cycle[i] = 0f;
				Function.NormalisedCycle[i] = 0f;
			}

			//Compute average velocities
			float[] avgVelocities = new float[Intervals.Length];
			for(int i=0; i<Intervals.Length; i++) {
				avgVelocities[i] = 0f;
				for(int j=Intervals[i].Start; j<=Intervals[i].End; j++) {
					avgVelocities[i] += Function.Velocities[j];
				}
				avgVelocities[i] /= Intervals[i].Length;
			}

			//Compute average values
			float avgA = 0f;
			float avgF = 0f;
			float avgS = 0f;
			float avgO = 0f;
			int elements = 0;
			for(int i=0; i<Intervals.Length; i++) {
				if(avgVelocities[i] > 0f) {
					avgA += Populations[i].GetWinner().Genes[0];
					avgF += Populations[i].GetWinner().Genes[1];
					avgS += Populations[i].GetWinner().Genes[2];
					avgO += Populations[i].GetWinner().Genes[3];
					elements += 1;
				}
			}
			if(elements != 0) {
				avgA /= elements;
				avgF /= elements;
				avgS /= elements;
				avgO /= elements;
			}

			//Compute cycle
			float min = float.MaxValue;
			float max = float.MinValue;
			for(int i=0; i<Populations.Length; i++) {
				Individual winner = Populations[i].GetWinner();
				for(int j=Intervals[i].Start; j<=Intervals[i].End; j++) {
					Function.Cycle[j] = Utility.LinSin(
						avgVelocities[i] > 0f ? winner.Genes[0] : avgA, 
						avgVelocities[i] > 0f ? winner.Genes[1] : avgF, 
						avgVelocities[i] > 0f ? winner.Genes[2] : avgS, 
						avgVelocities[i] > 0f ? winner.Genes[3] : avgO, 
						avgVelocities[i] > 0f ? winner.Genes[4] : 0f,
						(j-Intervals[i].Start)*Animation.FrameTime
					);
					min = Mathf.Min(min, Function.Cycle[j]);
					max = Mathf.Max(max, Function.Cycle[j]);
				}
			}
			for(int i=0; i<Function.NormalisedCycle.Length; i++) {
				Function.NormalisedCycle[i] = Utility.Normalise(Function.Cycle[i], min, max, 0f, 1f);
			}

			//Label phase

			//Fill with frequency negative turning points
			for(int k=0; k<Intervals.Length; k++) {
				if(avgVelocities[k] > 0f) {
					Individual winner = Populations[k].GetWinner();
					for(int i=Intervals[k].Start; i<=Intervals[k].End; i++) {
						int pivot = i-Intervals[k].Start;
						int left = Mathf.Max(0, pivot-1);
						int right = Mathf.Min(Function.Cycle.Length-1, pivot+1);
						float prevGradient = Utility.LinSin1(winner.Genes[0], winner.Genes[1], winner.Genes[2], winner.Genes[3], winner.Genes[4], left*Animation.FrameTime);
						float nextGradient = Utility.LinSin1(winner.Genes[0], winner.Genes[1], winner.Genes[2], winner.Genes[3], winner.Genes[4], right*Animation.FrameTime);
						float gradient = Utility.LinSin1(winner.Genes[0], winner.Genes[1], winner.Genes[2], winner.Genes[3], winner.Genes[4], pivot*Animation.FrameTime);
						if(prevGradient > gradient && nextGradient > gradient) {
							Function.Keys[i] = true;
						}
					}
				}
			}

			//Fill idle intervals with 1s intervals
			for(int k=0; k<Intervals.Length; k++) {
				if(avgVelocities[k] == 0f) {
					float start = Animation.Frames[Intervals[k].Start].Timestamp;
					float end = Animation.Frames[Intervals[k].End].Timestamp;
					BVHFrame prev = Function.GetPreviousKey(Animation.Frames[Intervals[k].Start]);
					BVHFrame next = Function.GetNextKey(Animation.Frames[Intervals[k].End]);
					if(prev != null) {
						start = prev.Timestamp;
					}
					if(next != null) {
						end = next.Timestamp;
					}
					float frequency = Mathf.Round(end-start) / (end-start);
					float timestamp = start + frequency;
					while(timestamp <= end && frequency > 0f) {
						Function.Keys[Animation.GetFrame(Mathf.Clamp(Mathf.Round(timestamp), 0f, Animation.TotalTime)).Index-1] = true;
						timestamp += frequency;
					}
				}
			}

			for(int i=0; i<Function.Keys.Length; i++) {
				if(Function.Keys[i]) {
					Function.SetPhase(Animation.Frames[i], i == 0 ? 0f : 1f);
				}
			}
		}

		private float ComputeGradient(int index, int offset) {
			if(offset == 0) {
				return 0f;
			} else {
				float gradient = 0f;
				if(offset > 0) {
					for(int i=1; i<=offset; i++) {
						gradient += Function.Cycle[Mathf.Min(Function.Cycle.Length-1, index+i)] - Function.Cycle[index];
					}
				}
				if(offset < 0) {
					for(int i=-1; i>=offset; i--) {
						gradient += Function.Cycle[Mathf.Max(0, index+i)] - Function.Cycle[index];
					}
				}
				return gradient / Mathf.Abs(offset);
			}
		}


		public float GetFitness() {
			float fitness = 0f;
			for(int i=0; i<Populations.Length; i++) {
				fitness += Populations[i].GetFitness();
			}
			return fitness / Populations.Length;
		}

		public class Population {
			public BVHEvolution Evolution;
			public Individual[] Individuals;
			public int Size;
			public int Dimensionality;

			public Population(BVHEvolution evolution, int size, int dimensionality) {
				Evolution = evolution;
				Size = size;
				Dimensionality = dimensionality;
			}

			public void Initialise(Interval interval) {
				//Create individuals
				Individuals = new Individual[Size];
				for(int i=0; i<Individuals.Length; i++) {
					Individuals[i] = new Individual(Dimensionality);
				}

				//Initialise randomly
				for(int i=0; i<Individuals.Length; i++) {
					Reroll(Individuals[i]);
				}

				//Evaluate fitness
				for(int i=0; i<Individuals.Length; i++) {
					Evaluate(Individuals[i], interval);
				}

				//Sort
				SortByFitness(Individuals);
			}

			public void Evolve(Interval interval) {
				//Create offspring
				Individual[] offspring = new Individual[Individuals.Length];
				for(int i=0; i<offspring.Length; i++) {
					offspring[i] = new Individual(Dimensionality);
				}

				//Copy elite
				Copy(Individuals[0], offspring[0]);

				//Evolve offspring
				for(int i=1; i<offspring.Length; i++) {
					Recombine(Select(Individuals), Select(Individuals), offspring[i]);
					Mutate(offspring[i]);
				}

				//Constraints
				for(int i=0; i<offspring.Length; i++) {
					Constrain(offspring[i]);
				}

				//Evaluate fitness
				for(int i=0; i<offspring.Length; i++) {
					Evaluate(offspring[i], interval);
				}

				//Sort
				SortByFitness(offspring);

				//Form new population
				Individuals = offspring;
			}

			public float GetAverageFitness() {
				float fitness = 0f;
				for(int i=0; i<Individuals.Length; i++) {
					fitness += Individuals[i].Fitness;
				}
				fitness /= Individuals.Length;
				return fitness;
			}

			public Individual GetWinner() {
				return Individuals[0];
			}

			public float GetFitness() {
				return GetWinner().Fitness;
			}

			private void Copy(Individual from, Individual to) {
				for(int i=0; i<from.Genes.Length; i++) {
					to.Genes[i] = from.Genes[i];
				}
			}

			private void Reroll(Individual individual) {
				for(int i=0; i<individual.Genes.Length; i++) {
					individual.Genes[i] = Random.Range(Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
				}
			}

			private void Recombine(Individual parentA, Individual parentB, Individual offspring) {
				if(Random.value < Evolution.RecombinationRate) {
					for(int i=0; i<offspring.Genes.Length; i++) {
						if(Random.value < 0.5f) {
							offspring.Genes[i] = parentA.Genes[i];
						} else {
							offspring.Genes[i] = parentB.Genes[i];
						}
					}
				} else {
					Reroll(offspring);
				}
			}

			private void Mutate(Individual individual) {
				for(int i=0; i<individual.Genes.Length; i++) {
					if(Random.value <= Evolution.MutationRate) {
						float span = Evolution.UpperBounds[i] - Evolution.LowerBounds[i];
						individual.Genes[i] += Random.Range(-Evolution.MutationStrength*span, Evolution.MutationStrength*span);
					}
				}	
			}

			private void Constrain(Individual indivdiual) {
				for(int i=0; i<indivdiual.Genes.Length; i++) {
					indivdiual.Genes[i] = Mathf.Clamp(indivdiual.Genes[i], Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
				}
			}

			//Rank-based selection of an individual
			private Individual Select(Individual[] pool) {
				double[] probabilities = new double[pool.Length];
				double rankSum = (double)(pool.Length*(pool.Length+1)) / 2.0;
				for(int i=0; i<pool.Length; i++) {
					probabilities[i] = (double)(pool.Length-i)/rankSum;
				}
				return pool[GetRandomWeightedIndex(probabilities, pool.Length)];
			}
			
			//Returns a random index with respect to the probability weights
			private int GetRandomWeightedIndex(double[] probabilities, int count) {
				double weightSum = 0.0;
				for(int i=0; i<count; i++) {
					weightSum += probabilities[i];
				}
				double rVal = Random.value * weightSum;
				for(int i=0; i<count; i++) {
					rVal -= probabilities[i];
					if(rVal <= 0.0) {
						return i;
					}
				}
				return count-1;
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
			private void Evaluate(Individual individual, Interval interval) {
				float fitness = 0f;
				for(int i=interval.Start; i<=interval.End; i++) {
					float y1 = Evolution.Function.Velocities[i];
					float y2 = Evolution.Function == Evolution.Animation.PhaseFunction ? Evolution.Animation.MirroredPhaseFunction.Velocities[i] : Evolution.Animation.PhaseFunction.Velocities[i];
					float x = Utility.LinSin(individual.Genes[0], individual.Genes[1], individual.Genes[2], individual.Genes[3], individual.Genes[4], (i-interval.Start)*Evolution.Animation.FrameTime);
					float error = (y1-x)*(y1-x) + (-y2-x)*(-y2-x);

					//error *= Evolution.Function.Velocities[i];
					float sqrError = error*error;
					fitness += sqrError;
				}
				fitness /= interval.Length;
				fitness = Mathf.Sqrt(fitness);
				individual.Fitness = fitness;
			}
		}

		public class Individual {
			public float[] Genes;
			public float Fitness;

			public Individual(int dimensionality) {
				Genes = new float[dimensionality];
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