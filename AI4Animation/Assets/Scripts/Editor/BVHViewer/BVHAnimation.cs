using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHAnimation : ScriptableObject {

	public Character Character;
	public Trajectory Trajectory;
	//public Vector3 Offset = Vector3.zero;
	//public Vector3 Rotation = Vector3.zero;
	public Vector3 Orientation = Vector3.zero;

	public bool ShowPreview = false;
	public bool ShowVelocities = false;
	public bool ShowTrajectory = true;
	
	public BVHFrame[] Frames = new BVHFrame[0];
	public BVHFrame CurrentFrame = null;
	public int TotalFrames = 0;
	public float TotalTime = 0f;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public bool Playing = false;

	public BVHPhaseFunction PhaseFunction;
	public BVHStyleFunction StyleFunction;
	public BVHContacts Contacts;

	public float Timescale = 1f;
	public float TimeWindow = 0f;
	public int SequenceStart = 0;
	public int SequenceEnd = 0;
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
	}

	public BVHAnimation Create(BVHViewer viewer) {
		Load(viewer.Path, viewer.UnitScale);
		PhaseFunction = new BVHPhaseFunction(this);
		StyleFunction = new BVHStyleFunction(this);
		Contacts = new BVHContacts(this);
		//string folder = viewer.Path.Substring(0, viewer.Path.LastIndexOf("/")+1);
		string name = viewer.Path.Substring(viewer.Path.LastIndexOf("/")+1);
		if(AssetDatabase.LoadAssetAtPath("Assets/Animation/Project/"+name+".asset", typeof(BVHAnimation)) == null) {
			AssetDatabase.CreateAsset(this , "Assets/Animation/Project/"+name+".asset");
		} else {
			int i = 1;
			while(AssetDatabase.LoadAssetAtPath("Assets/Animation/Project/"+name+" ("+i+").asset", typeof(BVHAnimation)) != null) {
				i += 1;
			}
			AssetDatabase.CreateAsset(this, "Assets/Animation/Project/"+name+" ("+i+").asset");
		}
		return this;
	}

	private void Load(string path, float unitScale) {
		string[] lines = File.ReadAllLines(path);
		char[] whitespace = new char[] {' '};
		int index = 0;

		//Build Hierarchy
		Character = new Character();
		Character.Bone bone = null;
		List<int[]> channels = new List<int[]>();
		List<Transformation> zero = new List<Transformation>();
		for(index = 0; index<lines.Length; index++) {
			if(lines[index] == "MOTION") {
				break;
			}
			string[] entries = lines[index].Split(whitespace);
			for(int entry=0; entry<entries.Length; entry++) {
				if(entries[entry].Contains("ROOT")) {
					bone = Character.AddBone(entries[entry+1], null);
					break;
				} else if(entries[entry].Contains("JOINT")) {
					bone = Character.AddBone(entries[entry+1], bone);
					break;
				} else if(entries[entry].Contains("End")) {
					index += 3;
					break;
				} else if(entries[entry].Contains("OFFSET")) {
					zero.Add(new Transformation(
						new Vector3(Utility.ReadFloat(entries[entry+1]), Utility.ReadFloat(entries[entry+2]), Utility.ReadFloat(entries[entry+3])), 
						Quaternion.identity
						));
					break;
				} else if(entries[entry].Contains("CHANNELS")) {
					int dimensions = Utility.ReadInt(entries[entry+1]);
					int[] channel = new int[dimensions+1];
					channel[0] = dimensions;
					for(int i=0; i<dimensions; i++) {
						if(entries[entry+2+i] == "Xposition") {
							channel[1+i] = 1;
						} else if(entries[entry+2+i] == "Yposition") {
							channel[1+i] = 2;
						} else if(entries[entry+2+i] == "Zposition") {
							channel[1+i] = 3;
						} else if(entries[entry+2+i] == "Xrotation") {
							channel[1+i] = 4;
						} else if(entries[entry+2+i] == "Yrotation") {
							channel[1+i] = 5;
						} else if(entries[entry+2+i] == "Zrotation") {
							channel[1+i] = 6;
						}
					}
					channels.Add(channel);
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
		float[][] motions = new float[TotalFrames][];
		for(int i=index; i<lines.Length; i++) {
			motions[i-index] = Utility.ReadArray(lines[i]);
		}

		//Build Frames
		System.Array.Resize(ref Frames, TotalFrames);
		for(int i=0; i<TotalFrames; i++) {
			Frames[i] = new BVHFrame(this, zero.ToArray(), channels.ToArray(), motions[i], i+1, i*FrameTime, unitScale);
		}

		//Create Trajectory
		ComputeTrajectory();

		//Initialise Variables
		TimeWindow = TotalTime;
		SequenceStart = 1;
		SequenceEnd = 1;

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
		ForwardKinematics();
	}

	public void ReloadFrame() {
		ForwardKinematics();
	}

	private void ForwardKinematics() {
		for(int i=0; i<Character.Bones.Length; i++) {
			//Character.Bones[i].SetPosition(Offset + Quaternion.Euler(Rotation) * CurrentFrame.Positions[i]);
			//Character.Bones[i].SetRotation(Quaternion.Euler(Rotation) * CurrentFrame.Rotations[i]);
			//Vector3 position = CurrentFrame.Positions[i];
			//position.x = -position.x;
			Character.Bones[i].SetPosition(CurrentFrame.Positions[i]);
			Character.Bones[i].SetRotation(CurrentFrame.Rotations[i]);
		}
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

	public void ComputeTrajectory() {
		Trajectory = new Trajectory(TotalFrames, 0);
		for(int i=0; i<TotalFrames; i++) {
			Trajectory.Points[i].Position = GetRootPosition(Frames[i]);
			Trajectory.Points[i].Direction = GetRootDirection(Frames[i]);
			Trajectory.Points[i].Postprocess();
		}
	}

	public void ExportSkeleton(Character.Bone bone, Transform parent) {
		Transform instance = new GameObject(bone.GetName()).transform;
		instance.SetParent(parent);
		instance.position = bone.GetPosition();
		instance.rotation = bone.GetRotation();
		for(int i=0; i<bone.GetChildCount(); i++) {
			ExportSkeleton(bone.GetChild(Character, i), instance);
		}
	}

	public Trajectory ExtractTrajectory(BVHFrame frame) {
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
		return trajectory;
	}

	public Vector3 GetRootPosition(BVHFrame frame) {
		return Utility.ProjectGround(frame.Positions[0], LayerMask.GetMask("Ground"));
	}

	public Vector3 GetRootDirection(BVHFrame frame) {
		//DIRTY HACK FOR SITTING...
		int hipIndex = Character.FindBone("Hips").GetIndex();
		int neckIndex = Character.FindBone("Neck").GetIndex();
		Vector3 forward = frame.Positions[neckIndex] - frame.Positions[hipIndex];
		forward.y = 0f;
		forward = forward.normalized;
		return forward;
	}

	/*
	private void SetOffset(Vector3 offset) {
		if(Offset != offset) {
			Offset = offset;
			ReloadFrame();
		}
	}

	private void SetRotation(Vector3 rotation) {
		if(Rotation != rotation) {
			Rotation = rotation;
			ReloadFrame();
		}
	}
	*/

	private void SetOrientation(Vector3 orientation) {
		if(Orientation != orientation) {
			Orientation = orientation;
		}
	}

	public void Inspector() {
		/*
		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("Name", GUILayout.Width(150f));
		string newName = EditorGUILayout.TextField(name);
		if(newName != name) {
			AssetDatabase.RenameAsset(AssetDatabase.GetAssetPath(this), newName);
		}
		EditorGUILayout.EndHorizontal();
		*/

		Character.Inspector();
		
		ShowVelocities = EditorGUILayout.Toggle("Show Velocities", ShowVelocities);
		ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", ShowTrajectory);
		//SetOffset(EditorGUILayout.Vector3Field("Offset", Offset));
		//SetRotation(EditorGUILayout.Vector3Field("Rotation", Rotation));
		SetOrientation(EditorGUILayout.Vector3Field("Orientation", Orientation));
		if(Utility.GUIButton("Compute Trajectory", Utility.DarkGreen, Utility.White)) {
			ComputeTrajectory();
		}

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("Frames: " + TotalFrames, GUILayout.Width(100f));
		EditorGUILayout.LabelField("Time: " + TotalTime.ToString("F3") + "s", GUILayout.Width(100f));
		EditorGUILayout.LabelField("Time/Frame: " + FrameTime.ToString("F3") + "s" + " (" + (1f/FrameTime).ToString("F1") + "Hz)", GUILayout.Width(175f));
		EditorGUILayout.LabelField("Preview:", GUILayout.Width(50f), GUILayout.Height(20f)); 
		ShowPreview = EditorGUILayout.Toggle(ShowPreview, GUILayout.Width(20f), GUILayout.Height(20f));
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


		PhaseFunction.Inspector();

		StyleFunction.Inspector();

		//Contacts.Inspector();

		Utility.SetGUIColor(Utility.LightGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(Utility.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.LabelField("Export");
			}

			if(Utility.GUIButton("Skeleton", Utility.DarkGrey, Utility.White)) {
				ExportSkeleton(Character.GetRoot(), null);
			}

			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Start", GUILayout.Width(67f));
			SequenceStart = EditorGUILayout.IntSlider(SequenceStart, 1, SequenceEnd, GUILayout.Width(182f));
			EditorGUILayout.LabelField("End", GUILayout.Width(67f));
			SequenceEnd = EditorGUILayout.IntSlider(SequenceEnd, SequenceStart, TotalFrames, GUILayout.Width(182f));
			EditorGUILayout.EndHorizontal();
			
		}
	}

	public void Draw() {
		if(ShowPreview) {
			float step = 1f;
			BVHAnimation.BVHFrame frame = CurrentFrame;
			UnityGL.Start();
			for(int i=1; i<TotalFrames; i++) {
				UnityGL.DrawLine(Frames[i-1].Positions[0], Frames[i].Positions[0], Utility.Magenta);
			}
			UnityGL.Finish();
			for(float i=0f; i<=TotalTime; i+=step) {
				LoadFrame(i);
				Character.DrawSimple();
			}
			LoadFrame(frame);
		}

		Contacts.Draw();

		if(ShowTrajectory) {
			Trajectory.Draw();
		} else {
			ExtractTrajectory(CurrentFrame).Draw();
		}
		
		Character.Draw();
		
		if(ShowVelocities) {
			UnityGL.Start();
			for(int i=0; i<CurrentFrame.Velocities.Length; i++) {
				UnityGL.DrawArrow(
					CurrentFrame.Positions[i],
					CurrentFrame.Positions[i] + CurrentFrame.Velocities[i],
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
	public class BVHPhaseFunction {
		public BVHAnimation Animation;

		public bool[] Keys;
		public float[] Phase;
		
		public Vector2 VariablesScroll;
		public bool[] Variables;
		
		public float[] Values;
		public float Smoothing;
		public float Amplification;

		public BVHPhaseFunction(BVHAnimation animation) {
			Animation = animation;
			Keys = new bool[Animation.TotalFrames];
			Phase = new float[Animation.TotalFrames];
			Keys[0] = true;
			Keys[Animation.TotalFrames-1] = true;
			Phase[0] = 0f;
			Phase[Animation.TotalFrames-1] = 1f;
			Variables = new bool[Animation.Character.Bones.Length];
			Values = new float[Animation.TotalFrames];
			SetSmoothing(0.25f);
			SetAmplification(1f);
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

		public void SetSmoothing(float smoothing) {
			if(Smoothing != smoothing) {
				Smoothing = smoothing;
				ComputeValues();
			}
		}

		public void SetAmplification(float amplification) {
			if(Amplification != amplification) {
				Amplification = amplification;
				ComputeValues();
			}
		}

		public void ToggleVariable(int index) {
			Variables[index] = !Variables[index];
			ComputeValues();
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

				if(IsKey(Animation.CurrentFrame)) {
					if(Utility.GUIButton("Unset Key", Utility.Grey, Utility.White)) {
						SetKey(Animation.CurrentFrame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", Utility.DarkGrey, Utility.White)) {
						SetKey(Animation.CurrentFrame, true);
					}
				}

				SetSmoothing(EditorGUILayout.FloatField("Smoothing", Smoothing));
				SetAmplification(EditorGUILayout.FloatField("Amplification", Amplification));

				if(IsKey(Animation.CurrentFrame)) {
					SetPhase(Animation.CurrentFrame, EditorGUILayout.Slider("Phase", GetPhase(Animation.CurrentFrame), 0f, 1f));
				} else {
					EditorGUI.BeginDisabledGroup(true);
					SetPhase(Animation.CurrentFrame, EditorGUILayout.Slider("Phase", GetPhase(Animation.CurrentFrame), 0f, 1f));
					EditorGUI.EndDisabledGroup();
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

				Vector3 prevPos = Vector3.zero;
				Vector3 newPos = Vector3.zero;
				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Values[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Values[i+start] * rect.height;
					UnityGL.DrawLine(prevPos, newPos, Utility.Cyan);
				}

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

				//Seconds labels
				float timestamp = startTime;
				while(timestamp <= endTime) {
					float floor = Mathf.FloorToInt(timestamp);
					if(floor >= startTime && floor <= endTime) {
						top.x = rect.xMin + (float)(Animation.GetFrame(floor).Index-start)/elements * rect.width;
						UnityGL.DrawCircle(top, 1.5f, Utility.White);
					}
					timestamp += 1f;
				}
				//

				top.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				UnityGL.DrawLine(top, bottom, Utility.Red);
				UnityGL.DrawCircle(top, 3f, Utility.Green);
				UnityGL.DrawCircle(bottom, 3f, Utility.Green);

				top.x = rect.xMin + (float)(Animation.SequenceStart-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.SequenceStart-start)/elements * rect.width;
				Vector3 a = top;
				Vector3 b = bottom;
				top.x = rect.xMin + (float)(Animation.SequenceEnd-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.SequenceEnd-start)/elements * rect.width;
				Vector3 c = top;
				Vector3 d = bottom;

				Color yellow = Utility.Yellow;
				yellow.a = 0.25f;
				UnityGL.DrawTriangle(a, b, c, yellow);
				UnityGL.DrawTriangle(d, b, c, yellow);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", Utility.DarkGrey, Utility.White, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();
			}

			UnityGL.Finish();
		}

		private void ComputeValues() {
			Values = new float[Animation.TotalFrames];
			float min = float.MaxValue;
			float max = float.MinValue;
			for(int i=0; i<Animation.TotalFrames; i++) {
				for(int j=0; j<Animation.Character.Bones.Length; j++) {
					if(Variables[j]) {
						Values[i] = Mathf.Max(Values[i], Animation.Frames[i].SmoothTranslationalVelocityValue(j, Smoothing) + Mathf.Deg2Rad*Animation.Frames[i].SmoothRotationalVelocityValue(j, Smoothing));
					}
				}
				if(Values[i] < min) {
					min = Values[i];
				}
				if(Values[i] > max) {
					max = Values[i];
				}
			}
			for(int i=0; i<Values.Length; i++) {
				Values[i] = Utility.Normalise(Values[i], min, max, 0f, 1f);
				Values[i] = (float)System.Math.Tanh((double)(Amplification*Values[i]));
			}
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
					AddStyle("Run");
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

				if(IsKey(Animation.CurrentFrame)) {
					if(Utility.GUIButton("Unset Key", Utility.Grey, Utility.White)) {
						SetKey(Animation.CurrentFrame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", Utility.DarkGrey, Utility.White)) {
						SetKey(Animation.CurrentFrame, true);
					}
				}

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

				top.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width;
				UnityGL.DrawLine(top, bottom, Utility.Red);
				UnityGL.DrawCircle(top, 3f, Utility.Green);
				UnityGL.DrawCircle(bottom, 3f, Utility.Green);

				top.x = rect.xMin + (float)(Animation.SequenceStart-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.SequenceStart-start)/elements * rect.width;
				Vector3 a = top;
				Vector3 b = bottom;
				top.x = rect.xMin + (float)(Animation.SequenceEnd-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(Animation.SequenceEnd-start)/elements * rect.width;
				Vector3 c = top;
				Vector3 d = bottom;

				Color yellow = Utility.Yellow;
				yellow.a = 0.25f;
				UnityGL.DrawTriangle(a, b, c, yellow);
				UnityGL.DrawTriangle(d, b, c, yellow);

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

	[System.Serializable]
	public class BVHContacts {
		public BVHAnimation Animation;

		public float VelocityThreshold;

		public Vector2 VariablesScroll;
		public bool[] Variables;

		public BVHContact[] Contacts;

		public BVHContacts(BVHAnimation animation) {
			Animation = animation;
			Variables = new bool[Animation.Character.Bones.Length];
			Contacts = new BVHContact[0];
		}

		public void AssignVelocityThreshold(float velocityThreshold) {
			if(VelocityThreshold != velocityThreshold) {
				VelocityThreshold = velocityThreshold;
				for(int i=0; i<Contacts.Length; i++) {
					Contacts[i].ComputeLabels(VelocityThreshold);
				}
			}
		}

		public void Inspector() {
			Utility.SetGUIColor(Utility.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Contacts");
				}

				AssignVelocityThreshold(EditorGUILayout.FloatField("Velocity Threshold", VelocityThreshold));

				VariablesScroll = EditorGUILayout.BeginScrollView(VariablesScroll, GUILayout.Height(100f));
				for(int i=0; i<Animation.Character.Bones.Length; i++) {
					if(Variables[i]) {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.DarkGreen, Utility.White)) {
							RemoveContact(i);
						}
					} else {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.DarkRed, Utility.White)) {
							AddContact(i);
						}
					}
				}
				EditorGUILayout.EndScrollView();

			}
		}

		public void AddContact(int index) {
			System.Array.Resize(ref Contacts, Contacts.Length+1);
			Contacts[Contacts.Length-1] = new BVHContact(Animation, index);
			Contacts[Contacts.Length-1].ComputeLabels(VelocityThreshold);
			Variables[index] = true;
		}

		public void RemoveContact(int index) {
			for(int i=index; i<Contacts.Length-1; i++) {
				Contacts[i] = Contacts[i+1];
			}
			System.Array.Resize(ref Contacts, Contacts.Length-1);
			Variables[index] = false;
		}

		public void Draw() {
			Color[] colors = Utility.GetRainbowColors(Contacts.Length);
			UnityGL.Start();
			for(int i=0; i<Contacts.Length; i++) {
				for(int j=0; j<Contacts[i].Labels.Length; j++) {
					if(Contacts[i].Labels[j]) {
						UnityGL.DrawIsocelesTriangle(Contacts[i].GetPosition(Animation.Frames[j]), 0.005f, colors[i]);
					}
				}
			}
			UnityGL.Finish();
		}
	}

	[System.Serializable]
	public class BVHContact {
		public BVHAnimation Animation;
		public bool[] Labels;
		public int Index;

		public BVHContact(BVHAnimation animation, int index) {
			Animation = animation;
			Labels = new bool[Animation.TotalFrames];
			Index = index;
		}

		public void ComputeLabels(float velocityThreshold) {
			for(int i=0; i<Labels.Length; i++) {
				Labels[i] = Animation.Frames[i].Velocities[Index].magnitude <= velocityThreshold;
			}
		}

		public bool GetContact(BVHFrame frame) {
			return Labels[frame.Index-1];
		}

		public Vector3 GetPosition(BVHFrame frame) {
			return frame.Positions[Index];
		}

		public Quaternion GetRotation(BVHFrame frame) {
			return frame.Rotations[Index];
		}
	}

	[System.Serializable]
	public class BVHFrame {
		public BVHAnimation Animation;

		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;
		public Vector3[] Velocities;
		
		public BVHFrame(BVHAnimation animation, Transformation[] zeros, int[][] channels, float[] motion, int index, float timestamp, float unitScale) {
			Animation = animation;
			Index = index;
			Timestamp = timestamp;
			Positions = new Vector3[Animation.Character.Bones.Length];
			Rotations = new Quaternion[Animation.Character.Bones.Length];
			Velocities = new Vector3[Animation.Character.Bones.Length];
			
			//Forward Kinematics
			int channel = 0;
			for(int i=0; i<Animation.Character.Bones.Length; i++) {
				if(i == 0) {
					//Root
					Vector3 position = new Vector3(motion[channel+0], motion[channel+1], motion[channel+2]);
					Quaternion rotation = Quaternion.identity;
					for(int j=0; j<3; j++) {
						if(channels[i][4+j] == 4) {
							rotation *= Quaternion.AngleAxis(motion[channel+3+j], Vector3.right);
						}
						if(channels[i][4+j] == 5) {
							rotation *= Quaternion.AngleAxis(motion[channel+3+j], Vector3.up);
						}
						if(channels[i][4+j] == 6) {
							rotation *= Quaternion.AngleAxis(motion[channel+3+j], Vector3.forward);
						}
					}
					Animation.Character.Bones[i].SetPosition(position / unitScale);
					Animation.Character.Bones[i].SetRotation(rotation);
					channel += 6;
				} else {
					//Childs
					Quaternion rotation = Quaternion.identity;
					for(int j=0; j<3; j++) {
						if(channels[i][1+j] == 4) {
							rotation *= Quaternion.AngleAxis(motion[channel+j], Vector3.right);
						}
						if(channels[i][1+j] == 5) {
							rotation *= Quaternion.AngleAxis(motion[channel+j], Vector3.up);
						}
						if(channels[i][1+j] == 6) {
							rotation *= Quaternion.AngleAxis(motion[channel+j], Vector3.forward);
						}
					}
					Animation.Character.Bones[i].SetPosition(Animation.Character.Bones[i].GetParent(Animation.Character).GetPosition() + Animation.Character.Bones[i].GetParent(Animation.Character).GetRotation() * zeros[i].Position / unitScale);
					Animation.Character.Bones[i].SetRotation(Animation.Character.Bones[i].GetParent(Animation.Character).GetRotation() * zeros[i].Rotation * rotation);
					channel += 3;
				}
			}
			for(int i=0; i<Animation.Character.Bones.Length; i++) {
				Positions[i] = Animation.Character.Bones[i].GetPosition();
				Rotations[i] = Animation.Character.Bones[i].GetRotation();
				if(Index > 1) {
					//Velocities[i] = (Positions[i] - Animation.GetFrame(Index-1).Positions[i]) / Animation.FrameTime;
					Velocities[i] = Positions[i] - Animation.GetFrame(Index-1).Positions[i];
				}
			}
		}

		public Vector3 TranslationalVelocityVector(int index) {
			if(Index == 1) {
				return Vector3.zero;
			}
			return Positions[index] - Animation.GetFrame(Index-1).Positions[index];
		}

		public float TranslationalVelocityValue(int index) {
			if(Index == 1) {
				return 0f;
			}
			return (Positions[index] - Animation.GetFrame(Index-1).Positions[index]).magnitude;
		}

		public float RotationalVelocityValue(int index) {
			if(Index == 1) {
				return 0f;
			}
			return Quaternion.Angle(Rotations[index], Animation.GetFrame(Index-1).Rotations[index]);
		}

		public Vector3 SmoothTranslationalVelocityVector(int index, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, Timestamp+smoothing/2f));
			Vector3 velocity = Vector3.zero;
			for(int k=1; k<frames.Length; k++) {
				//velocity += (frames[k].Positions[index] - frames[k-1].Positions[index]) / Animation.FrameTime;
				velocity += frames[k].Positions[index] - frames[k-1].Positions[index];
			}
			velocity /= frames.Length;
			return velocity;
		}

		public float SmoothTranslationalVelocityValue(int index, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, Timestamp+smoothing/2f));
			Vector3 velocity = Vector3.zero;
			for(int k=1; k<frames.Length; k++) {
				//velocity += (frames[k].Positions[index] - frames[k-1].Positions[index]).magnitude / Animation.FrameTime;
				velocity += (frames[k].Positions[index] - frames[k-1].Positions[index]);
			}
			velocity /= frames.Length;
			return velocity.magnitude;
		}

		public float SmoothRotationalVelocityValue(int index, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, Timestamp+smoothing/2f));
			float velocity = 0f;
			for(int k=1; k<frames.Length; k++) {
				//velocity += Quaternion.Angle(frames[k-1].Rotations[index], frames[k].Rotations[index]) / Animation.FrameTime;
				velocity += Quaternion.Angle(frames[k-1].Rotations[index], frames[k].Rotations[index]);
			}
			velocity /= frames.Length;
			return velocity;
		}

		/*
		public float SmoothAccelerationValue(int index, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, Timestamp+smoothing/2f));
			float acceleration = 0f;
			for(int k=1; k<frames.Length; k++) {
				//acceleration += (frames[k].Velocities[index] - frames[k-1].Velocities[index]).magnitude / Animation.FrameTime;
				acceleration += (frames[k].Velocities[index] - frames[k-1].Velocities[index]).magnitude;
			}
			acceleration /= frames.Length;
			return acceleration;
		}
		*/
	}

}