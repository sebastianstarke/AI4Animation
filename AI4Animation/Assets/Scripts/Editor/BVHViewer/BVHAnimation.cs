using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHAnimation : ScriptableObject {

	public Character Character;
	public Vector3 ForwardOrientation = Vector3.zero;

	public bool ShowPreview = false;
	public bool ShowVelocities = false;
	
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
		if(AssetDatabase.LoadAssetAtPath("Assets/Resources/BVH/MoCap.asset", typeof(BVHAnimation)) == null) {
			AssetDatabase.CreateAsset(this , "Assets/Resources/BVH/MoCap.asset");
		} else {
			int i = 1;
			while(AssetDatabase.LoadAssetAtPath("Assets/Resources/BVH/"+"MoCap ("+i+").asset", typeof(BVHAnimation)) != null) {
				i += 1;
			}
			AssetDatabase.CreateAsset(this, "Assets/Resources/BVH/"+"MoCap ("+i+").asset");
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

		//Skip frame count
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

	public void LoadFrame(BVHFrame Frame) {
		if(Frame == null) {
			return;
		}
		if(CurrentFrame == Frame) {
			return;
		}
		CurrentFrame = Frame;
		for(int i=0; i<Character.Bones.Length; i++) {
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

	public void Draw() {
		if(ShowPreview) {
			float step = 1f;
			BVHAnimation.BVHFrame frame = CurrentFrame;
			UnityGL.Start();
			for(int i=2; i<=TotalFrames; i++) {
				UnityGL.DrawLine(GetFrame(i-1).Positions[0], GetFrame(i).Positions[0], Utility.Magenta);
			}
			UnityGL.Finish();
			for(float i=0f; i<=TotalTime; i+=step) {
				LoadFrame(i);
				Character.DrawSimple();
			}
			LoadFrame(frame);
		}

		Contacts.Draw();

		GenerateTrajectory(CurrentFrame).Draw();
		
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

	public void ExportSkeleton(Character.Bone bone, Transform parent) {
		Transform instance = new GameObject(bone.GetName()).transform;
		instance.SetParent(parent);
		instance.position = bone.GetPosition();
		instance.rotation = bone.GetRotation();
		for(int i=0; i<bone.GetChildCount(); i++) {
			ExportSkeleton(bone.GetChild(Character, i), instance);
		}
	}

	public Trajectory GenerateTrajectory(BVHFrame frame) {
		Trajectory trajectory = new Trajectory();
		//Past
		int past = trajectory.GetPastPoints() * trajectory.GetDensity();
		for(int i=0; i<past; i+=trajectory.GetDensity()) {
			float timestamp = Mathf.Clamp(frame.Timestamp - 1f + (float)i/(float)past, 0f, TotalTime);
			trajectory.Points[i].SetPosition(GetFrame(timestamp).Positions[0]);
			Vector3 direction = GetFrame(timestamp).Rotations[0] * Quaternion.Euler(ForwardOrientation) * Vector3.forward;
			direction.y = 0f;
			direction = direction.normalized;
			trajectory.Points[i].SetDirection(direction);
		}
		//Current
		trajectory.Points[past].SetPosition(frame.Positions[0]);
		Vector3 dir = frame.Rotations[0] * Quaternion.Euler(ForwardOrientation) * Vector3.forward;
		dir.y = 0f;
		dir = dir.normalized;
		trajectory.Points[past].SetDirection(dir);
		//Future
		int future = trajectory.GetFuturePoints() * trajectory.GetDensity();
		for(int i=trajectory.GetDensity(); i<=future; i+=trajectory.GetDensity()) {
			float timestamp = Mathf.Clamp(frame.Timestamp + (float)i/(float)future, 0f, TotalTime);
			trajectory.Points[past+i].SetPosition(GetFrame(timestamp).Positions[0]);
			Vector3 direction = GetFrame(timestamp).Rotations[0] * Quaternion.Euler(ForwardOrientation) * Vector3.forward;
			direction.y = 0f;
			direction = direction.normalized;
			trajectory.Points[past+i].SetDirection(direction);
		}
		return trajectory;
	}

	[System.Serializable]
	public class BVHPhaseFunction {
		public BVHAnimation Animation;

		public bool[] Keys;
		
		public Vector2 VariablesScroll;
		public bool[] Variables;
		
		public float[] Values;
		public float Smoothing;
		public float TimeWindow;

		public BVHPhaseFunction(BVHAnimation animation) {
			Animation = animation;
			Keys = new bool[Animation.TotalFrames];
			Variables = new bool[Animation.Character.Bones.Length];
			Values = new float[Animation.TotalFrames];
			SetSmoothing(0.1f);
			TimeWindow = Animation.TotalTime;
		}

		public void SetSmoothing(float smoothing) {
			if(Smoothing != smoothing) {
				Smoothing = smoothing;
				ComputeFunction();
			}
		}

		public void ToggleVariable(int index) {
			Variables[index] = !Variables[index];
			ComputeFunction();
		}

		public BVHFrame GetFirstKey() {
			for(int i=0; i<Keys.Length; i++) {
				if(Keys[i]) {
					return Animation.Frames[i];
				}
			}
			return null;
		}

		public BVHFrame GetLastKey() {
			for(int i=Keys.Length-1; i>=0; i--) {
				if(Keys[i]) {
					return Animation.Frames[i];
				}
			}
			return null;
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

		public void SetKey(BVHFrame frame, bool value) {
			Keys[frame.Index-1] = value;
		}

		public bool IsKey(BVHFrame frame) {
			return Keys[frame.Index-1];
		}

		public void Inspector() {
			Utility.SetGUIColor(Utility.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Phase Function");
				}

				SetKey(Animation.CurrentFrame, EditorGUILayout.Toggle("Key", IsKey(Animation.CurrentFrame)));

				VariablesScroll = EditorGUILayout.BeginScrollView(VariablesScroll, GUILayout.Height(100f));
				for(int i=0; i<Animation.Character.Bones.Length; i++) {
					if(Variables[i]) {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.Green, Utility.Black, TextAnchor.MiddleCenter)) {
							ToggleVariable(i);
						}
					} else {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.Red, Utility.Black, TextAnchor.MiddleCenter)) {
							ToggleVariable(i);
						}
					}
				}
				EditorGUILayout.EndScrollView();

				SetSmoothing(EditorGUILayout.FloatField("Smoothing", Smoothing));
				TimeWindow = EditorGUILayout.Slider("Time Window", TimeWindow, 2f*Animation.FrameTime, Animation.TotalTime);

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", Utility.DarkGrey, Utility.White, TextAnchor.MiddleCenter, 25f, 50f)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, Utility.Black);

				float startTime = Animation.CurrentFrame.Timestamp-TimeWindow/2f;
				float endTime = Animation.CurrentFrame.Timestamp+TimeWindow/2f;
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
				Handles.color = Utility.Cyan;
				for(int i=1; i<elements; i++) {
					Vector3 prevPos = new Vector3((float)(i-1)/(elements-1), Values[i+start-1], 0f);
					Vector3 newPos = new Vector3((float)(i)/(elements-1), Values[i+start], 0f);
					Handles.DrawLine(
						new Vector3(rect.xMin + prevPos.x * rect.width, rect.yMax - prevPos.y * rect.height, 0f), 
						new Vector3(rect.xMin + newPos.x * rect.width, rect.yMax - newPos.y * rect.height, 0f));
				}

				BVHFrame A = GetFirstKey();
				if(A != null) {
					Handles.color = Utility.Magenta;
					Handles.DrawLine(
						new Vector3(rect.xMin + (float)(A.Index-start)/elements * rect.width, rect.yMax, 0f), 
						new Vector3(rect.xMin + (float)(A.Index-start)/elements * rect.width, rect.yMax - rect.height, 0f));
					BVHFrame B = GetNextKey(A);
					while(A != B && B != null) {
						Handles.color = Utility.White;
						Handles.DrawLine(
						new Vector3(rect.xMin + (float)(A.Index-start)/elements * rect.width, rect.yMax, 0f), 
						new Vector3(rect.xMin + (float)(B.Index-start)/elements * rect.width, rect.yMax - rect.height, 0f));
						Handles.color = Utility.Magenta;
						Handles.DrawLine(
							new Vector3(rect.xMin + (float)(B.Index-start)/elements * rect.width, rect.yMax, 0f), 
							new Vector3(rect.xMin + (float)(B.Index-start)/elements * rect.width, rect.yMax - rect.height, 0f));
						A = B;
						B = GetNextKey(A);
					}
				}

				Handles.color = Utility.Red;
				Handles.DrawLine(
					new Vector3(rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width, rect.yMax, 0f), 
					new Vector3(rect.xMin + (float)(Animation.CurrentFrame.Index-start)/elements * rect.width, rect.yMax - rect.height, 0f));
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", Utility.DarkGrey, Utility.White, TextAnchor.MiddleCenter, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();
			}
		}

		private void ComputeFunction() {
			Values = new float[Animation.TotalFrames];
			float min = float.MaxValue;
			float max = float.MinValue;
			for(int i=0; i<Animation.TotalFrames; i++) {
				float[] translationalVelocities = GenerateTranslationalVelocities(Animation.Frames[i], Smoothing);
				float[] rotationalVelocities = GenerateRotationalVelocities(Animation.Frames[i], Smoothing);
				for(int j=0; j<Animation.Character.Bones.Length; j++) {
					if(Variables[j]) {
						Values[i] = Mathf.Max(Values[i], translationalVelocities[j] + Mathf.Deg2Rad*rotationalVelocities[j]);
					}
				}
				Values[i] = Mathf.Max(Values[i], 0f);
				if(Values[i] < min) {
					min = Values[i];
				}
				if(Values[i] > max) {
					max = Values[i];
				}
			}
			for(int i=0; i<Values.Length; i++) {
				Values[i] = Utility.Normalise(Values[i], min, max, 0f, 1f);
			}
		}

		public float[] GenerateTranslationalVelocities(BVHFrame frame, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, frame.Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, frame.Timestamp+smoothing/2f));
			float[] velocities = new float[Animation.Character.Bones.Length];
			for(int v=0; v<Animation.Character.Bones.Length; v++) {
				for(int k=1; k<frames.Length; k++) {
					velocities[v] += (frames[k].Positions[v] - frames[k-1].Positions[v]).magnitude / Animation.FrameTime;
				}
				velocities[v] /= frames.Length;
			}
			return velocities;
		}

		public float[] GenerateRotationalVelocities(BVHFrame frame, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, frame.Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, frame.Timestamp+smoothing/2f));
			float[] velocities = new float[Animation.Character.Bones.Length];
			for(int v=0; v<Animation.Character.Bones.Length; v++) {
				for(int k=1; k<frames.Length; k++) {
					velocities[v] += Quaternion.Angle(frames[k-1].Rotations[v], frames[k].Rotations[v]) / Animation.FrameTime;
				}
				velocities[v] /= frames.Length;
			}
			return velocities;
		}

		public float[] GenerateAccelerations(BVHFrame frame, float smoothing) {
			BVHFrame[] frames = Animation.GetFrames(Mathf.Max(0f, frame.Timestamp-smoothing/2f), Mathf.Min(Animation.TotalTime, frame.Timestamp+smoothing/2f));
			float[] accelerations = new float[Animation.Character.Bones.Length];
			for(int v=0; v<Animation.Character.Bones.Length; v++) {
				for(int k=1; k<frames.Length; k++) {
					accelerations[v] += (frames[k].Velocities[v] - frames[k-1].Velocities[v]).magnitude / Animation.FrameTime;
				}
				accelerations[v] /= frames.Length;
			}
			return accelerations;
		}
	}

	[System.Serializable]
	public class BVHStyleFunction {
		public BVHAnimation Animation;

		public bool[] Keys;
		public BVHStyle[] Styles;

		public BVHStyleFunction(BVHAnimation animation) {
			Animation = animation;

			Keys = new bool[animation.TotalFrames];
			Keys[0] = true;
			Keys[animation.TotalFrames-1] = true;
			Styles = new BVHStyle[0];
		}

		public void AddStyle() {
			System.Array.Resize(ref Styles, Styles.Length+1);
			Styles[Styles.Length-1] = new BVHStyle("Style", Animation.TotalFrames);
		}

		public void RemoveStyle() {
			if(Styles.Length == 0) {
				return;
			}
			System.Array.Resize(ref Styles, Styles.Length-1);
		}

		public BVHFrame GetFirstKey() {
			return Animation.Frames[0];
		}

		public BVHFrame GetLastKey() {
			return Animation.Frames[Animation.TotalFrames-1];
		}

		public BVHFrame GetPreviousKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index-1; i>=1; i--) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return GetFirstKey();
		}

		public BVHFrame GetNextKey(BVHFrame frame) {
			if(frame != null) {
				for(int i=frame.Index+1; i<=Animation.TotalFrames; i++) {
					if(Keys[i-1]) {
						return Animation.Frames[i-1];
					}
				}
			}
			return GetLastKey();
		}

		public void SetKey(BVHFrame frame, bool value) {
			if(Keys[frame.Index-1] != value) {
				Keys[frame.Index-1] = value;
				for(int i=0; i<Styles.Length; i++) {
					Interpolate(i, frame);
				}
			}
		}

		public bool IsKey(BVHFrame frame) {
			return Keys[frame.Index-1];
		}

		public void SetValue(int dimension, BVHFrame frame, float value) {
			if(Styles[dimension].Values[frame.Index-1] != value) {
				Styles[dimension].Values[frame.Index-1] = value;
				Interpolate(dimension, frame);
			}
		}

		public float GetValue(int dimension, BVHFrame frame) {
			return Styles[dimension].Values[frame.Index-1];
		}

		private void Interpolate(int dimension, BVHFrame frame) {
			if(IsKey(frame)) {
				Styles[dimension].Interpolate(GetPreviousKey(frame), frame);
				Styles[dimension].Interpolate(frame, GetNextKey(frame));
			} else {
				Styles[dimension].Interpolate(GetPreviousKey(frame), GetNextKey(frame));
			}
		}

		public void Inspector() {
			Utility.SetGUIColor(Utility.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Style Function");
				}

				if(Animation.CurrentFrame == GetFirstKey() || Animation.CurrentFrame == GetLastKey()) {
					EditorGUI.BeginDisabledGroup(true);
					SetKey(Animation.CurrentFrame, EditorGUILayout.Toggle("Key", IsKey(Animation.CurrentFrame)));
					EditorGUI.EndDisabledGroup();
				} else {
					SetKey(Animation.CurrentFrame, EditorGUILayout.Toggle("Key", IsKey(Animation.CurrentFrame)));
				}

				for(int i=0; i<Styles.Length; i++) {
					EditorGUILayout.BeginHorizontal();
					Styles[i].Name = EditorGUILayout.TextField(Styles[i].Name, GUILayout.Width(150f));
					if(IsKey(Animation.CurrentFrame)) {
						SetValue(i, Animation.CurrentFrame, EditorGUILayout.Slider(GetValue(i, Animation.CurrentFrame), 0f, 1f));
					} else {
						EditorGUI.BeginDisabledGroup(true);
						SetValue(i, Animation.CurrentFrame, EditorGUILayout.Slider(GetValue(i, Animation.CurrentFrame), 0f, 1f));
						EditorGUI.EndDisabledGroup();
					}
					EditorGUILayout.EndHorizontal();
				}
				

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Add Style", Utility.DarkGrey, Utility.White, TextAnchor.MiddleCenter)) {
					AddStyle();
				}
				if(Utility.GUIButton("Remove Style", Utility.DarkGrey, Utility.White, TextAnchor.MiddleCenter)) {
					RemoveStyle();
				}
				EditorGUILayout.EndHorizontal();

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", Utility.DarkGrey, Utility.White, TextAnchor.MiddleCenter, 25f, 50f)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));

				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, Utility.Black);

				Color[] colors = Utility.GetRainbowColors(Styles.Length);

				BVHFrame A = GetFirstKey();
				BVHFrame B = GetNextKey(A);
				while(A != B) {
					for(int i=0; i<Styles.Length; i++) {
						Handles.color = colors[i];
						Vector3 prevPos = new Vector3((float)A.Index/Animation.TotalFrames, Styles[i].Values[A.Index-1], 0);
						Vector3 newPos = new Vector3((float)B.Index/Animation.TotalFrames, Styles[i].Values[B.Index-1], 0f);
						prevPos = new Vector3(rect.xMin + prevPos.x * rect.width, rect.yMax - prevPos.y * rect.height, 0f);
						newPos = new Vector3(rect.xMin + newPos.x * rect.width, rect.yMax - newPos.y * rect.height, 0f);
						Handles.DrawLine(prevPos, newPos);
					}
					Handles.color = Utility.Magenta;
					Handles.DrawLine(
					new Vector3(rect.xMin + (float)A.Index/Animation.TotalFrames * rect.width, rect.yMax, 0f), 
					new Vector3(rect.xMin + (float)A.Index/Animation.TotalFrames * rect.width, rect.yMax - rect.height, 0f));
					A = B;
					B = GetNextKey(A);
				}
				Handles.color = Utility.Magenta;
				Handles.DrawLine(
				new Vector3(rect.xMin + (float)B.Index/Animation.TotalFrames * rect.width, rect.yMax, 0f), 
				new Vector3(rect.xMin + (float)B.Index/Animation.TotalFrames * rect.width, rect.yMax - rect.height, 0f));

				Handles.color = Utility.Red;
				Handles.DrawLine(
					new Vector3(rect.xMin + (float)Animation.CurrentFrame.Index/Animation.TotalFrames * rect.width, rect.yMax, 0f), 
					new Vector3(rect.xMin + (float)Animation.CurrentFrame.Index/Animation.TotalFrames * rect.width, rect.yMax - rect.height, 0f));

				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", Utility.DarkGrey, Utility.White, TextAnchor.MiddleCenter, 25f, 50f)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();

			}
		}
	}

	[System.Serializable]
	public class BVHStyle {
		public string Name;
		public float[] Values;

		public BVHStyle(string name, int length) {
			Name = name;
			Values = new float[length];
		}

		public void Interpolate(BVHFrame a, BVHFrame b) {
			if(a == null || b == null) {
				Debug.Log("A given frame was null.");
				return;
			}
			int dist = b.Index - a.Index;
			if(dist >= 2) {
				for(int i=a.Index+1; i<b.Index; i++) {
					float rateA = (float)((float)i-(float)a.Index)/(float)dist;
					float rateB = (float)((float)b.Index-(float)i)/(float)dist;
					Values[i-1] = rateB*Values[a.Index-1] + rateA*Values[b.Index-1];
				}
			}
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
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.Green, Utility.Black, TextAnchor.MiddleCenter)) {
							RemoveContact(i);
						}
					} else {
						if(Utility.GUIButton(Animation.Character.Bones[i].GetName(), Utility.Red, Utility.Black, TextAnchor.MiddleCenter)) {
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

		//TODO BETTER!
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
					Velocities[i] = (Positions[i] - Animation.GetFrame(Index-1).Positions[i]) / Animation.FrameTime;
				}
			}
		}
	}

}