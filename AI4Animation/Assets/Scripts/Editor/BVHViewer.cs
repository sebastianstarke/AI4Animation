using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHViewer : EditorWindow {

	public static EditorWindow Window;

	public float UnitScale = 10f;
	public string Path = string.Empty;

	public bool Preview = false;

	public bool Loaded = false;

	public Character Character = null;
	public Vector3 ForwardOrientation = Vector3.zero;
	public float[] PhaseFunction = new float[0];
	
	public Frame[] Frames = new Frame[0];
	public Frame CurrentFrame = null;
	public int TotalFrames = 0;
	public float TotalTime = 0f;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public bool Playing = false;

	public System.DateTime Timestamp;

	[MenuItem ("Addons/BVH Viewer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHViewer));
		
		Vector2 size = new Vector2(600f, 600f);
		Window.minSize = size;
		Window.maxSize = size;
	}

	void Awake() {
		Character = new Character();
	}

	void OnFocus() {
		SceneView.onSceneGUIDelegate -= this.OnSceneGUI;
		SceneView.onSceneGUIDelegate += this.OnSceneGUI;

		Timestamp = Utility.GetTimestamp();
	}

	void OnDestroy() {
		SceneView.onSceneGUIDelegate -= this.OnSceneGUI;
	}

	void Update() {
		if(EditorApplication.isCompiling) {
			Unload();
		}
		if(!Loaded) {
			return;
		}
		if(Playing) {
			PlayTime += (float)Utility.GetElapsedTime(Timestamp);
			Timestamp = Utility.GetTimestamp();
			if(PlayTime > TotalTime) {
				PlayTime -= TotalTime;
			}
			LoadFrame(PlayTime);
		}
	}

	void OnGUI() {
		Utility.SetGUIColor(Utility.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			using(new EditorGUILayout.VerticalScope ("Box")) {

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Importer");
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					UnitScale = EditorGUILayout.FloatField("Unit Scale", UnitScale);
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Path", GUILayout.Width(30));
					Path = EditorGUILayout.TextField(Path);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Path = EditorUtility.OpenFilePanel("BVH Viewer", Path == string.Empty ? Application.dataPath : Path.Substring(0, Path.LastIndexOf("/")), "bvh");
						GUI.SetNextControlName("");
						GUI.FocusControl("");
					}
					EditorGUILayout.EndHorizontal();
				}
				
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Load", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Load(Path);
				}
				if(Utility.GUIButton("Unload", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Unload();
				}
				EditorGUILayout.EndHorizontal();

			}

			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Settings");
				}

				Preview = EditorGUILayout.Toggle("Preview", Preview);
				ForwardOrientation = EditorGUILayout.Vector3Field("Forward Orientation", ForwardOrientation);
			}
			

			if(!Loaded) {
				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("No animation loaded.");
				}
				return;
			}

			Utility.SetGUIColor(Utility.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Character.Inspector();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Total Frames: " + TotalFrames, GUILayout.Width(140f));
				EditorGUILayout.LabelField("Total Time: " + TotalTime + "s", GUILayout.Width(140f));
				EditorGUILayout.LabelField("Frame Time: " + FrameTime + "s" + " (" + (1f/FrameTime).ToString("F1") + "Hz)", GUILayout.Width(200f));
				EditorGUILayout.EndHorizontal();

				EditorGUILayout.BeginHorizontal();
				if(Playing) {
					if(Utility.GUIButton("||", Color.grey, Color.white, TextAnchor.MiddleCenter, 20f, 20f)) {
						Stop();
					}
				} else {
					if(Utility.GUIButton("|>", Color.grey, Color.white, TextAnchor.MiddleCenter, 20f, 20f)) {
						Play();
					}
				}
				int newIndex = EditorGUILayout.IntSlider(CurrentFrame.Index, 1, TotalFrames, GUILayout.Width(470f));
				if(newIndex != CurrentFrame.Index) {
					Frame frame = GetFrame(newIndex);
					PlayTime = frame.Timestamp;
					LoadFrame(frame.Index);
				}
				EditorGUILayout.LabelField(CurrentFrame.Timestamp.ToString() + "s", GUILayout.Width(100f));
				EditorGUILayout.EndHorizontal();

				DrawFunctions();

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Previous Keyframe", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Frame prev = GetPreviousKeyframe(CurrentFrame);
					if(prev != null) {
						LoadFrame(prev);
					}
				}
				if(Utility.GUIButton("Next Keyframe", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Frame next = GetNextKeyframe(CurrentFrame);
					if(next != null) {
						LoadFrame(next);
					}
				}
				EditorGUILayout.EndHorizontal();

				CurrentFrame.SetKeyframe(EditorGUILayout.Toggle("Keyframe", CurrentFrame.IsKeyframe));

				if(!CurrentFrame.IsKeyframe) {
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.Slider("Phase", CurrentFrame.Phase, 0f, 1f);
					EditorGUI.EndDisabledGroup();
				} else {
					CurrentFrame.Interpolate(EditorGUILayout.Slider("Phase", CurrentFrame.Phase, 0f, 1f));
				}

				if(Utility.GUIButton("Export Skeleton", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					ExportSkeleton(Character.GetRoot(), null);
				}

				//if(Utility.GUIButton("Compute Foot Contacts", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					//ComputeFootContacts(); //TODO
				//}

			}

		}
	}

	void OnSceneGUI(SceneView view) {
		if(Loaded) {

			if(Preview) {
				float step = 1f;
				Frame frame = CurrentFrame;
				UnityGL.Start();
				for(int i=2; i<=TotalFrames; i++) {
					UnityGL.DrawLine(GetFrame(i-1).Positions[0], GetFrame(i).Positions[0], Color.magenta);
				}
				UnityGL.Finish();
				for(float i=0f; i<=TotalTime; i+=step) {
					LoadFrame(i);
					Character.DrawSimple();
				}
				LoadFrame(frame);
			}

			GenerateTrajectory(CurrentFrame).Draw();

			Character.Draw();

		}
	}


	private void Load(string path) {
		Unload();

		string[] lines = File.ReadAllLines(Path);
		char[] whitespace = new char[] {' '};
		int index = 0;

		//Build Hierarchy
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
			Frames[i] = new Frame(this, zero.ToArray(), channels.ToArray(), motions[i], i+1, i*FrameTime, UnitScale);
		}

		//Load First Frame
		LoadFrame(1);

		Loaded = true;
	}

	private void Unload() {
		if(!Loaded) {
			return;
		}
		Loaded = false;
		Character.Clear();
		System.Array.Resize(ref Frames, 0);
		CurrentFrame = null;
		TotalFrames = 0;
		TotalTime = 0f;
		FrameTime = 0f;
		PlayTime = 0f;
		Playing = false;
	}

	private void Play() {
		Timestamp = Utility.GetTimestamp();
		Playing = true;
	}

	private void Stop() {
		Playing = false;
	}

	private void LoadFrame(Frame Frame) {
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
		SceneView.RepaintAll();
		Repaint();
	}

	private void LoadFrame(int index) {
		LoadFrame(GetFrame(index));
	}

	private void LoadFrame(float time) {
		LoadFrame(GetFrame(time));
	}

	private Frame GetFrame(int index) {
		if(index < 1 || index > TotalFrames) {
			Debug.Log("Please specify an index between 1 and " + TotalFrames + ".");
			return null;
		}
		return Frames[index-1];
	}

	private Frame GetFrame(float time) {
		if(time < 0f || time > TotalTime) {
			Debug.Log("Please specify a time between 0 and " + TotalTime + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.FloorToInt(time / FrameTime) + 1, TotalFrames));
	}

	/*
	private Frame GetFirstKeyframe() {
		for(int i=1; i<=TotalFrames; i++) {
			if(GetFrame(i).IsKeyframe) {
				return GetFrame(i);
			}
		}
		return null;
	}

	private Frame GetLastKeyframe() {
		for(int i=TotalFrames; i>=1; i--) {
			if(GetFrame(i).IsKeyframe) {
				return GetFrame(i);
			}
		}
		return null;
	}
	*/

	private Frame GetPreviousKeyframe(Frame frame) {
		for(int i=frame.Index-1; i>=1; i--) {
			if(GetFrame(i).IsKeyframe) {
				return GetFrame(i);
			}
		}
		return null;
	}

	private Frame GetNextKeyframe(Frame frame) {
		for(int i=frame.Index+1; i<=TotalFrames; i++) {
			if(GetFrame(i).IsKeyframe) {
				return GetFrame(i);
			}
		}
		return null;
	}

	private Trajectory GenerateTrajectory(Frame frame) {
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

	private void InterpolatePhase(Frame a, Frame b) {
		if(a == null || b == null) {
			Debug.Log("A given frame was null.");
			return;
		}
		int dist = b.Index - a.Index - 1;
		if(dist != 0) {
			for(int i=a.Index+1; i<b.Index; i++) {
				float rateA = (float)((float)GetFrame(i).Index-(float)a.Index)/(float)dist;
				float rateB = (float)((float)b.Index-(float)GetFrame(i).Index)/(float)dist;
				rateA = rateA * rateA + rateA;
				rateB = rateB * rateB + rateB;
				GetFrame(i).Phase = rateB/(rateA+rateB)*a.Phase + rateA/(rateA+rateB)*b.Phase;
			}
		}
	}

	private void InterpolateGaits(Frame a, Frame b) {
		if(a == null || b == null) {
			Debug.Log("A given frame was null.");
			return;
		}
	}

	private void DrawFunctions() {
		float height = 50f;

		EditorGUILayout.BeginVertical(GUILayout.Height(height));

		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
		EditorGUI.DrawRect(rect, Color.black);

		//All values between 0 and 1
		Handles.color = Color.green;
		for(int i=1; i<Frames.Length; i++) {
			Vector3 prevPos = new Vector3((float)(i-1)/Frames.Length, Frames[i-1].Phase, 0);
			Vector3 newPos = new Vector3((float)i/Frames.Length, Frames[i].Phase, 0f);
			Handles.DrawLine(
				new Vector3(rect.xMin + prevPos.x * rect.width, rect.yMax - prevPos.y * rect.height, 0f), 
				new Vector3(rect.xMin + newPos.x * rect.width, rect.yMax - newPos.y * rect.height, 0f));
		}
		Handles.color = Color.red;
		Handles.DrawLine(
			new Vector3(rect.xMin + (float)CurrentFrame.Index/Frames.Length * rect.width, rect.yMax - 0f * rect.height, 0f), 
			new Vector3(rect.xMin + (float)CurrentFrame.Index/Frames.Length * rect.width, rect.yMax - 1f * rect.height, 0f));

		EditorGUILayout.EndVertical();
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

	[System.Serializable]
	public class Frame {
		public BVHViewer Viewer;

		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public bool IsKeyframe;
		public float Phase;

		public Frame(BVHViewer viewer, Transformation[] zeros, int[][] channels, float[] motion, int index, float timestamp, float unitScale) {
			Viewer = viewer;
			Index = index;
			Timestamp = timestamp;
			Positions = new Vector3[Viewer.Character.Bones.Length];
			Rotations = new Quaternion[Viewer.Character.Bones.Length];
			//Forward Kinematics
			int channel = 0;
			for(int i=0; i<Viewer.Character.Bones.Length; i++) {
				if(i == 0) {
					//Root
					Vector3 position = new Vector3(motion[channel+0], motion[channel+1], motion[channel+2]);
					Quaternion rotation =
						GetAngleAxis(motion[channel+3], channels[i][4]) *
						GetAngleAxis(motion[channel+4], channels[i][5]) *
						GetAngleAxis(motion[channel+5], channels[i][6]);
					Viewer.Character.Bones[i].SetPosition(position / unitScale);
					Viewer.Character.Bones[i].SetRotation(rotation);
					channel += 6;
				} else {
					//Childs
					Quaternion rotation =
						GetAngleAxis(motion[channel+0], channels[i][1]) *
						GetAngleAxis(motion[channel+1], channels[i][2]) *
						GetAngleAxis(motion[channel+2], channels[i][3]);
					Viewer.Character.Bones[i].SetPosition(Viewer.Character.Bones[i].GetParent(Viewer.Character).GetPosition() + Viewer.Character.Bones[i].GetParent(Viewer.Character).GetRotation() * zeros[i].Position / unitScale);
					Viewer.Character.Bones[i].SetRotation(Viewer.Character.Bones[i].GetParent(Viewer.Character).GetRotation() * zeros[i].Rotation * rotation);
					channel += 3;
				}
			}
			for(int i=0; i<Viewer.Character.Bones.Length; i++) {
				Positions[i] = Viewer.Character.Bones[i].GetPosition();
				Rotations[i] = Viewer.Character.Bones[i].GetRotation();
			}
			
			IsKeyframe = false;
			Phase = 0f;
		}

		public void SetKeyframe(bool value) {
			if(IsKeyframe == value) {
				return;
			}

			IsKeyframe = value;

			if(!IsKeyframe) {
				Frame prev = Viewer.GetPreviousKeyframe(this);
				if(prev == null) {
					prev = Viewer.GetFrame(1);
				}
				Frame next = Viewer.GetNextKeyframe(this);
				if(next == null) {
					next = Viewer.GetFrame(Viewer.TotalFrames);
				}

				Viewer.InterpolatePhase(prev, next);
			}
		
		}

		public void Interpolate(float phase) {
			if(Phase == phase) {
				return;
			}

			Phase = phase;

			Frame prev = Viewer.GetPreviousKeyframe(this);
			if(prev == null) {
				prev = Viewer.GetFrame(1);
			}
			Frame next = Viewer.GetNextKeyframe(this);
			if(next == null) {
				next = Viewer.GetFrame(Viewer.TotalFrames);
			}

			Viewer.InterpolatePhase(prev, this);
			Viewer.InterpolatePhase(this, next);
			
		}

		private Quaternion GetAngleAxis(float angle, int axis) {
			if(axis == 4) {
				return Quaternion.AngleAxis(angle, Vector3.right);
			}
			if(axis == 5) {
				return Quaternion.AngleAxis(angle, Vector3.up);
			}
			if(axis == 6) {
				return Quaternion.AngleAxis(angle, Vector3.forward);
			}
			return Quaternion.identity;
		}
	}

}
