using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHViewer : EditorWindow {

	private static EditorWindow Window;

	//public float Framerate = 60f;
	public float UnitScale = 10f;
	public string Path = string.Empty;

	public Character Character;

	public Keyframe[] Keyframes = new Keyframe[0];
	public Keyframe CurrentKeyframe = null;

	public int TotalFrames = 0;
	public float TotalTime = 0f;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public bool Loaded = false;
	public bool Playing = false;

	public bool ShowCapture = false;

	public System.DateTime Timestamp;

	[MenuItem ("Addons/BVH Viewer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHViewer));
		
		Vector2 size = new Vector2(600f, 400f);
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
		if(Playing) {
			PlayTime += (float)Utility.GetElapsedTime(Timestamp);
			Timestamp = Utility.GetTimestamp();
			if(PlayTime > TotalTime) {
				PlayTime -= TotalTime;
			}
			ShowKeyframe(PlayTime);
		}
	}

	void OnGUI() {
		Inspect();
	}

	void OnSceneGUI(SceneView view) {
		if(Loaded) {
			Character.Draw();
		}
	}

	public void Load(string path) {
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
						new Vector3(ReadFloat(entries[entry+1]), ReadFloat(entries[entry+2]), ReadFloat(entries[entry+3])), 
						Quaternion.identity
						));
					break;
				} else if(entries[entry].Contains("CHANNELS")) {
					int dimensions = ReadInt(entries[entry+1]);
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
		TotalFrames = ReadInt(lines[index].Substring(8));

		//Read frame time
		index += 1;
		FrameTime = ReadFloat(lines[index].Substring(12));

		//Compute total time
		TotalTime = TotalFrames * FrameTime;

		//Read motions
		index += 1;
		float[][] motions = new float[TotalFrames][];
		for(int i=index; i<lines.Length; i++) {
			motions[i-index] = ReadArray(lines[i]);
		}

		//Build Keyframes
		System.Array.Resize(ref Keyframes, TotalFrames);
		for(int i=0; i<TotalFrames; i++) {
			Keyframes[i] = new Keyframe(Character, zero, channels, motions[i], i+1, i*FrameTime, UnitScale);
		}

		//Load First Keyframe
		ShowKeyframe(1);

		Loaded = true;
	}

	public void Unload() {
		if(!Loaded) {
			return;
		}
		Character.Clear();
		System.Array.Resize(ref Keyframes, 0);
		CurrentKeyframe = null;

		TotalFrames = 0;
		TotalTime = 0f;
		FrameTime = 0f;
		PlayTime = 0f;

		Loaded = false;
		Playing = false;
		ShowCapture = false;
	}

	public void Play() {
		Timestamp = Utility.GetTimestamp();
		Playing = true;
	}

	public void Stop() {
		Playing = false;
	}

	public void ShowKeyframe(Keyframe keyframe) {
		if(keyframe == null) {
			return;
		}
		if(CurrentKeyframe == keyframe) {
			return;
		}
		CurrentKeyframe = keyframe;
		for(int i=0; i<Character.Bones.Length; i++) {
			Character.Bones[i].SetPosition(CurrentKeyframe.Positions[i]);
			Character.Bones[i].SetRotation(CurrentKeyframe.Rotations[i]);
		}
		SceneView.RepaintAll();
		Repaint();
	}

	public void ShowKeyframe(int index) {
		ShowKeyframe(GetKeyframe(index));
	}

	public void ShowKeyframe(float time) {
		ShowKeyframe(GetKeyframe(time));
	}

	public Keyframe GetKeyframe(int index) {
		if(index < 1 || index > TotalFrames) {
			Debug.Log("Please specify an index between 1 and " + TotalFrames + ".");
			return null;
		}
		return Keyframes[index-1];
	}

	public Keyframe GetKeyframe(float time) {
		if(time < 0f || time > TotalTime) {
			Debug.Log("Please specify a time between 0 and " + TotalTime + ".");
			return null;
		}
		return GetKeyframe(Mathf.Min(Mathf.RoundToInt(time / FrameTime) + 1, TotalFrames));
	}

	public int ReadInt(string value) {
		value = FilterValueField(value);
		return ParseInt(value);
	}

	public float ReadFloat(string value) {
		value = FilterValueField(value);
		return ParseFloat(value);
	}

	public float[] ReadArray(string value) {
		value = FilterValueField(value);
		if(value.StartsWith(" ")) {
			value = value.Substring(1);
		}
		if(value.EndsWith(" ")) {
			value = value.Substring(0, value.Length-1);
		}
		string[] values = value.Split(' ');
		float[] array = new float[values.Length];
		for(int i=0; i<array.Length; i++) {
			array[i] = ParseFloat(values[i]);
		}
		return array;
	}

	public Vector3 ReadVector3(string value) {
		value = FilterValueField(value);
		string[] values = value.Split(' ');
		float x = ParseFloat(values[0]);
		float y = ParseFloat(values[1]);
		float z = ParseFloat(values[2]);
		return new Vector3(x,y,z);
	}

	public Vector4 ReadColor(string value) {
		value = FilterValueField(value);
		string[] values = value.Split(' ');
		float r = ParseFloat(values[0]);
		float g = ParseFloat(values[1]);
		float b = ParseFloat(values[2]);
		float a = ParseFloat(values[3]);
		return new Color(r,g,b,a);
	}

	public string FilterValueField(string value) {
		while(value.Contains("  ")) {
			value = value.Replace("  "," ");
		}
		while(value.Contains("< ")) {
			value = value.Replace("< ","<");
		}
		while(value.Contains(" >")) {
			value = value.Replace(" >",">");
		}
		while(value.Contains(" .")) {
			value = value.Replace(" ."," 0.");
		}
		while(value.Contains(". ")) {
			value = value.Replace(". ",".0");
		}
		while(value.Contains("<.")) {
			value = value.Replace("<.","<0.");
		}
		return value;
	}

	public int ParseInt(string value) {
		int parsed = 0;
		if(int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + "!");
			return 0;
		}
	}

	public float ParseFloat(string value) {
		float parsed = 0f;
		if(float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + "!");
			return 0f;
		}
	}

	[System.Serializable]
	public class Keyframe {
		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public Keyframe(Character Character, List<Transformation> zero, List<int[]> channels, float[] motion, int index, float timestamp, float unitScale) {
			Index = index;
			Timestamp = timestamp;
			Positions = new Vector3[Character.Bones.Length];
			Rotations = new Quaternion[Character.Bones.Length];
			//Forward Kinematics
			int channel = 0;
			for(int i=0; i<Character.Bones.Length; i++) {
				if(i == 0) {
					Vector3 position = new Vector3(motion[channel+0], motion[channel+1], motion[channel+2]);
					Quaternion rotation =
						GetAngleAxis(motion[channel+3], channels[i][4]) *
						GetAngleAxis(motion[channel+4], channels[i][5]) *
						GetAngleAxis(motion[channel+5], channels[i][6]);
					Character.Bones[i].SetPosition(position / unitScale);
					Character.Bones[i].SetRotation(rotation);
					channel += 6;
				} else {
					Quaternion rotation =
						GetAngleAxis(motion[channel+0], channels[i][1]) *
						GetAngleAxis(motion[channel+1], channels[i][2]) *
						GetAngleAxis(motion[channel+2], channels[i][3]);
					Character.Bones[i].SetPosition(Character.Bones[i].GetParent(Character).GetPosition() + Character.Bones[i].GetParent(Character).GetRotation() * zero[i].Position / unitScale);
					Character.Bones[i].SetRotation(Character.Bones[i].GetParent(Character).GetRotation() * zero[i].Rotation * rotation);
					channel += 3;
				}
			}
			for(int i=0; i<Character.Bones.Length; i++) {
				Positions[i] = Character.Bones[i].GetPosition();
				Rotations[i] = Character.Bones[i].GetRotation();
			}
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

	private void Inspect() {
		Utility.SetGUIColor(Utility.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			using(new EditorGUILayout.VerticalScope ("Box")) {

				using(new EditorGUILayout.VerticalScope ("Box")) {
					//Framerate = EditorGUILayout.FloatField("Framerate", Framerate);
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
					if(GUILayout.Button("X", GUILayout.Width(20))) {
						Unload();
					}
					EditorGUILayout.EndHorizontal();
				}
				
				if(Utility.GUIButton("Import", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Load(Path);
				}

			}

			using(new EditorGUILayout.VerticalScope ("Box")) {
				if(Loaded) {
					Character.Inspector();
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Total Frames: " + TotalFrames, GUILayout.Width(150f));
					EditorGUILayout.LabelField("Total Time: " + TotalTime, GUILayout.Width(150f));
					EditorGUILayout.LabelField("Frame Time: " + FrameTime + "(" + (1f/FrameTime).ToString("F1") + "Hz)", GUILayout.Width(300f));
					EditorGUILayout.EndHorizontal();
					//ShowCapture = EditorGUILayout.Toggle("Show Capture", ShowCapture);
					EditorGUILayout.BeginHorizontal();
					ControlKeyframe();
					EditorGUILayout.LabelField(CurrentKeyframe.Timestamp.ToString() + "s", GUILayout.Width(100f));
					EditorGUILayout.EndHorizontal();
					if(Playing) {
						if(Utility.GUIButton("Stop", Color.white, Color.grey, TextAnchor.MiddleCenter)) {
							Stop();
						}
					} else {
						if(Utility.GUIButton("Play", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
							Play();
						}
					}
					
				} else {
					EditorGUILayout.LabelField("No animation loaded.");
				}
			}
		}
	}

	private void ControlKeyframe() {
		int newIndex = EditorGUILayout.IntSlider(CurrentKeyframe.Index, 1, TotalFrames, GUILayout.Width(500f));
		if(newIndex != CurrentKeyframe.Index) {
			Keyframe frame = GetKeyframe(newIndex);
			PlayTime = frame.Timestamp;
			ShowKeyframe(frame.Index);
		}
	}
}
