using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHViewer : EditorWindow {

	private static EditorWindow Window;

	public float UnitScale = 10f;
	public string Path = string.Empty;

	public Character Character;

	public Frame[] Frames = new Frame[0];
	public Frame CurrentFrame = null;

	public int TotalFrames = 0;
	public float TotalTime = 0f;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public int Framerate = 60;
	public bool Loaded = false;
	public bool Playing = false;
	public bool Preview = false;

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
			ShowFrame(PlayTime);
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

			if(!Loaded) {
				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("No animation loaded.");
				}
				return;
			}

			using(new EditorGUILayout.VerticalScope ("Box")) {

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Viewer");
				}

				Character.Inspector();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Total Frames: " + TotalFrames, GUILayout.Width(140f));
				EditorGUILayout.LabelField("Total Time: " + TotalTime + "s", GUILayout.Width(140f));
				EditorGUILayout.LabelField("Frame Time: " + FrameTime + "s" + " (" + (1f/FrameTime).ToString("F1") + "Hz)", GUILayout.Width(200f));
				EditorGUILayout.LabelField("Preview", GUILayout.Width(50f));
				Preview = EditorGUILayout.Toggle(Preview, GUILayout.Width(20f));
				EditorGUILayout.EndHorizontal();
				Framerate = Mathf.Max(1, EditorGUILayout.IntField("Framerate", Framerate));

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
					ShowFrame(frame.Index);
				}
				EditorGUILayout.LabelField(CurrentFrame.Timestamp.ToString() + "s", GUILayout.Width(100f));
				EditorGUILayout.EndHorizontal();

			}

			using(new EditorGUILayout.VerticalScope ("Box")) {

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Processing");

					if(Utility.GUIButton("Export Skeleton", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						ExportSkeleton(Character.GetRoot(), null);
					}

				}


			}

		}
	}

	void OnSceneGUI(SceneView view) {
		if(Loaded) {

			if(Preview) {
				Frame frame = CurrentFrame;
				float step = 1f;
				UnityGL.Start();
				for(int i=2; i<=TotalFrames; i++) {
					UnityGL.DrawLine(GetFrame(i-1).Positions[0], GetFrame(i).Positions[0], Color.magenta);
				}
				UnityGL.Finish();
				for(float i=0f; i<=TotalTime; i+=step) {
					ShowFrame(i);
					Color boneColor = Color.Lerp(Color.red, Color.green, i/TotalTime);
					boneColor.a = 1f/3f;
					Color jointColor = Color.black;
					jointColor.a = 1f/3f;
					DrawSimple(Character.GetRoot(), boneColor, jointColor);
				}
				ShowFrame(frame);
			}

			Character.Draw();

		}
	}

	private void DrawSimple(Character.Bone bone, Color boneColor, Color jointColor) {;
		Handles.color = boneColor;
		for(int i=0; i<bone.GetChildCount(); i++) {
			Handles.DrawLine(bone.GetPosition(), bone.GetChild(Character, i).GetPosition());
		}
		Handles.color = jointColor;
		Handles.SphereHandleCap(0, bone.GetPosition(), bone.GetRotation(), Character.BoneSize, EventType.repaint);
		Handles.color = Color.white;
		for(int i=0; i<bone.GetChildCount(); i++) {
			DrawSimple(bone.GetChild(Character, i), boneColor, jointColor);
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
			Frames[i] = new Frame(Character, zero.ToArray(), channels.ToArray(), motions[i], i+1, i*FrameTime, UnitScale);
		}

		//Load First Frame
		ShowFrame(1);

		Loaded = true;
	}

	public void Unload() {
		if(!Loaded) {
			return;
		}
		Character.Clear();
		System.Array.Resize(ref Frames, 0);
		CurrentFrame = null;

		TotalFrames = 0;
		TotalTime = 0f;
		FrameTime = 0f;
		PlayTime = 0f;
		Framerate = 60;

		Loaded = false;
		Playing = false;
		Preview = false;
	}

	public void Play() {
		Timestamp = Utility.GetTimestamp();
		Playing = true;
	}

	public void Stop() {
		Playing = false;
	}

	public void ShowFrame(Frame Frame) {
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

	public void ShowFrame(int index) {
		ShowFrame(GetFrame(index));
	}

	public void ShowFrame(float time) {
		ShowFrame(GetFrame(time, Framerate));
	}

	public Frame GetFrame(int index) {
		if(index < 1 || index > TotalFrames) {
			Debug.Log("Please specify an index between 1 and " + TotalFrames + ".");
			return null;
		}
		return Frames[index-1];
	}

	public Frame GetFrame(float time, float framerate) {
		if(time < 0f || time > TotalTime) {
			Debug.Log("Please specify a time between 0 and " + TotalTime + ".");
			return null;
		}
		float overflow = Mathf.Repeat(time, 1f/framerate);
		time -= overflow;
		return GetFrame(Mathf.Min(Mathf.FloorToInt(time / FrameTime) + 1, TotalFrames));
	}

	[System.Serializable]
	public class Frame {
		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public Frame(Character Character, Transformation[] zeros, int[][] channels, float[] motion, int index, float timestamp, float unitScale) {
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
					Character.Bones[i].SetPosition(Character.Bones[i].GetParent(Character).GetPosition() + Character.Bones[i].GetParent(Character).GetRotation() * zeros[i].Position / unitScale);
					Character.Bones[i].SetRotation(Character.Bones[i].GetParent(Character).GetRotation() * zeros[i].Rotation * rotation);
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

}
