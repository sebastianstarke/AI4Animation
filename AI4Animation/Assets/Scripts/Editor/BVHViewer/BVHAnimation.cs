using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public class BVHAnimation : ScriptableObject {

	public Character Character;
	public Vector3 ForwardOrientation = Vector3.zero;

	public bool Preview = false;
	
	public BVHFrame[] Frames = new BVHFrame[0];
	public BVHFrame CurrentFrame = null;
	public int TotalFrames = 0;
	public float TotalTime = 0f;
	public float FrameTime = 0f;
	public float PlayTime = 0f;
	public bool Playing = false;

	public BVHFunction PhaseFunction;
	public BVHFunction StyleFunction;

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
		PhaseFunction = new BVHFunction(this, "Phase Function", new string[1]{"Phase"});
		StyleFunction = new BVHFunction(this, "Style Function", new string[5]{"Idle", "Walk", "Run", "Jump", "Sit"});
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
		return GetFrame(Mathf.Min(Mathf.FloorToInt(time / FrameTime) + 1, TotalFrames));
	}

	public void Draw() {
		if(Preview) {
			float step = 1f;
			BVHAnimation.BVHFrame frame = CurrentFrame;
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
		CurrentFrame.GenerateTrajectory().Draw();
		Character.Draw();
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

	[System.Serializable]
	public class BVHFunction {
		public BVHAnimation Animation;

		public string Name;
		public bool[] Keys;
		public BVHDimension[] Dimensions; //All values are between 0 and 1

		public BVHFunction(BVHAnimation animation, string name, string[] dimensions) {
			Animation = animation;
			Name = name;
			Keys = new bool[animation.TotalFrames];
			Keys[0] = true;
			Keys[animation.TotalFrames-1] = true;
			Dimensions = new BVHDimension[dimensions.Length];
			for(int i=0; i<Dimensions.Length; i++) {
				Dimensions[i] = new BVHDimension(dimensions[i], animation.TotalFrames);
			}
		}
		
		public BVHFrame GetFirstKey() {
			return Animation.GetFrame(1);
		}

		public BVHFrame GetLastKey() {
			return Animation.GetFrame(Animation.TotalFrames);
		}

		public BVHFrame GetPreviousKey(BVHFrame frame) {
			for(int i=frame.Index-1; i>=1; i--) {
				if(IsKey(Animation.GetFrame(i))) {
					return Animation.GetFrame(i);
				}
			}
			return GetFirstKey();
		}

		public BVHFrame GetNextKey(BVHFrame frame) {
			for(int i=frame.Index+1; i<=Animation.TotalFrames; i++) {
				if(IsKey(Animation.GetFrame(i))) {
					return Animation.GetFrame(i);
				}
			}
			return GetLastKey();
		}

		public void SetValue(int dimension, BVHFrame frame, float value) {
			if(Dimensions[dimension].Values[frame.Index-1] != value) {
				Dimensions[dimension].Values[frame.Index-1] = value;
				Interpolate(dimension, frame);
			}
		}

		public float GetValue(int dimension, BVHFrame frame) {
			return Dimensions[dimension].Values[frame.Index-1];
		}

		public void SetKey(BVHFrame frame, bool value) {
			if(Keys[frame.Index-1] != value) {
				Keys[frame.Index-1] = value;
				for(int i=0; i<Dimensions.Length; i++) {
					Interpolate(i, frame);
				}
			}
		}

		public bool IsKey(BVHFrame frame) {
			return Keys[frame.Index-1];
		}

		private void Interpolate(int dimension, BVHFrame frame) {
			if(IsKey(frame)) {
				Interpolate(dimension, GetPreviousKey(frame), frame);
				Interpolate(dimension, frame, GetNextKey(frame));
			} else {
				Interpolate(dimension, GetPreviousKey(frame), GetNextKey(frame));
			}
		}

		private void Interpolate(int dimension, BVHFrame a, BVHFrame b) {
			if(a == null || b == null) {
				Debug.Log("A given frame was null.");
				return;
			}
			int dist = b.Index - a.Index - 1;
			if(dist != 0) {
				for(int i=a.Index+1; i<b.Index; i++) {
					float rateA = (float)((float)i-(float)a.Index)/(float)dist;
					float rateB = (float)((float)b.Index-(float)i)/(float)dist;
					Dimensions[dimension].Values[i-1] = rateB*Dimensions[dimension].Values[a.Index-1] + rateA*Dimensions[dimension].Values[b.Index-1];
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
					EditorGUILayout.LabelField(Name);
				}

				if(Animation.CurrentFrame == GetFirstKey() || Animation.CurrentFrame == GetLastKey()) {
					EditorGUI.BeginDisabledGroup(true);
					SetKey(Animation.CurrentFrame, EditorGUILayout.Toggle("Key", IsKey(Animation.CurrentFrame)));
					EditorGUI.EndDisabledGroup();
				} else {
					SetKey(Animation.CurrentFrame, EditorGUILayout.Toggle("Key", IsKey(Animation.CurrentFrame)));
				}

				if(!IsKey(Animation.CurrentFrame)) {
					EditorGUI.BeginDisabledGroup(true);
					for(int i=0; i<Dimensions.Length; i++) {
						SetValue(i, Animation.CurrentFrame, EditorGUILayout.Slider(Dimensions[i].Name, GetValue(i, Animation.CurrentFrame), 0f, 1f));
					}
					EditorGUI.EndDisabledGroup();
				} else {
					for(int i=0; i<Dimensions.Length; i++) {
						SetValue(i, Animation.CurrentFrame, EditorGUILayout.Slider(Dimensions[i].Name, GetValue(i, Animation.CurrentFrame), 0f, 1f));
					}
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, Color.black);
				
				//All values between 0 and 1
				/*
				Handles.color = Color.green;
				for(int i=1; i<Animation.TotalFrames; i++) {
					Vector3 prevPos = new Vector3((float)(i-1)/Animation.TotalFrames, Values[i-1], 0);
					Vector3 newPos = new Vector3((float)i/Animation.TotalFrames, Values[i], 0f);
					Handles.DrawLine(
						new Vector3(rect.xMin + prevPos.x * rect.width, rect.yMax - prevPos.y * rect.height, 0f), 
						new Vector3(rect.xMin + newPos.x * rect.width, rect.yMax - newPos.y * rect.height, 0f));
				}
				*/

				Color[] colors = Utility.GetRainbowColors(Dimensions.Length);

				BVHFrame A = GetFirstKey();
				BVHFrame B = GetNextKey(A);
				while(A != B) {
					for(int i=0; i<Dimensions.Length; i++) {
						Handles.color = colors[i];
						Vector3 prevPos = new Vector3((float)(A.Index-1)/Animation.TotalFrames, Dimensions[i].Values[A.Index-1], 0);
						Vector3 newPos = new Vector3((float)(B.Index-1)/Animation.TotalFrames, Dimensions[i].Values[B.Index-1], 0f);
						prevPos = new Vector3(rect.xMin + prevPos.x * rect.width, rect.yMax - prevPos.y * rect.height, 0f);
						newPos = new Vector3(rect.xMin + newPos.x * rect.width, rect.yMax - newPos.y * rect.height, 0f);
						Handles.DrawLine(prevPos, newPos);
					}
					A = B;
					B = GetNextKey(A);
				}
				Handles.color = Color.red;
				Handles.DrawLine(
					new Vector3(rect.xMin + (float)Animation.CurrentFrame.Index/Animation.TotalFrames * rect.width, rect.yMax - 0f * rect.height, 0f), 
					new Vector3(rect.xMin + (float)Animation.CurrentFrame.Index/Animation.TotalFrames * rect.width, rect.yMax - 1f * rect.height, 0f));
				EditorGUILayout.EndVertical();
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Previous Key", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Animation.LoadFrame(GetPreviousKey(Animation.CurrentFrame));
				}
				if(Utility.GUIButton("Next Key", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Animation.LoadFrame(GetNextKey(Animation.CurrentFrame));
				}
				EditorGUILayout.EndHorizontal();
			}
		}

		[System.Serializable]
		public class BVHDimension {
			public string Name;
			public float[] Values;
			public BVHDimension(string name, int length) {
				Name = name;
				Values = new float[length];
			}
		}
	}

	[System.Serializable]
	public class BVHFrame {
		public BVHAnimation Animation;

		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public BVHFrame(BVHAnimation animation, Transformation[] zeros, int[][] channels, float[] motion, int index, float timestamp, float unitScale) {
			Animation = animation;
			Index = index;
			Timestamp = timestamp;
			Positions = new Vector3[Animation.Character.Bones.Length];
			Rotations = new Quaternion[Animation.Character.Bones.Length];
			
			//Forward Kinematics
			int channel = 0;
			for(int i=0; i<Animation.Character.Bones.Length; i++) {
				if(i == 0) {
					//Root
					Vector3 position = new Vector3(motion[channel+0], motion[channel+1], motion[channel+2]);
					Quaternion rotation =
						GetAngleAxis(motion[channel+3], channels[i][4]) *
						GetAngleAxis(motion[channel+4], channels[i][5]) *
						GetAngleAxis(motion[channel+5], channels[i][6]);
					Animation.Character.Bones[i].SetPosition(position / unitScale);
					Animation.Character.Bones[i].SetRotation(rotation);
					channel += 6;
				} else {
					//Childs
					Quaternion rotation =
						GetAngleAxis(motion[channel+0], channels[i][1]) *
						GetAngleAxis(motion[channel+1], channels[i][2]) *
						GetAngleAxis(motion[channel+2], channels[i][3]);
					Animation.Character.Bones[i].SetPosition(Animation.Character.Bones[i].GetParent(Animation.Character).GetPosition() + Animation.Character.Bones[i].GetParent(Animation.Character).GetRotation() * zeros[i].Position / unitScale);
					Animation.Character.Bones[i].SetRotation(Animation.Character.Bones[i].GetParent(Animation.Character).GetRotation() * zeros[i].Rotation * rotation);
					channel += 3;
				}
			}
			for(int i=0; i<Animation.Character.Bones.Length; i++) {
				Positions[i] = Animation.Character.Bones[i].GetPosition();
				Rotations[i] = Animation.Character.Bones[i].GetRotation();
			}
		}

		public Trajectory GenerateTrajectory() {
			Trajectory trajectory = new Trajectory();
			//Past
			int past = trajectory.GetPastPoints() * trajectory.GetDensity();
			for(int i=0; i<past; i+=trajectory.GetDensity()) {
				float timestamp = Mathf.Clamp(Timestamp - 1f + (float)i/(float)past, 0f, Animation.TotalTime);
				trajectory.Points[i].SetPosition(Animation.GetFrame(timestamp).Positions[0]);
				Vector3 direction = Animation.GetFrame(timestamp).Rotations[0] * Quaternion.Euler(Animation.ForwardOrientation) * Vector3.forward;
				direction.y = 0f;
				direction = direction.normalized;
				trajectory.Points[i].SetDirection(direction);
			}
			//Current
			trajectory.Points[past].SetPosition(Positions[0]);
			Vector3 dir = Rotations[0] * Quaternion.Euler(Animation.ForwardOrientation) * Vector3.forward;
			dir.y = 0f;
			dir = dir.normalized;
			trajectory.Points[past].SetDirection(dir);
			//Future
			int future = trajectory.GetFuturePoints() * trajectory.GetDensity();
			for(int i=trajectory.GetDensity(); i<=future; i+=trajectory.GetDensity()) {
				float timestamp = Mathf.Clamp(Timestamp + (float)i/(float)future, 0f, Animation.TotalTime);
				trajectory.Points[past+i].SetPosition(Animation.GetFrame(timestamp).Positions[0]);
				Vector3 direction = Animation.GetFrame(timestamp).Rotations[0] * Quaternion.Euler(Animation.ForwardOrientation) * Vector3.forward;
				direction.y = 0f;
				direction = direction.normalized;
				trajectory.Points[past+i].SetDirection(direction);
			}
			return trajectory;
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