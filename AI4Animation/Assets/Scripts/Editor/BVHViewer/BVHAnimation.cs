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
	
	public System.DateTime Timestamp;

	public void EditorUpdate() {
		if(Playing) {
			PlayTime += (float)Utility.GetElapsedTime(Timestamp);
			Timestamp = Utility.GetTimestamp();
			if(PlayTime > TotalTime) {
				PlayTime -= TotalTime;
			}
			LoadFrame(PlayTime);
		}
	}

	public BVHAnimation Create(BVHViewer viewer) {
		Load(viewer.Path, viewer.UnitScale);
		AssetDatabase.CreateAsset(this , "Assets/Resources/BVH/MoCap.asset");
		AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
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
		Timestamp = Utility.GetTimestamp();
		Playing = true;
	}

	public void Stop() {
		Playing = false;
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


	public BVHFrame GetPreviousKeyframe(BVHFrame frame) {
		for(int i=frame.Index-1; i>=1; i--) {
			if(GetFrame(i).IsKeyframe) {
				return GetFrame(i);
			}
		}
		return null;
	}

	public BVHFrame GetNextKeyframe(BVHFrame frame) {
		for(int i=frame.Index+1; i<=TotalFrames; i++) {
			if(GetFrame(i).IsKeyframe) {
				return GetFrame(i);
			}
		}
		return null;
	}

	public void InterpolatePhase(BVHFrame a, BVHFrame b) {
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

	public void InterpolateStyles(BVHFrame a, BVHFrame b) {
		if(a == null || b == null) {
			Debug.Log("A given frame was null.");
			return;
		}
	}

	public void DrawPhaseFunction() {
		EditorGUILayout.LabelField("Phase Function");

		EditorGUILayout.BeginVertical(GUILayout.Height(50f));

		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
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

	public void DrawStyleFunction() {
		EditorGUILayout.LabelField("Style Function");

		EditorGUILayout.BeginVertical(GUILayout.Height(50f));

		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, Color.black);

		//TODO

		Handles.color = Color.red;
		Handles.DrawLine(
			new Vector3(rect.xMin + (float)CurrentFrame.Index/Frames.Length * rect.width, rect.yMax - 0f * rect.height, 0f), 
			new Vector3(rect.xMin + (float)CurrentFrame.Index/Frames.Length * rect.width, rect.yMax - 1f * rect.height, 0f));

		EditorGUILayout.EndVertical();
	}

	public void Draw() {
		if(Preview) {
			DrawPreview();
		}
		CurrentFrame.GenerateTrajectory().Draw();
		Character.Draw();
	}

	private void DrawPreview() {
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
	public class BVHFrame {
		public BVHAnimation Animation;

		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public bool IsKeyframe;
		public float Phase;

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
			
			IsKeyframe = false;
			Phase = 0f;
		}

		public void SetKeyframe(bool value) {
			if(IsKeyframe == value) {
				return;
			}

			IsKeyframe = value;

			if(!IsKeyframe) {
				BVHFrame prev = Animation.GetPreviousKeyframe(this);
				if(prev == null) {
					prev = Animation.GetFrame(1);
				}
				BVHFrame next = Animation.GetNextKeyframe(this);
				if(next == null) {
					next = Animation.GetFrame(Animation.TotalFrames);
				}

				Animation.InterpolatePhase(prev, next);
			}
		
		}

		public void Interpolate(float phase) {
			if(Phase == phase) {
				return;
			}

			Phase = phase;

			BVHFrame prev = Animation.GetPreviousKeyframe(this);
			if(prev == null) {
				prev = Animation.GetFrame(1);
			}
			BVHFrame next = Animation.GetNextKeyframe(this);
			if(next == null) {
				next = Animation.GetFrame(Animation.TotalFrames);
			}

			Animation.InterpolatePhase(prev, this);
			Animation.InterpolatePhase(this, next);
			
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
	}

}