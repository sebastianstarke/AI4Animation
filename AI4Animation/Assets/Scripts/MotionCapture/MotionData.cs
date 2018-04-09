using System.IO;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class MotionData : ScriptableObject {

	public enum Axis {X, Y, Z};

	public BVHData Source;

	public string Name = string.Empty;
	public float Framerate = 1f;
	public float UnitScale = 100f;
	public string[] Styles = new string[0];
	public Axis MirrorAxis = Axis.X;
	public Frame[] Frames;
	public int[] Symmetry;

	public float GetTotalTime() {
		return GetTotalFrames() / Framerate;
	}

	public int GetTotalFrames() {
		return Frames.Length;
	}

	public Frame GetFrame(int index) {
		if(index < 1 || index > GetTotalFrames()) {
			Debug.Log("Please specify an index between 1 and " + GetTotalFrames() + ".");
			return null;
		}
		return Frames[index-1];
	}

	public Frame GetFrame(float time) {
		if(time < 0f || time > GetTotalTime()) {
			Debug.Log("Please specify a time between 0 and " + GetTotalTime() + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.RoundToInt(time * Framerate) + 1, GetTotalFrames()));
	}

	public Vector3 GetMirrorAxis() {
		switch(MirrorAxis) {
			case Axis.X:
			return Vector3.right;
			case Axis.Y:
			return Vector3.up;
			case Axis.Z:
			return Vector3.forward;
			default:
			return Vector3.zero;
		}
	}

	public void SetUnitScale(float value) {
		if(UnitScale != value) {
			UnitScale = value;
			Generate();
		}
	}

	public void AddStyle(string name) {
		ArrayExtensions.Add(ref Styles, name);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Add(ref Frames[i].Style, 0);
		}
	}

	public void RemoveStyle() {
		ArrayExtensions.Shrink(ref Styles);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Shrink(ref Frames[i].Style);
		}
	}

	public void RemoveStyle(string name) {
		int index = ArrayExtensions.FindIndex(ref Styles, name);
		if(index >= 0) {
			ArrayExtensions.RemoveAt(ref Styles, index);
			for(int i=0; i<GetTotalFrames(); i++) {
				ArrayExtensions.RemoveAt(ref Frames[i].Style, index);
			}
		}
	}

	public void Load(string path) {
		Name = path.Substring(path.LastIndexOf("/")+1);

		string[] lines = File.ReadAllLines(path);
		char[] whitespace = new char[] {' '};
		int index = 0;

		//Read BVH
		Source = new BVHData();
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
					Source.AddBone(name, parent, offset, new int[0]);
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
					Source.AddBone(name, parent, offset, channels);
					break;
				} else if(entries[entry].Contains("}")) {
					name = parent;
					parent = name == "None" ? "None" : Source.FindBone(name).Parent;
					break;
				}
			}
		}

		//Read frame count
		index += 1;
		while(lines[index].Length == 0) {
			index += 1;
		}
		ArrayExtensions.Resize(ref Frames, Utility.ReadInt(lines[index].Substring(8)));

		//Read frame time
		index += 1;
		Framerate = Mathf.RoundToInt(1f / Utility.ReadFloat(lines[index].Substring(12)));

		//Read motions
		index += 1;
		for(int i=index; i<lines.Length; i++) {
			Source.AddMotion(Utility.ReadArray(lines[i]));
		}

		//Detect settings
		DetectSymmetry();

		//Create frames
		for(int i=0; i<GetTotalFrames(); i++) {
			Frames[i] = new Frame(this, i+1, (float)i / Framerate);
		}

		//Generate
		Generate();
	}

	public void DetectSymmetry() {
		Symmetry = new int[Source.Bones.Length];
		for(int i=0; i<Source.Bones.Length; i++) {
			string name = Source.Bones[i].Name;
			if(name.Contains("Left")) {
				BVHData.Bone bone = Source.FindBone("Right"+name.Substring(4));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(name.Contains("Right")) {
				BVHData.Bone bone = Source.FindBone("Left"+name.Substring(5));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(name.StartsWith("L") && char.IsUpper(name[1])) {
				BVHData.Bone bone = Source.FindBone("R"+name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(name.StartsWith("R") && char.IsUpper(name[1])) {
				BVHData.Bone bone = Source.FindBone("L"+name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else {
				Symmetry[i] = i;
			}
		}
	}

	public void Generate() {
		for(int i=0; i<GetTotalFrames(); i++) {
			Frames[i].Generate();
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
			ArrayExtensions.Add(ref Bones, new Bone(Bones.Length, name, parent, offset, channels));
		}

		public Bone FindBone(string name) {
			return System.Array.Find(Bones, x => x.Name == name);
		}

		public void AddMotion(float[] values) {
			ArrayExtensions.Add(ref Motions, new Motion(values));
		}

		[System.Serializable]
		public class Bone {
			public int Index;
			public string Name;
			public string Parent;
			public Vector3 Offset;
			public int[] Channels;
			public Bone(int index, string name, string parent, Vector3 offset, int[] channels) {
				Index = index;
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
	public class Frame {
		public MotionData Data;
		public int Index;
		public float Timestamp;
		public Matrix4x4[] Local;
		public Matrix4x4[] World;
		public float[] Style;

		public Frame(MotionData data, int index, float timestamp) {
			Data = data;
			Index = index;
			Timestamp = timestamp;
			Local = new Matrix4x4[Data.Source.Bones.Length];
			World = new Matrix4x4[Data.Source.Bones.Length];
			Style = new float[0];
		}

		public void Generate() {
			int channel = 0;
			BVHData.Motion motion = Data.Source.Motions[Index-1];
			for(int i=0; i<Data.Source.Bones.Length; i++) {
				BVHData.Bone info = Data.Source.Bones[i];
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

				position = (position == Vector3.zero ? info.Offset : position) / Data.UnitScale;
				Local[i] = Matrix4x4.TRS(position, rotation, Vector3.one);
				World[i] = info.Parent == "None" ? Local[i] : World[Data.Source.FindBone(info.Parent).Index] * Local[i];
			}
			//for(int i=0; i<Animation.Character.Hierarchy.Length; i++) {
			//	Local[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Animation.Corrections[i]), Vector3.one);
			//	World[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Animation.Corrections[i]), Vector3.one);
			//}
		}

		public Matrix4x4[] GetBoneTransformations(bool mirrored) {
			Matrix4x4[] transformations = new Matrix4x4[World.Length];
			for(int i=0; i<World.Length; i++) {
				transformations[i] = GetBoneTransformation(i, mirrored);
			}
			return transformations;
		}

		public Matrix4x4 GetBoneTransformation(int index, bool mirrored) {
			return mirrored ? World[Data.Symmetry[index]].GetMirror(Data.GetMirrorAxis()) : World[index];
		}

		public Vector3[] GetBoneVelocities(bool mirrored) {
			Vector3[] velocities = new Vector3[World.Length];
			for(int i=0; i<World.Length; i++) {
				velocities[i] = GetBoneVelocity(i, mirrored);
			}
			return velocities;
		}

		public Vector3 GetBoneVelocity(int index, bool mirrored) {
			if(Index == 1) {
				return mirrored ? 
				((Data.GetFrame(Index+1).World[Data.Symmetry[index]].GetPosition() - World[Data.Symmetry[index]].GetPosition()) * Data.Framerate).GetMirror(Data.GetMirrorAxis())
				: 
				(Data.GetFrame(Index+1).World[index].GetPosition() - World[index].GetPosition()) * Data.Framerate;
			} else {
				return mirrored ? 
				((Data.GetFrame(Index+1).World[Data.Symmetry[index]].GetPosition() - World[Data.Symmetry[index]].GetPosition()) * Data.Framerate).GetMirror(Data.GetMirrorAxis())
				: 
				(World[index].GetPosition() - Data.GetFrame(Index-1).World[index].GetPosition()) * Data.Framerate;
			}
		}

		public Matrix4x4 GetRoot(bool mirrored) {
			Matrix4x4 transformation = GetBoneTransformation(0, mirrored);
			Vector3 position = Utility.ProjectGround(transformation.GetPosition(), LayerMask.GetMask("Ground"));
			Vector3 forward = transformation.GetForward();
			forward.y = 0f;
			return Matrix4x4.TRS(position, Quaternion.LookRotation(forward, Vector3.up), Vector3.one);
		}

		public Trajectory GetTrajectory(bool mirrored) {
			Trajectory trajectory = new Trajectory(12, 0);
			for(int i=0; i<6; i++) {
				MotionData.Frame previous = Data.GetFrame(Mathf.Clamp(Timestamp - 1f + (float)i/6f, 0f, Data.GetTotalTime()));
				trajectory.Points[i].SetTransformation(previous.GetRoot(mirrored));
				trajectory.Points[i].SetVelocity(previous.GetBoneVelocity(0, mirrored));
			}
			trajectory.Points[6].SetTransformation(GetRoot(mirrored));
			trajectory.Points[6].SetVelocity(GetBoneVelocity(0, mirrored));
			for(int i=1; i<=5; i++) {
				MotionData.Frame future = Data.GetFrame(Mathf.Clamp(Timestamp + (float)i/5f, 0f, Data.GetTotalTime()));
				trajectory.Points[6+i].SetTransformation(future.GetRoot(mirrored));
				trajectory.Points[6+i].SetVelocity(future.GetBoneVelocity(0, mirrored));
			}
			return trajectory;
		}
	}

}
