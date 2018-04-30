#if UNITY_EDITOR
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class MotionData : ScriptableObject {

	public enum Axis {XPositive, YPositive, ZPositive, XNegative, YNegative,ZNegative};

	public BVHData Source;

	public string Name = string.Empty;
	public float Framerate = 1f;
	public float UnitScale = 100f;
	public string[] Styles = new string[0];
	public float StyleTransition = 0.5f;
	public float MotionSmoothing = 0f;
	public Axis MirrorAxis = Axis.XPositive;
	public int[] Symmetry = new int[0];
	public Vector3[] Corrections = new Vector3[0];
	public LayerMask GroundMask = -1;
	public LayerMask ObjectMask = -1;
	public Axis ForwardAxis = Axis.ZPositive;
	public int HeightMapSensor = 0;
	public float HeightMapSize = 0.25f;
	public int DepthMapSensor = 0;
	public Axis DepthMapAxis = Axis.ZPositive;
	public int DepthMapResolution = 20;
	public float DepthMapSize = 10f;
	public float DepthMapDistance = 10f;
	public Sequence[] Sequences = new Sequence[0];

	public Frame[] Frames;

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

	public Frame[] GetFrames(int start, int end) {
		if(start < 1 || end > GetTotalFrames()) {
			Debug.Log("Please specify indices between 1 and " + GetTotalFrames() + ".");
			return null;
		}
		int count = end-start+1;
		Frame[] frames = new Frame[count];
		for(int i=start; i<=end; i++) {
			frames[i-start] = GetFrame(i);
		}
		return frames;
	}

	public Frame[] GetFrames(float start, float end) {
		if(start < 0f || end > GetTotalTime()) {
			Debug.Log("Please specify times between 0 and " + GetTotalTime() + ".");
			return null;
		}
		return GetFrames(GetFrame(start).Index, GetFrame(end).Index);
	}

	public Vector3 GetAxis(Axis axis) {
		switch(axis) {
			case Axis.XPositive:
			return Vector3.right;
			case Axis.YPositive:
			return Vector3.up;
			case Axis.ZPositive:
			return Vector3.forward;
			case Axis.XNegative:
			return -Vector3.right;
			case Axis.YNegative:
			return -Vector3.up;
			case Axis.ZNegative:
			return -Vector3.forward;
			default:
			return Vector3.zero;
		}
	}

	public void SetUnitScale(float value) {
		if(UnitScale != value) {
			UnitScale = value;
			ComputePostures();
		}
	}

	public void SetStyleTransition(float value) {
		if(StyleTransition != value) {
			StyleTransition = value;
			ComputeStyles();
		}
	}

	public void SetMotionSmoothing(float value) {
		if(MotionSmoothing != value) {
			MotionSmoothing = value;
		}
	}

	public void AddStyle(string name) {
		ArrayExtensions.Add(ref Styles, name);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Add(ref Frames[i].StyleFlags, false);
			ArrayExtensions.Add(ref Frames[i].StyleValues, 0);
		}
	}

	public void RemoveStyle() {
		ArrayExtensions.Shrink(ref Styles);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Shrink(ref Frames[i].StyleFlags);
			ArrayExtensions.Shrink(ref Frames[i].StyleValues);
		}
	}

	public void RemoveStyle(string name) {
		int index = ArrayExtensions.FindIndex(ref Styles, name);
		if(index >= 0) {
			ArrayExtensions.RemoveAt(ref Styles, index);
			for(int i=0; i<GetTotalFrames(); i++) {
				ArrayExtensions.RemoveAt(ref Frames[i].StyleFlags, index);
				ArrayExtensions.RemoveAt(ref Frames[i].StyleValues, index);
			}
		}
	}

	public void ClearStyles() {
		ArrayExtensions.Clear(ref Styles);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Clear(ref Frames[i].StyleFlags);
			ArrayExtensions.Clear(ref Frames[i].StyleValues);
		}
	}

	public void AddSequence() {
		ArrayExtensions.Add(ref Sequences, new Sequence(this));
	}

	public void AddSequence(int start, int end) {
		ArrayExtensions.Add(ref Sequences, new Sequence(this, start, end));
	}

	public void RemoveSequence() {
		ArrayExtensions.Shrink(ref Sequences);
	}

	public void RemoveSequence(int index) {
		ArrayExtensions.RemoveAt(ref Sequences, index);
	}

	public MotionData Create(string path, string currentDirectory) {
		Name = path.Substring(path.LastIndexOf("/")+1);
		if(AssetDatabase.LoadAssetAtPath(currentDirectory+Name+".asset", typeof(MotionData)) == null) {
			AssetDatabase.CreateAsset(this , currentDirectory+Name+".asset");
		} else {
			int i = 1;
			while(AssetDatabase.LoadAssetAtPath(currentDirectory+Name+Name+" ("+i+").asset", typeof(MotionData)) != null) {
				i += 1;
			}
			AssetDatabase.CreateAsset(this,currentDirectory+Name+Name+" ("+i+").asset");
		}

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
		DetectHeightMapSensor();
		DetectDepthMapSensor();
		DetectSymmetry();
		DetectCorrections();

		//Create frames
		for(int i=0; i<GetTotalFrames(); i++) {
			Frames[i] = new Frame(this, i+1, (float)i / Framerate);
		}

		//Generate
		ComputePostures();
		ComputeStyles();
		AddSequence();

		//Finish
		return this;
	}

	public void DetectHeightMapSensor() {
		BVHData.Bone bone = Source.FindBone("Hips");
		if(bone == null) {
			Debug.Log("Could not find height map sensor.");
		} else {
			HeightMapSensor = bone.Index;
		}
	}


	public void DetectDepthMapSensor() {
		BVHData.Bone bone = Source.FindBone("Head");
		if(bone == null) {
			Debug.Log("Could not find depth map sensor.");
		} else {
			DepthMapSensor = bone.Index;
		}
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

	public void DetectCorrections() {
		Corrections = new Vector3[Source.Bones.Length];
	}

	public void SetSymmetry(int source, int target) {
		if(Symmetry[source] != target) {
			Symmetry[source] = target;
		}
	}

	public void SetCorrection(int index, Vector3 correction) {
		if(Corrections[index] != correction) {
			Corrections[index] = correction;
			ComputePostures();
		}
	}

	public void ComputePostures() {
		for(int i=1; i<=GetTotalFrames(); i++) {
			GetFrame(i).ComputePosture();
		}
	}

	public void ComputeStyles() {
		for(int i=0; i<Styles.Length; i++) {
			for(int j=1; j<=GetTotalFrames(); j++) {
				Frame frame = GetFrame(j);
				if(frame.IsStyleKey(i)) {
					frame.ApplyStyleKeyValues(i);
				}
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
	public class Sequence {
		public MotionData Data;
		public int Start;
		public int End;

		public Sequence(MotionData data) {
			Data = data;
			Auto();
		}

		public Sequence(MotionData data, int start, int end) {
			Data = data;
			Start = start;
			End = end;
		}

		public int GetLength() {
			return End - Start + 1;
		}
		
		public float GetDuration() {
			return (float)GetLength() / Data.Framerate;
		}
		
		public void Auto() {
			Start = Data.GetFrame(Mathf.Min(1f, Data.GetTotalTime())).Index;
			End = Data.GetFrame(Mathf.Max(Data.GetTotalTime()-1f, 0f)).Index;
		}

	}

	[System.Serializable]
	public class Frame {
		public MotionData Data;
		public int Index;
		public float Timestamp;
		public Matrix4x4[] Local;
		public Matrix4x4[] World;
		public bool[] StyleFlags;
		public float[] StyleValues;

		public Frame(MotionData data, int index, float timestamp) {
			Data = data;
			Index = index;
			Timestamp = timestamp;
			Local = new Matrix4x4[Data.Source.Bones.Length];
			World = new Matrix4x4[Data.Source.Bones.Length];
			StyleFlags = new bool[0];
			StyleValues = new float[0];
		}

		public void ComputePosture() {
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
			for(int i=0; i<Data.Source.Bones.Length; i++) {
				Local[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Data.Corrections[i]), Vector3.one);
				World[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Data.Corrections[i]), Vector3.one);
			}
		}

		public Matrix4x4[] GetBoneTransformations(bool mirrored) {
			Matrix4x4[] transformations = new Matrix4x4[World.Length];
			for(int i=0; i<World.Length; i++) {
				transformations[i] = GetBoneTransformation(i, mirrored);
			}
			return transformations;
		}

		public Matrix4x4 GetBoneTransformation(int index, bool mirrored) {
			if(Data.MotionSmoothing == 0f) {
				return mirrored ? World[Data.Symmetry[index]].GetMirror(Data.GetAxis(Data.MirrorAxis)) : World[index];
			} else {
				Frame[] frames = Data.GetFrames(Mathf.Max(Timestamp - Data.MotionSmoothing, 0f), Mathf.Min(Timestamp + Data.MotionSmoothing, Data.GetTotalTime()));
				Vector3 P = Vector3.zero;
				Vector3 Z = Vector3.zero;
				Vector3 Y = Vector3.zero;
				float sum = 0f;
				for(int i=0; i<frames.Length; i++) {
					Frame frame = frames[i];
					float weight = 2f * (float)(i+1) / (float)(frames.Length+1);
					if(weight > 1f) {
						weight = 2f - weight;
					}
					P += weight * (mirrored ? frame.World[Data.Symmetry[index]].GetPosition().GetMirror(Data.GetAxis(Data.MirrorAxis)) : frame.World[index].GetPosition());
					Z += weight * (mirrored ? frame.World[Data.Symmetry[index]].GetForward().GetMirror(Data.GetAxis(Data.MirrorAxis)) : frame.World[index].GetForward());
					Y += weight * (mirrored ? frame.World[Data.Symmetry[index]].GetUp().GetMirror(Data.GetAxis(Data.MirrorAxis)) : frame.World[index].GetUp());
					sum += weight;
				}
				P /= sum;
				Z /= sum;
				Y /= sum;
				return Matrix4x4.TRS(P, Quaternion.LookRotation(Z, Y), Vector3.one);
			}
		}

		public Vector3[] GetBoneVelocities(bool mirrored) {
			Vector3[] velocities = new Vector3[World.Length];
			for(int i=0; i<World.Length; i++) {
				velocities[i] = GetBoneVelocity(i, mirrored);
			}
			return velocities;
		}

		public Vector3 GetBoneVelocity(int index, bool mirrored) {
			if((Data.MotionSmoothing == 0f)) {
				if(Index == 1) {
					return mirrored ? 
					((Data.GetFrame(Index+1).World[Data.Symmetry[index]].GetPosition() - World[Data.Symmetry[index]].GetPosition()) * Data.Framerate).GetMirror(Data.GetAxis(Data.MirrorAxis))
					: 
					(Data.GetFrame(Index+1).World[index].GetPosition() - World[index].GetPosition()) * Data.Framerate;
				} else {
					return mirrored ? 
					((World[index].GetPosition() - Data.GetFrame(Index-1).World[index].GetPosition()) * Data.Framerate).GetMirror(Data.GetAxis(Data.MirrorAxis))
					: 
					(World[index].GetPosition() - Data.GetFrame(Index-1).World[index].GetPosition()) * Data.Framerate;
				}
			} else {
				Frame[] frames = Data.GetFrames(Mathf.Max(Timestamp - Data.MotionSmoothing, 0f), Mathf.Min(Timestamp + Data.MotionSmoothing, Data.GetTotalTime()));
				Vector3 velocity = Vector3.zero;
				float sum = 0f;
				for(int i=0; i<frames.Length; i++) {
					Frame frame = frames[i];
					float weight = 2f * (float)(i+1) / (float)(frames.Length+1);
					if(weight > 1f) {
						weight = 2f - weight;
					}
					if(frame.Index == 1) {
						velocity += weight * (mirrored ? 
						((Data.GetFrame(frame.Index+1).World[Data.Symmetry[index]].GetPosition() - frame.World[Data.Symmetry[index]].GetPosition()) * Data.Framerate).GetMirror(Data.GetAxis(Data.MirrorAxis))
						: 
						(Data.GetFrame(frame.Index+1).World[index].GetPosition() - frame.World[index].GetPosition()) * Data.Framerate);
					} else {
						velocity += weight * (mirrored ? 
						((frame.World[index].GetPosition() - Data.GetFrame(frame.Index-1).World[index].GetPosition()) * Data.Framerate).GetMirror(Data.GetAxis(Data.MirrorAxis))
						: 
						(frame.World[index].GetPosition() - Data.GetFrame(frame.Index-1).World[index].GetPosition()) * Data.Framerate);
					}
					sum += weight;
				}
				velocity /= sum;
				return velocity;
			}
		}

		public Matrix4x4 GetRoot(bool mirrored) {
			Matrix4x4 transformation = GetBoneTransformation(0, mirrored);
			Vector3 position = Utility.ProjectGround(transformation.GetPosition(), Data.GroundMask);
			Vector3 forward = Quaternion.FromToRotation(Vector3.forward, Data.GetAxis(Data.ForwardAxis)) * transformation.GetForward();
			forward.y = 0f;
			return Matrix4x4.TRS(position, Quaternion.LookRotation(forward, Vector3.up), Vector3.one);
		}

		public Vector3 GetRootMotion(bool mirrored) {
			Matrix4x4 currentRoot = GetRoot(mirrored);
			Matrix4x4 previousRoot = Data.GetFrame(Mathf.Max(Index-1, 1)).GetRoot(mirrored);
			Matrix4x4 delta = currentRoot.GetRelativeTransformationTo(previousRoot);
			return new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z) * Data.Framerate;
		}

		public Trajectory GetTrajectory(bool mirrored) {
			Trajectory trajectory = new Trajectory(12, Data.Styles.Length);
			for(int i=0; i<6; i++) {
				Frame previous = Data.GetFrame(Mathf.Clamp(Timestamp - 1f + (float)i/6f, 0f, Data.GetTotalTime()));
				trajectory.Points[i].SetTransformation(previous.GetRoot(mirrored));
				trajectory.Points[i].SetVelocity(previous.GetBoneVelocity(0, mirrored));
				trajectory.Points[i].Styles = (float[])previous.StyleValues.Clone();
			}
			trajectory.Points[6].SetTransformation(GetRoot(mirrored));
			trajectory.Points[6].SetVelocity(GetBoneVelocity(0, mirrored));
			trajectory.Points[6].Styles = (float[])StyleValues.Clone();
			for(int i=1; i<=5; i++) {
				Frame future = Data.GetFrame(Mathf.Clamp(Timestamp + (float)i/5f, 0f, Data.GetTotalTime()));
				trajectory.Points[6+i].SetTransformation(future.GetRoot(mirrored));
				trajectory.Points[6+i].SetVelocity(future.GetBoneVelocity(0, mirrored));
				trajectory.Points[6+i].Styles = (float[])future.StyleValues.Clone();
			}
			return trajectory;
		}

		public HeightMap GetHeightMap(bool mirrored) {
			HeightMap heightMap = new HeightMap(Data.HeightMapSize);
			Matrix4x4 pivot = GetBoneTransformation(Data.HeightMapSensor, mirrored);
			heightMap.Sense(pivot, Data.ObjectMask);
			return heightMap;
		}

		public DepthMap GetDepthMap(bool mirrored) {
			DepthMap depthMap = new DepthMap(Data.DepthMapResolution, Data.DepthMapSize, Data.DepthMapDistance);
			Matrix4x4 pivot = GetBoneTransformation(Data.DepthMapSensor, mirrored);
			pivot *= Matrix4x4.TRS(Vector3.zero, Quaternion.FromToRotation(Vector3.forward, Data.GetAxis(Data.DepthMapAxis)), Vector3.one);
			depthMap.Sense(pivot, Data.ObjectMask);
			return depthMap;
		}

		public void ToggleStyle(int index) {
			Frame next = GetNextStyleKey(index);
			StyleFlags[index] = !StyleFlags[index];
			int start = Index+1;
			int end = next == null ? Data.GetTotalFrames() : next.Index-1;
			for(int i=start; i<=end; i++) {
				Data.GetFrame(i).StyleFlags[index] = StyleFlags[index];
			}
			ApplyStyleKeyValues(index);
		}

		public void ApplyStyleKeyValues(int index) {
			//Previous Frames
			Frame previousKey = GetPreviousStyleKey(index);
			previousKey = previousKey == null ? Data.GetFrame(1) : previousKey;
			Frame pivot = Data.GetFrame(Mathf.Max(previousKey.Index, Data.GetFrame(Mathf.Max(Timestamp - Data.StyleTransition, previousKey.Timestamp)).Index));
			if(pivot == this) {
				StyleValues[index] = StyleFlags[index] ? 1f : 0f;
			} else {
				for(int i=previousKey.Index; i<=pivot.Index; i++) {
					Data.GetFrame(i).StyleValues[index] = previousKey.StyleFlags[index] ? 1f : 0f;
				}
				float valA = pivot.StyleFlags[index] ? 1f : 0f;
				float valB = this.StyleFlags[index] ? 1f : 0f;
				for(int i=pivot.Index; i<=this.Index; i++) {
					float weight = (float)(i-pivot.Index) / (float)(this.Index - pivot.Index);
					Data.GetFrame(i).StyleValues[index] = (1f-weight) * valA + weight * valB;
				}
			}
			//Next Frames
			Frame nextKey = GetNextStyleKey(index);
			nextKey = nextKey == null ? Data.GetFrame(Data.GetTotalFrames()) : nextKey;
			for(int i=this.Index; i<=nextKey.Index; i++) {
				Data.GetFrame(i).StyleValues[index] = StyleFlags[index] ? 1f : 0f;
			}
			previousKey = StyleFlags[index] ? this : previousKey;
			pivot = Data.GetFrame(Mathf.Max(previousKey.Index, Data.GetFrame(Mathf.Max(nextKey.Timestamp - Data.StyleTransition, Timestamp)).Index));
			if(pivot == nextKey) {
				StyleValues[index] = StyleFlags[index] ? 1f : 0f;
			} else {
				float valA = pivot.StyleFlags[index] ? 1f : 0f;
				float valB = nextKey.StyleFlags[index] ? 1f : 0f;
				for(int i=pivot.Index; i<=nextKey.Index; i++) {
					float weight = (float)(i-pivot.Index) / (float)(nextKey.Index - pivot.Index);
					Data.GetFrame(i).StyleValues[index] = (1f-weight) * valA + weight * valB;
				}
			}
		}

		public Frame GetAnyNextStyleKey() {
			Frame frame = this;
			while(frame.Index < Data.GetTotalFrames()) {
				frame = Data.GetFrame(frame.Index+1);
				if(frame.IsAnyStyleKey()) {
					return frame;
				}
			}
			return null;
		}

		public Frame GetAnyPreviousStyleKey() {
			Frame frame = this;
			while(frame.Index > 1) {
				frame = Data.GetFrame(frame.Index-1);
				if(frame.IsAnyStyleKey()) {
					return frame;
				}
			}
			return null;
		}

		public bool IsAnyStyleKey() {
			for(int i=0; i<StyleFlags.Length; i++) {
				if(IsStyleKey(i)) {
					return true;
				}
			}
			return false;
		}

		public Frame GetNextStyleKey(int index) {
			Frame frame = this;
			while(frame.Index < Data.GetTotalFrames()) {
				frame = Data.GetFrame(frame.Index+1);
				if(frame.IsStyleKey(index)) {
					return frame;
				}
			}
			return null;
		}

		public Frame GetPreviousStyleKey(int index) {
			Frame frame = this;
			while(frame.Index > 1) {
				frame = Data.GetFrame(frame.Index-1);
				if(frame.IsStyleKey(index)) {
					return frame;
				}
			}
			return null;
		}

		public bool IsStyleKey(int index) {
			Frame previous = this;
			if(Index > 1) {
				previous = Data.GetFrame(Index-1);
			}
			if(!StyleFlags[index] && previous.StyleFlags[index]) {
				return true;
			}
			if(StyleFlags[index] && !previous.StyleFlags[index]) {
				return true;
			}
			return false;
		}
	}

}
#endif