#if UNITY_EDITOR
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

public class MotionData : ScriptableObject {

	public enum Axis {XPositive, YPositive, ZPositive, XNegative, YNegative, ZNegative};

	public Hierarchy Source;

	public string Name = string.Empty;
	public float Framerate = 1f;
	public float Scaling = 1f;
	public int RootSmoothing = 0;
	public string[] Styles = new string[0];
	public float StyleTransition = 0.5f;
	public Axis MirrorAxis = Axis.XPositive;
	public int[] Symmetry = new int[0];
	public LayerMask GroundMask = -1;
	public LayerMask ObjectMask = -1;
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

	public void SetStyleTransition(float value) {
		if(StyleTransition != value) {
			StyleTransition = value;
			ComputeStyles();
		}
	}

	public void AddStyle(string name) {
		ArrayExtensions.Add(ref Styles, name);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Add(ref Frames[i].StyleFlags, false);
			ArrayExtensions.Add(ref Frames[i].StyleValues, 0);
		}
		for(int s=0; s<Sequences.Length; s++) {
			ArrayExtensions.Add(ref Sequences[s].TransitionCopies, 0);
			ArrayExtensions.Add(ref Sequences[s].StyleCopies, 0);
		}
	}

	public void RemoveStyle() {
		ArrayExtensions.Shrink(ref Styles);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Shrink(ref Frames[i].StyleFlags);
			ArrayExtensions.Shrink(ref Frames[i].StyleValues);
		}
		for(int s=0; s<Sequences.Length; s++) {
			ArrayExtensions.Shrink(ref Sequences[s].TransitionCopies);
			ArrayExtensions.Shrink(ref Sequences[s].StyleCopies);
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
			for(int s=0; s<Sequences.Length; s++) {
				ArrayExtensions.RemoveAt(ref Sequences[s].TransitionCopies, index);
				ArrayExtensions.RemoveAt(ref Sequences[s].StyleCopies, index);
			}
		}
	}

	public void ClearStyles() {
		ArrayExtensions.Clear(ref Styles);
		for(int i=0; i<GetTotalFrames(); i++) {
			ArrayExtensions.Clear(ref Frames[i].StyleFlags);
			ArrayExtensions.Clear(ref Frames[i].StyleValues);
		}
		for(int s=0; s<Sequences.Length; s++) {
			ArrayExtensions.Clear(ref Sequences[s].TransitionCopies);
			ArrayExtensions.Clear(ref Sequences[s].StyleCopies);
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

	public void DetectHeightMapSensor() {
		Hierarchy.Bone bone = Source.FindBone("Hips");
		if(bone == null) {
			Debug.Log("Could not find height map sensor.");
		} else {
			HeightMapSensor = bone.Index;
		}
	}


	public void DetectDepthMapSensor() {
		Hierarchy.Bone bone = Source.FindBone("Head");
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
				int pivot = name.IndexOf("Left");
				Hierarchy.Bone bone = Source.FindBone(name.Substring(0, pivot)+"Right"+name.Substring(pivot+4));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(name.Contains("Right")) {
				int pivot = name.IndexOf("Right");
				Hierarchy.Bone bone = Source.FindBone(name.Substring(0, pivot)+"Left"+name.Substring(pivot+5));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(name.StartsWith("L") && char.IsUpper(name[1])) {
				Hierarchy.Bone bone = Source.FindBone("R"+name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(name.StartsWith("R") && char.IsUpper(name[1])) {
				Hierarchy.Bone bone = Source.FindBone("L"+name.Substring(1));
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

	public void SetSymmetry(int source, int target) {
		if(Symmetry[source] != target) {
			Symmetry[source] = target;
		}
	}

	public void SetCorrection(int index, Vector3 correction) {
		//if(Corrections[index] != correction) {
		//	Corrections[index] = correction;
		//	ComputePostures();
		//}
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
	public class Hierarchy {
		public Bone[] Bones;

		public Hierarchy() {
			Bones = new Bone[0];
		}

		public void AddBone(string name, string parent) {
			ArrayExtensions.Add(ref Bones, new Bone(Bones.Length, name, parent));
		}

		public Bone FindBone(string name) {
			return System.Array.Find(Bones, x => x.Name == name);
		}

		public Bone FindBoneContains(string name) {
			return System.Array.Find(Bones, x => x.Name.Contains(name));
		}

		[System.Serializable]
		public class Bone {
			public int Index;
			public string Name;
			public string Parent;
			public Bone(int index, string name, string parent) {
				Index = index;
				Name = name;
				Parent = parent;
			}
		}
	}

	[System.Serializable]
	public class Sequence {
		public MotionData Data;
		public int[] TransitionCopies;
		public int[] StyleCopies;
		public Interval[] Copies;
		public int Start;
		public int End;

		public Sequence(MotionData data) {
			Data = data;
			TransitionCopies = new int[data.Styles.Length];
			StyleCopies = new int[data.Styles.Length];
			Copies = new Interval[0];
			SetStart(1);
			SetEnd(data.GetTotalFrames());
		}

		public Sequence(MotionData data, int start, int end) {
			Data = data;
			TransitionCopies = new int[data.Styles.Length];
			StyleCopies = new int[data.Styles.Length];
			Copies = new Interval[0];
			SetStart(start);
			SetEnd(end);
		}

		public Interval[] GetIntervals() {
			Interval[] intervals = new Interval[1 + Copies.Length];
			intervals[0] = new Interval(Start, End);
			for(int i=1; i<intervals.Length; i++) {
				intervals[i] = Copies[i-1];
			}
			return intervals;
		}

		public int GetLength() {
			return End - Start + 1;
		}
		
		public float GetDuration() {
			return (float)GetLength() / Data.Framerate;
		}

		public void SetStart(int value) {
			value = Mathf.Clamp(value, 1, Data.GetTotalFrames());
			if(Start != value) {
				Start = value;
				CreateCopies();
			}
		}

		public void SetEnd(int value) {
			value = Mathf.Clamp(value, 1, Data.GetTotalFrames());
			if(End != value) {
				End = value;
				CreateCopies();
			}
		}

		public void SetTransitionCopies(int index, int value) {
			value = Mathf.Max(0, value);
			if(TransitionCopies[index] != value) {
				TransitionCopies[index] = value;
				CreateCopies();
			}
		}

		public void SetTransitionCopies(string style, int value) {
			SetTransitionCopies(System.Array.FindIndex(Data.Styles, x => x == style), value);
		}

		public void SetStyleCopies(int index, int value) {
			value = Mathf.Max(0, value);
			if(StyleCopies[index] != value) {
				StyleCopies[index] = value;
				CreateCopies();
			}
		}

		public void SetStyleCopies(string style, int value) {
			SetStyleCopies(System.Array.FindIndex(Data.Styles, x => x == style), value);
		}

		public void CreateCopies() {
			Copies = new Interval[0];

			//Transition Copies
			for(int c=0; c<TransitionCopies.Length; c++) {
				for(int i=Start; i<=End; i++) {
					if(Data.GetFrame(i).IsTransition(c)) {
						Interval copy = new Interval();
						copy.Start = i;
						while(true) {
							i += 1;
							if(i==End) {
								copy.End = i;
								break;
							}
							if(!Data.GetFrame(i).IsTransition(c)) {
								copy.End = i-1;
								break;
							}
						}
						for(int k=0; k<TransitionCopies[c]; k++) {
							ArrayExtensions.Add(ref Copies, copy);
						}
					}
				}
			}

			//Style Copies
			for(int c=0; c<StyleCopies.Length; c++) {
				if(StyleCopies[c] > 0) {
					for(int i=Start; i<=End; i++) {
						if(Data.GetFrame(i).StyleValues[c] == 1f) {
							Interval copy = new Interval();
							copy.Start = i;
							while(true) {
								i += 1;
								if(i==End) {
									copy.End = i;
									break;
								}
								if(Data.GetFrame(i).StyleValues[c] != 1f) {
									copy.End = i-1;
									break;
								}
							}
							for(int k=0; k<StyleCopies[c]; k++) {
								ArrayExtensions.Add(ref Copies, copy);
							}
						}
					}
				}
			}
		}

		[System.Serializable]
		public class Interval {
			public int Start;
			public int End;
			public Interval() {
				Start = 0;
				End = 0;
			}
			public Interval(int start, int end) {
				Start = start;
				End = end;
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

		public Frame GetPreviousFrame() {
			return Data.GetFrame(Mathf.Clamp(Index-1, 1, Data.GetTotalFrames()));
		}

		public Frame GetNextFrame() {
			return Data.GetFrame(Mathf.Clamp(Index+1, 1, Data.GetTotalFrames()));
		}

		public Matrix4x4[] GetBoneTransformations(bool mirrored) {
			Matrix4x4[] transformations = new Matrix4x4[World.Length];
			for(int i=0; i<World.Length; i++) {
				transformations[i] = GetBoneTransformation(i, mirrored);
			}
			return transformations;
		}

		public Matrix4x4 GetBoneTransformation(int index, bool mirrored, int smoothing = 0) {
			if(smoothing  == 0) {
				return Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Data.Scaling * Vector3.one) * (mirrored ? World[Data.Symmetry[index]].GetMirror(Data.GetAxis(Data.MirrorAxis)) : World[index]);
			} else {
				Frame[] frames = Data.GetFrames(Mathf.Clamp(Index - smoothing, 1, Data.GetTotalFrames()), Mathf.Clamp(Index + smoothing, 1, Data.GetTotalFrames()));
				Vector3 P = Vector3.zero;
				Vector3 Z = Vector3.zero;
				Vector3 Y = Vector3.zero;
				float sum = 0f;
				for(int i=0; i<frames.Length; i++) {
					float weight = 2f * (float)(i+1) / (float)(frames.Length+1);
					if(weight > 1f) {
						weight = 2f - weight;
					}
					Matrix4x4 matrix = mirrored ? frames[i].World[Data.Symmetry[index]].GetMirror(Data.GetAxis(Data.MirrorAxis)) : frames[i].World[index];
					P += weight * matrix.GetPosition();
					Z += weight * matrix.GetForward();
					Y += weight * matrix.GetUp();
					sum += weight;
				}
				P /= sum;
				Z /= sum;
				Y /= sum;
				return Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Data.Scaling * Vector3.one) * Matrix4x4.TRS(P, Quaternion.LookRotation(Z, Y), Vector3.one);
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
			if(Index == 1) {
				return GetNextFrame().GetBoneVelocity(index, mirrored);
			} else {
				return (GetBoneTransformation(index, mirrored).GetPosition() - GetPreviousFrame().GetBoneTransformation(index, mirrored).GetPosition()) * Data.Framerate;
			}
		}

		public Matrix4x4 GetRootTransformation(bool mirrored) {
			return Matrix4x4.TRS(GetRootPosition(mirrored), GetRootRotation(mirrored), Vector3.one);
		}

		private Vector3 GetRootPosition(bool mirrored) {
			return Utility.ProjectGround(GetBoneTransformation(0, mirrored, Data.RootSmoothing).GetPosition(), Data.GroundMask);
		}

		private Quaternion GetRootRotation(bool mirrored) {
			
			//Vector3 v1 = GetBoneTransformation(Data.Source.FindBone("RightHip").Index, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(Data.Source.FindBone("LeftHip").Index, mirrored, Data.RootSmoothing).GetPosition();
			//Vector3 v2 = GetBoneTransformation(Data.Source.FindBone("RightShoulder").Index, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(Data.Source.FindBone("LeftShoulder").Index, mirrored, Data.RootSmoothing).GetPosition();
			//v1.y = 0f;
			//v2.y = 0f;
			//Vector3 v = (v1+v2).normalized;
			//Vector3 forward = -Vector3.Cross(v, Vector3.up);
			//forward.y = 0f;
			

			/*
			Vector3 neck = GetBoneTransformation(Data.Source.FindBone("Neck").Index, mirrored, Data.RootSmoothing).GetPosition();
			Vector3 hips = GetBoneTransformation(Data.Source.FindBone("Hips").Index, mirrored, Data.RootSmoothing).GetPosition();
			//int leftShoulder = Data.Source.FindBone("LeftShoulder").Index;
			//int rightShoulder = Data.Source.FindBone("RightShoulder").Index;
			//int leftUpLeg = Data.Source.FindBone("LeftUpLeg").Index;
			////int rightUpLeg = Data.Source.FindBone("RightUpLeg").Index;
			Vector3 forward = Vector3.zero;
			forward += neck - hips;
			//forward += GetBoneTransformation(leftShoulder, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(leftUpLeg, mirrored, Data.RootSmoothing).GetPosition();
			//forward += GetBoneTransformation(rightShoulder, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(rightUpLeg, mirrored, Data.RootSmoothing).GetPosition();
			*/

			Vector3 forward = GetBoneTransformation(Data.Source.FindBoneContains("Hip").Index, mirrored, Data.RootSmoothing).GetForward();

			forward.y = 0f;
			return Quaternion.LookRotation(forward.normalized, Vector3.up);
		}

		public Vector3 GetRootVelocity(bool mirrored) {
			if(Index == 1) {
				return GetNextFrame().GetRootVelocity(mirrored);
			} else {
				Vector3 velocity = (GetBoneTransformation(0, mirrored, Data.RootSmoothing).GetPosition() - GetPreviousFrame().GetBoneTransformation(0, mirrored, Data.RootSmoothing).GetPosition()) * Data.Framerate;
				velocity.y = 0f;
				return velocity;
			}
		}

		public Vector3 GetRootMotion(bool mirrored) {
			if(Index == 1) {
				return GetNextFrame().GetRootMotion(mirrored);
			} else {
				Matrix4x4 reference = GetPreviousFrame().GetRootTransformation(mirrored);
				Matrix4x4 current = GetRootTransformation(mirrored);
				Matrix4x4 delta = current.GetRelativeTransformationTo(reference);
				Vector3 translationalMotion = delta.GetPosition() * Data.Framerate;
				float angularMotion = Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up) * Data.Framerate;
				return new Vector3(translationalMotion.x, angularMotion, translationalMotion.z);
			}
		}

		public float GetSpeed(bool mirrored) {
			float length = 0f;
			Vector3[] positions = new Vector3[6];
			positions[0] = GetRootPosition(mirrored);
			positions[0].y = 0f;
			for(int i=1; i<=5; i++) {
				Frame future = Data.GetFrame(Mathf.Clamp(Timestamp + (float)i/5f, 0f, Data.GetTotalTime()));
				positions[i] = future.GetRootPosition(mirrored);
				positions[i].y = 0f;
			}
			for(int i=1; i<=5; i++) {
				length += Vector3.Distance(positions[i-1], positions[i]);
			}
			return length;
		}

		public Trajectory GetTrajectory(bool mirrored) {
			Trajectory trajectory = new Trajectory(12, Data.Styles.Length);

			//Current
			trajectory.Points[6].SetTransformation(GetRootTransformation(mirrored));
			trajectory.Points[6].SetVelocity(GetRootVelocity(mirrored));
			trajectory.Points[6].SetSpeed(GetSpeed(mirrored));
			trajectory.Points[6].Styles = (float[])StyleValues.Clone();

			//Past
			for(int i=0; i<6; i++) {
				float delta = -1f + (float)i/6f;
				if(Timestamp + delta < 0f) {
					float pivot = - Timestamp - delta;
					float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
					float ratio = (pivot - Timestamp) / (clamped - Timestamp);
					if(float.IsNaN(ratio)) {
						ratio = 1f;
					}
					Frame reference = Data.GetFrame(clamped);
					trajectory.Points[i].SetPosition(Data.GetFrame(0f).GetRootPosition(mirrored) - ratio * (reference.GetRootPosition(mirrored) - Data.GetFrame(0f).GetRootPosition(mirrored)));
					trajectory.Points[i].SetRotation(reference.GetRootRotation(mirrored));
					trajectory.Points[i].SetVelocity(reference.GetRootVelocity(mirrored));
					trajectory.Points[i].SetSpeed(reference.GetSpeed(mirrored));
					trajectory.Points[i].Styles = (float[])reference.StyleValues.Clone();
				} else {
					Frame previous = Data.GetFrame(Mathf.Clamp(Timestamp + delta, 0f, Data.GetTotalTime()));
					trajectory.Points[i].SetTransformation(previous.GetRootTransformation(mirrored));
					trajectory.Points[i].SetVelocity(previous.GetRootVelocity(mirrored));
					trajectory.Points[i].SetSpeed(previous.GetSpeed(mirrored));
					trajectory.Points[i].Styles = (float[])previous.StyleValues.Clone();
				}
			}

			//Future
			for(int i=1; i<=5; i++) {
				float delta = (float)i/5f;
				if(Timestamp + delta > Data.GetTotalTime()) {
					float pivot = 2f*Data.GetTotalTime() - Timestamp - delta;
					float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
					float ratio = (pivot - Timestamp) / (clamped - Timestamp);
					if(float.IsNaN(ratio)) {
						ratio = 1f;
					}
					Frame reference = Data.GetFrame(clamped);
					trajectory.Points[6+i].SetPosition(Data.GetFrame(Data.GetTotalTime()).GetRootPosition(mirrored) - ratio * (reference.GetRootPosition(mirrored) - Data.GetFrame(Data.GetTotalTime()).GetRootPosition(mirrored)));
					trajectory.Points[6+i].SetRotation(reference.GetRootRotation(mirrored));
					trajectory.Points[6+i].SetVelocity(reference.GetRootVelocity(mirrored));
					trajectory.Points[6+i].SetSpeed(reference.GetSpeed(mirrored));
					trajectory.Points[6+i].Styles = (float[])reference.StyleValues.Clone();
				} else {
					Frame future = Data.GetFrame(Mathf.Clamp(Timestamp + delta, 0f, Data.GetTotalTime()));
					trajectory.Points[6+i].SetTransformation(future.GetRootTransformation(mirrored));
					trajectory.Points[6+i].SetVelocity(future.GetRootVelocity(mirrored));
					trajectory.Points[6+i].SetSpeed(future.GetSpeed(mirrored));
					trajectory.Points[6+i].Styles = (float[])future.StyleValues.Clone();
				}
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

		public bool IsAnyTransition() {
			for(int i=0; i<StyleValues.Length; i++) {
				if(IsTransition(i)) {
					return true;
				}
			}
			return false;
		}

		public bool IsTransition(int index) {
			return StyleValues[index] > 0f & StyleValues[index] < 1f;
		}

	}

}
#endif