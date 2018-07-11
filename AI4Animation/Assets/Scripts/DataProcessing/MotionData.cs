#if UNITY_EDITOR
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

public class MotionData : ScriptableObject {

	public enum AXIS {XPositive, YPositive, ZPositive, XNegative, YNegative, ZNegative};

	public Hierarchy Source = null;
	public Frame[] Frames = new Frame[0];
	public Module[] Modules = new Module[0];

	public float Framerate = 1f;
	public float Scaling = 1f;
	public int RootSmoothing = 0;
	public AXIS MirrorAxis = AXIS.XPositive;
	public int[] Symmetry = new int[0];
	public LayerMask Ground = -1;
	public bool Export = true;
	public Sequence[] Sequences = new Sequence[0];

	public float GetTotalTime() {
		return GetTotalFrames() / Framerate;
	}

	public int GetTotalFrames() {
		return Frames.Length;
	}

	public Frame GetFirstFrame() {
		return Frames[0];
	}

	public Frame GetLastFrame() {
		return Frames[Frames.Length-1];
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
			frames[i-start] = Frames[i-1];
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

	public void AddModule(Module.TYPE type) {
		if(System.Array.Find(Modules, x => x.Type() == type)) {
			Debug.Log("Module of type " + type.ToString() + " already exists.");
		} else {
			switch(type) {
				case Module.TYPE.Style:
				ArrayExtensions.Add(ref Modules, ScriptableObject.CreateInstance<StyleModule>().Initialise(this));
				break;
				case Module.TYPE.Phase:
				ArrayExtensions.Add(ref Modules, ScriptableObject.CreateInstance<PhaseModule>().Initialise(this));
				break;
				case Module.TYPE.Contact:
				ArrayExtensions.Add(ref Modules, ScriptableObject.CreateInstance<ContactModule>().Initialise(this));
				break;
				case Module.TYPE.DepthMap:
				ArrayExtensions.Add(ref Modules, ScriptableObject.CreateInstance<DepthMapModule>().Initialise(this));
				break;
				case Module.TYPE.HeightMap:
				ArrayExtensions.Add(ref Modules, ScriptableObject.CreateInstance<HeightMapModule>().Initialise(this));
				break;
			}
			AssetDatabase.AddObjectToAsset(Modules[Modules.Length-1], this);
		}
	}

	public void RemoveModule(Module.TYPE type) {
		Module module = GetModule(type);
		if(!module) {
			Debug.Log("Module of type " + type.ToString() + " does not exist.");
		} else {
			ArrayExtensions.Remove(ref Modules, module);
			Utility.Destroy(module);
		}
	}

	public Module GetModule(Module.TYPE type) {
		return System.Array.Find(Modules, x => x.Type() == type);
	}

	public Vector3 GetAxis(AXIS axis) {
		switch(axis) {
			case AXIS.XPositive:
			return Vector3.right;
			case AXIS.YPositive:
			return Vector3.up;
			case AXIS.ZPositive:
			return Vector3.forward;
			case AXIS.XNegative:
			return -Vector3.right;
			case AXIS.YNegative:
			return -Vector3.up;
			case AXIS.ZNegative:
			return -Vector3.forward;
			default:
			return Vector3.zero;
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
			public int Index = -1;
			public string Name = "";
			public string Parent = "";
			public Vector3 Alignment = Vector3.zero;
			public bool Active = true;
			public Bone(int index, string name, string parent) {
				Index = index;
				Name = name;
				Parent = parent;
				Alignment = Vector3.zero;
				Active = true;
			}
		}
	}

	[System.Serializable]
	public class Sequence {
		public MotionData Data;
		//public int[] TransitionCopies;
		//public int[] StyleCopies;
		public Interval[] Copies;
		public int Start;
		public int End;

		public Sequence(MotionData data) {
			Data = data;
			//TransitionCopies = new int[data.Styles.Length];
			//StyleCopies = new int[data.Styles.Length];
			Copies = new Interval[0];
			SetStart(1);
			SetEnd(data.GetTotalFrames());
		}

		public Sequence(MotionData data, int start, int end) {
			Data = data;
			//TransitionCopies = new int[data.Styles.Length];
			//StyleCopies = new int[data.Styles.Length];
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
			//value = Mathf.Max(0, value);
			//if(TransitionCopies[index] != value) {
			//	TransitionCopies[index] = value;
			//	CreateCopies();
			//}
		}

		public void SetTransitionCopies(string style, int value) {
			//SetTransitionCopies(System.Array.FindIndex(Data.Styles, x => x == style), value);
		}

		public void SetStyleCopies(int index, int value) {
			//value = Mathf.Max(0, value);
			//if(StyleCopies[index] != value) {
			//	StyleCopies[index] = value;
			//	CreateCopies();
			//}
		}

		public void SetStyleCopies(string style, int value) {
			//SetStyleCopies(System.Array.FindIndex(Data.Styles, x => x == style), value);
		}

		public void CreateCopies() {
			/*
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
			*/
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

	public void Inspector(MotionEditor editor) {
		
	}

}
#endif