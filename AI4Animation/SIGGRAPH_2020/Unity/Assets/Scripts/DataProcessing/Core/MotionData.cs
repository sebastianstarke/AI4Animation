#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using System;
using System.IO;
using System.Collections.Generic;

public class MotionData : ScriptableObject {

	public Hierarchy Source = null;
	public Frame[] Frames = new Frame[0];
	public Module[] Modules = new Module[0];
	public Sequence[] Sequences = new Sequence[0];
	public Vector2 Valid = new Vector2(0f, 1f);

	public float Framerate = 1f;
	public Vector3 Offset = Vector3.zero;
	public float Scale = 1f;
	public Axis MirrorAxis = Axis.XPositive;
	public int[] Symmetry = new int[0];

	public bool Export = false;
	public bool Tagged = false;
	
	[NonSerialized] private bool Precomputable = false;

	public void MarkDirty(bool asset=true, bool scene=true) {
		if(asset) {
			EditorUtility.SetDirty(this);
		}
		if(scene) {
			EditorSceneManager.MarkSceneDirty(GetScene());
		}
	}

	public bool IsDirty() {
		return EditorUtility.IsDirty(this) || GetScene().isDirty;
	}

	public void SetPrecomputable(bool value) {
		if(Precomputable != value) {
			Precomputable = value;
			ResetPrecomputation();
		}
	}

	public bool IsPrecomputable(float timestamp) {
		return Precomputable && GetPrecomputedIndex(timestamp) >= 0 && GetPrecomputedIndex(timestamp) < GetPrecomputationLength();
	}

	public int GetPrecomputedIndex(float timestamp) {
		return Mathf.RoundToInt(timestamp * Framerate) + GetPrecomputationPadding();
	}

	private int GetPrecomputationLength() {
		return Frames.Length + 2*GetPrecomputationPadding();
	}

	private int GetPrecomputationPadding() {
		return Mathf.RoundToInt(MotionEditor.GetInstance().GetTimeSeries().Window * Framerate);
	}

	public Precomputable<T>[] ResetPrecomputable<T>(Precomputable<T>[] item) {
		return Precomputable ? new Precomputable<T>[GetPrecomputationLength()] : null;
	}

	public void ResetPrecomputation() {
		foreach(Module m in Modules) {
			m.ResetPrecomputation();
		}
	}

	public string GetName() {
		return name;
	}
	
	public float GetDeltaTime() {
		return 1f / Framerate;
	}

	public float GetTotalTime() {
		return (Frames.Length-1) / Framerate;
	}

	public int GetTotalFrames() {
		return Frames.Length;
	}

	public void SetFirstValidFrame(Frame frame) {
		Valid.x = (float)(frame.Index-1) / (float)(Frames.Length-1);
	}

	public void SetLastValidFrame(Frame frame) {
		Valid.y = (float)(frame.Index-1) / (float)(Frames.Length-1);
	}

	public Frame GetFirstValidFrame() {
		return GetFrame(Valid.x * GetTotalTime());
	}

	public Frame GetLastValidFrame() {
		return GetFrame(Valid.y * GetTotalTime());
	}

	public Frame GetFrame(int index) {
		return Frames[Mathf.Clamp(index-1, 0, Frames.Length-1)];
		/*
		if(index < 1 || index > GetTotalFrames()) {
			Debug.Log("Please specify an index between 1 and " + GetTotalFrames() + ". Given " + index + ".");
			return null;
		}
		return Frames[index-1];
		*/
	}

	public Frame GetFrame(float time) {
		return Frames[Mathf.Clamp(Mathf.RoundToInt(time * Framerate), 0, Frames.Length-1)];
		/*
		if(time < 0f || time > GetTotalTime()) {
			Debug.Log("Please specify a time between 0 and " + GetTotalTime() + ". Given " + time + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.RoundToInt(time * Framerate) + 1, GetTotalFrames()));
		*/
	}

	//Returns absolute timestamps around frame at framerate steps.
	public float[] SimulateTimestamps(Frame frame, int padding) {
		float step = 1f/Framerate;
		float start = frame.Timestamp - padding*step;
		float[] timestamps = new float[2*padding+1];
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = start + i*step;
		}
		return timestamps;
	}

	//Returns absolute timestamps around frame at framerate steps.
	public float[] SimulateTimestamps(Frame frame, int pastPadding, int futurePadding) {
		float step = 1f/Framerate;
		float start = frame.Timestamp - pastPadding*step;
		float[] timestamps = new float[Mathf.RoundToInt((float)(pastPadding+futurePadding)/step)+1];
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = start + i*step;
		}
		return timestamps;
	}

	//Returns absolute timestamps around frame at framerate steps.
	public float[] SimulateTimestamps(Frame frame, float padding) {
		float step = 1f/Framerate;
		float start = frame.Timestamp - padding;
		float[] timestamps = new float[2*Mathf.RoundToInt(padding/step)+1];
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = start + i*step;
		}
		return timestamps;
	}

	//Returns absolute timestamps around frame at framerate steps.
	public float[] SimulateTimestamps(Frame frame, float pastPadding, float futurePadding) {
		float step = 1f/Framerate;
		float start = frame.Timestamp - pastPadding;
		float[] timestamps = new float[Mathf.RoundToInt((pastPadding+futurePadding)/step)+1];
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = start + i*step;
		}
		return timestamps;
	}

	public float[] GetTimestamps(int start, int end) {
		Frame[] frames = GetFrames(start, end);
		float[] timestamps = new float[frames.Length];
		for(int i=0; i<frames.Length; i++) {
			timestamps[i] = frames[i].Timestamp;
		}
		return timestamps;
	}

	public float[] GetTimestamps(float start, float end) {
		return GetTimestamps(GetFrame(start).Index, GetFrame(end).Index);
	}

	//Window is size in seconds of the time window, resolution is between 0 and 1 to control the sampling.
	public float[] GetTimeWindow(float window, float resolution) {
		window = Mathf.Max(window, 0f);
		resolution = Mathf.Clamp(resolution, 0f, 1f);
		float[] timeWindow = new float[Mathf.RoundToInt(resolution*window/GetDeltaTime())+1];
		if(timeWindow.Length > 1) {
			for(int i=0; i<timeWindow.Length; i++) {
				timeWindow[i] = -window/2f + (float)i/(float)(timeWindow.Length-1)*window;
			}
		}
		return timeWindow;
	}

	public Frame[] GetFrames(Frame start, Frame end) {
		return GetFrames(start.Index, end.Index);
	}

	public Frame[] GetFrames(int start, int end) {
		start = Mathf.Clamp(start, 1, GetTotalFrames());
		end = Mathf.Clamp(end, 1, GetTotalFrames());
		int count = end-start+1;
		Frame[] frames = new Frame[count];
		for(int i=start; i<=end; i++) {
			frames[i-start] = Frames[i-1];
		}
		return frames;
	}

	public Frame[] GetFrames(float start, float end) {
		return GetFrames(GetFrame(start).Index, GetFrame(end).Index);
	}

	public void AddSequence() {
		ArrayExtensions.Append(ref Sequences, new Sequence(1, GetTotalFrames()));
	}

	public void AddSequence(int start, int end) {
		ArrayExtensions.Append(ref Sequences, new Sequence(start, end));
	}

	public void RemoveSequence(Sequence sequence) {
		ArrayExtensions.Remove(ref Sequences, sequence);
	}

	public void SetSequence(int index, int start, int end) {
		while(Sequences.Length < (index+1)) {
			AddSequence();
		} 
		Sequences[index].SetStart(start);
		Sequences[index].SetEnd(end);
	}

	public void ResetSequences() {
		Sequences = new Sequence[0];
		AddSequence();
	}

	public bool ContainedInSequences(Frame frame) {
		foreach(Sequence s in Sequences) {
			if(s.Contains(frame.Index)) {
				return true;
			}
		}
		return false;
	}

	public bool IsFullyUnrolled() {
		bool[] selected = new bool[Frames.Length];
		foreach(Sequence seq in Sequences) {
			for(int i=seq.Start; i<=seq.End; i++) {
				selected[i-1] = true;
			}
		}
		return selected.All(true);
	}

	public Sequence GetUnrolledSequence() {
		if(Sequences.Length == 0) {
			return new Sequence(0, 0);
		}
		if(Sequences.Length == 1) {
			return Sequences[0];
		}
		int start = Frames.Length;
		int end = 1;
		foreach(Sequence seq in Sequences) {
			start = Mathf.Min(seq.Start, start);
			end = Mathf.Max(seq.End, end);
		}
		return new Sequence(start, end);
	}
	
	public void Load(MotionEditor editor) {
		//Check Missing Modules
		for(int i=0; i<Modules.Length; i++) {
			if(Modules[i] == null) {
				Debug.Log("Removing missing module in asset " + GetName() + ".");
				MarkDirty();
				ArrayExtensions.RemoveAt(ref Modules, i);
				i--;
			}
		}

		//Check Transformations
		bool repaired = false;
		foreach(Frame frame in Frames) {
			if(frame.Repair()) {
				repaired = true;
			}
		}
		if(repaired) {
			Debug.Log("Generated missing transformations in asset " + GetName() + ".");
			MarkDirty();
		}

		//Check Sequences
		if(Sequences.Length == 0) {
			AddSequence();
			Debug.Log("Generated missing sequence in asset " + GetName() + ".");
			MarkDirty();
		}

		//Send Load
		foreach(Module module in Modules) {
			module.Load(editor);
		}

		//Open Scene
		GetScene();
	}

	public string GetParentDirectoryPath() {
		return Path.GetDirectoryName(GetDirectoryPath());
	}

	public string GetDirectoryPath() {
		return Path.GetDirectoryName(AssetDatabase.GetAssetPath(this));
	}

	public string GetScenePath() {
		return GetDirectoryPath() + "/" + name + ".unity";
	}

	public Scene GetScene() {
		for(int i=0; i<SceneManager.sceneCount; i++) {
			Scene scene = SceneManager.GetSceneAt(i);
			if(scene.name == name) {
				if(!scene.isLoaded) {
					if(Application.isPlaying) {
						EditorSceneManager.LoadSceneInPlayMode(scene.path, new LoadSceneParameters(LoadSceneMode.Additive));
					} else {
						EditorSceneManager.OpenScene(scene.path, OpenSceneMode.Additive);
					}
				}
				return scene;
			}
		}
		if(Application.isPlaying) {
			if(VerifyScene()) {
				EditorSceneManager.LoadSceneInPlayMode(GetScenePath(), new LoadSceneParameters(LoadSceneMode.Additive));
			} else {
				Debug.Log("Creating temporary scene for data " + name + ".");
				SceneManager.CreateScene(name);			}
		} else {
			Scene active = EditorSceneManager.GetActiveScene();
			Lightmapping.bakedGI = false;
			Lightmapping.realtimeGI = false;
			if(VerifyScene()) {
				EditorSceneManager.OpenScene(GetScenePath(), OpenSceneMode.Additive);
			} else {
				Debug.Log("Recreating scene for data " + name + " in folder " + GetScenePath() + ".");
				EditorSceneManager.SaveScene(EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive), GetScenePath());
			}
			EditorSceneManager.SetActiveScene(SceneManager.GetSceneByName(name));
			Lightmapping.bakedGI = false;
			Lightmapping.realtimeGI = false;
			EditorSceneManager.SetActiveScene(active);
		}
		return SceneManager.GetSceneByName(name);

		bool VerifyScene() {
			string[] assets = AssetDatabase.FindAssets("t:Scene", new string[1]{GetDirectoryPath()});
			if(assets.Length == 0) {
				return false;
			}
			string path = AssetDatabase.GUIDToAssetPath(assets.First());
			string id = path.Substring(path.LastIndexOf("/")+1);
			id = id.Substring(0, id.LastIndexOf("."));
			return name == id;
		}
	}

	public void CreateScene() {
		UnityEngine.SceneManagement.Scene active = EditorSceneManager.GetActiveScene();
		UnityEngine.SceneManagement.Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
		EditorSceneManager.SetActiveScene(scene);
		Lightmapping.bakedGI = false;
		Lightmapping.realtimeGI = false;
		EditorSceneManager.SetActiveScene(active);
		EditorSceneManager.SaveScene(scene, GetScenePath());
		EditorSceneManager.CloseScene(scene, true);
	}

	public Actor CreateActor() {
		Actor actor = new GameObject("Skeleton").AddComponent<Actor>();
		List<Transform> instances = new List<Transform>();
		for(int i=0; i<Source.Bones.Length; i++) {
			Transform instance = new GameObject(Source.Bones[i].Name).transform;
			instance.SetParent(Source.Bones[i].Parent == "None" ? actor.GetRoot() : actor.FindTransform(Source.Bones[i].Parent));
			Matrix4x4 matrix = Frames.First().GetBoneTransformation(i, false);
			instance.position = matrix.GetPosition();
			instance.rotation = matrix.GetRotation();
			instance.localScale = Vector3.one;
			instances.Add(instance);
		}
		Transform root = actor.FindTransform(Source.Bones[0].Name);
		root.position = new Vector3(0f, root.position.y, 0f);
		root.rotation = Quaternion.Euler(root.eulerAngles.x, 0f, root.eulerAngles.z);
		actor.ExtractSkeleton(instances.ToArray());
		return actor;
	}

	private Module CreateModule(string id) {
		Module module = (Module)ScriptableObject.CreateInstance(id);
		if(module == null) {
			Debug.Log("Module of class type " + id + " could not be loaded in " + GetName() + ".");
		} else {
			ArrayExtensions.Append(ref Modules, module.Initialize(this));
			AssetDatabase.AddObjectToAsset(Modules.Last(), this);
		}
		return module;
	}

	public T AddModule<T>() {
		if(HasModule<T>()) {
			Debug.Log("Module of type " + typeof(T).Name + " already exists in " + GetName() + ".");
			return GetModule<T>();
		} else {
			return CreateModule(typeof(T).ToString()).ToType<T>();
		}
	}

	public void AddModule(Module.ID type) {
		if(System.Array.Find(Modules, x => x.GetID() == type) != null) {
			Debug.Log("Module of type " + type + " already exists in " + GetName() + ".");
		} else {
			CreateModule(type + "Module");
		}
	}

	public void RemoveModule<T>() {
		if(HasModule<T>()) {
			Module module = GetModule<T>().ToType<Module>();
			ArrayExtensions.Remove(ref Modules, module);
			Utility.Destroy(module);
		} else {
			Debug.Log("Module of type " + typeof(T).Name + " does not exist in " + GetName() + ".");
		}
	}

	public void RemoveModule(Module.ID type) {
		if(System.Array.Find(Modules, x => x.GetID() == type) == null) {
			Debug.Log("Module of type " + type + " does not exist in " + GetName() + ".");
		} else {
			ArrayExtensions.RemoveAt(ref Modules, System.Array.FindIndex(Modules, x => x.GetID() == type));
		}
	}

	public void RemoveAllModules() {
		while(Modules.Length > 0) {
			Utility.Destroy(Modules[0]);
			ArrayExtensions.RemoveAt(ref Modules, 0);
		}
	}

	public bool HasModule<T>() {
		return Modules.HasType<T>();
	}

	public T GetModule<T>() {
		return Modules.FindType<T>();
	}

	public void SampleTransformations(MotionData reference, params string[] bones) {
		if(reference == null) {
			Debug.Log("No reference specified.");
			return;
		}
		foreach(string bone in bones) {
			Hierarchy.Bone self = Source.FindBone(bone);
			Hierarchy.Bone other = reference.Source.FindBone(bone);
			if(self == null || other == null) {
				Debug.Log("No mapping found for resampling bone " + bone + ".");
			} else {
				foreach(Frame frame in Frames) {
					frame.Transformations[self.Index] = reference.SampleSourceTransformation(frame.Timestamp.Normalize(0f, GetTotalTime(), 0f, reference.GetTotalTime()), other.Index);
				}
			}
		}
	}

	public Matrix4x4 SampleSourceTransformation(float timestamp, int bone) {
		Frame frame = GetFrame(timestamp);
		if(timestamp < frame.Timestamp) {
			Frame previous = GetFrame(frame.Index-1);
			float ratio = (timestamp - previous.Timestamp) / (frame.Timestamp - previous.Timestamp);
			return Utility.Interpolate(previous.GetSourceTransformation(bone, false), frame.GetSourceTransformation(bone, false), ratio);
		}
		if(timestamp > frame.Timestamp) {
			Frame next = GetFrame(frame.Index+1);
			float ratio = (timestamp - frame.Timestamp) / (next.Timestamp - frame.Timestamp);
			return Utility.Interpolate(frame.GetSourceTransformation(bone, false), next.GetSourceTransformation(bone, false), ratio);
		}
		return frame.GetSourceTransformation(bone, false);
	}

	public void DetectSymmetry() {
		Symmetry = new int[Source.Bones.Length];
		for(int i=0; i<Source.Bones.Length; i++) {
			if(Source.Bones[i].Name.Contains("Left")) {
				int pivot = Source.Bones[i].Name.IndexOf("Left");
				Hierarchy.Bone bone = Source.FindBone(Source.Bones[i].Name.Substring(0, pivot)+"Right"+Source.Bones[i].Name.Substring(pivot+4));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(Source.Bones[i].Name.Contains("Right")) {
				int pivot = Source.Bones[i].Name.IndexOf("Right");
				Hierarchy.Bone bone = Source.FindBone(Source.Bones[i].Name.Substring(0, pivot)+"Left"+Source.Bones[i].Name.Substring(pivot+5));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(Source.Bones[i].Name.StartsWith("L") && char.IsUpper(Source.Bones[i].Name[1])) {
				Hierarchy.Bone bone = Source.FindBone("R"+Source.Bones[i].Name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(Source.Bones[i].Name.StartsWith("R") && char.IsUpper(Source.Bones[i].Name[1])) {
				Hierarchy.Bone bone = Source.FindBone("L"+Source.Bones[i].Name.Substring(1));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
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

	public void Inspector(MotionEditor editor) {
		Frame frame = editor.GetCurrentFrame();

		EditorGUI.BeginChangeCheck();

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();								
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("+", UltiDraw.LightGrey, Color.black, 40f, 20f * Sequences.Length)) {
				AddSequence();
			}
			EditorGUILayout.BeginVertical();
			for(int i=0; i<Sequences.Length; i++) {									
				EditorGUILayout.BeginHorizontal();
				GUILayout.FlexibleSpace();
				if(Utility.GUIButton("<", Color.cyan, Color.black, 20f, 15f)) {
					Sequences[i].SetStart(frame.Index);
				}
				// EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
				Sequences[i].SetStart(Mathf.Clamp(EditorGUILayout.IntField(Sequences[i].Start, GUILayout.Width(100f)), 1, GetTotalFrames()));
				// EditorGUILayout.LabelField("End", GUILayout.Width(50f));GetTotal
				Sequences[i].SetEnd(Mathf.Clamp(EditorGUILayout.IntField(Sequences[i].End, GUILayout.Width(100f)), 1, GetTotalFrames()));
				if(Utility.GUIButton(">", Color.cyan, Color.black, 20f, 15f)) {
					Sequences[i].SetEnd(frame.Index);
				}
				GUILayout.FlexibleSpace();
				EditorGUI.BeginDisabledGroup(Sequences.Length == 1);
				if(Utility.GUIButton("-", UltiDraw.DarkRed, Color.black, 40f, 15f)) {
					RemoveSequence(Sequences[i]);
					i--;
				}
				EditorGUI.EndDisabledGroup();
				EditorGUILayout.EndHorizontal();
			}
			EditorGUILayout.EndVertical();
			EditorGUILayout.EndHorizontal();
		}

		foreach(Module m in Modules) {
			m.Inspector(editor);
		}

		Utility.ResetGUIColor();
		Utility.SetGUIColor(UltiDraw.White);
		int module = EditorGUILayout.Popup(0, ArrayExtensions.Concat(new string[1]{"Add Module..."}, Module.GetIDs()));
		if(module > 0) {
			AddModule((Module.ID)(module-1));
		}
		Utility.ResetGUIColor();

		Utility.SetGUIColor(UltiDraw.DarkGrey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			if(Utility.GUIButton("Advanced", editor.Advanced ? UltiDraw.Cyan : UltiDraw.DarkGrey, editor.Advanced ? UltiDraw.Black : UltiDraw.White)) {
				editor.Advanced = !editor.Advanced;
			}

			if(editor.Advanced) {
				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Offset = EditorGUILayout.Vector3Field("Offset", Offset);
					Scale = EditorGUILayout.FloatField("Scale", Scale);
					MirrorAxis = (Axis)EditorGUILayout.EnumPopup("Mirror Axis", MirrorAxis);

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Exportable", Export ? UltiDraw.Cyan : UltiDraw.Grey, Export ? UltiDraw.Black : UltiDraw.LightGrey)) {
						Export = !Export;
					}
					if(Utility.GUIButton("Tagged", Tagged ? UltiDraw.Cyan : UltiDraw.Grey, Tagged ? UltiDraw.Black : UltiDraw.LightGrey)) {
						Tagged = !Tagged;
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Mark As First Valid Frame", UltiDraw.DarkGrey, UltiDraw.White)) {
						SetFirstValidFrame(editor.GetCurrentFrame());
					}
					if(Utility.GUIButton("Mark As Last Valid Frame", UltiDraw.DarkGrey, UltiDraw.White)) {
						SetLastValidFrame(editor.GetCurrentFrame());
					}
					EditorGUILayout.EndHorizontal();

					if(Utility.GUIButton("Inspect Hierarchy", editor.InspectHierarchy ? UltiDraw.Cyan : UltiDraw.DarkGrey, editor.InspectHierarchy ? UltiDraw.Black : UltiDraw.White)) {
						editor.InspectHierarchy = !editor.InspectHierarchy;
					}
					if(editor.InspectHierarchy) {
						for(int i=0; i<Source.Bones.Length; i++) {
							EditorGUILayout.BeginHorizontal();
							EditorGUI.BeginDisabledGroup(true);
							EditorGUILayout.TextField(Source.GetBoneNames()[i]);
							EditorGUI.EndDisabledGroup();
							SetSymmetry(i, EditorGUILayout.Popup(Symmetry[i], Source.GetBoneNames()));
							EditorGUILayout.LabelField("Alignment", GUILayout.Width(60f));
							Source.Bones[i].Alignment = EditorGUILayout.Vector3Field("", Source.Bones[i].Alignment);
							EditorGUILayout.EndHorizontal();
						}
					}
				}
			}
		}

		if(EditorGUI.EndChangeCheck()) {
			MarkDirty();
		}
	}

	public void Callback(MotionEditor editor) {
		foreach(Module m in Modules) {
			m.Callback(editor);
		}
	}

	public void GUI(MotionEditor editor) {
		foreach(Module m in Modules) {
			m.GUI(editor);
		}
	}

	public void Draw(MotionEditor editor) {
		foreach(Module m in Modules) {
			m.Draw(editor);
		}
	}

	[System.Serializable]
	public class Hierarchy {
		public Bone[] Bones;

		private string[] Names = null;

		public Hierarchy() {
			Bones = new Bone[0];
		}

		public void AddBone(string name, string parent) {
			ArrayExtensions.Append(ref Bones, new Bone(Bones.Length, name, parent));
		}

		public Bone FindBone(string name) {
			return System.Array.Find(Bones, x => x.Name == name);
		}

		public Bone FindBoneContains(params string[] names) {
			return System.Array.Find(Bones, x => x.Name.Contains(names));
		}

		public string[] GetBoneNames() {
			if(Names == null || Names.Length != Bones.Length) {
				Names = new string[Bones.Length];
				for(int i=0; i<Bones.Length; i++) {
					Names[i] = Bones[i].Name;
				}
			}
			return Names;
		}

		public string[] GetBoneNames(params int[] bones) {
			string[] names = new string[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				names[i] = Bones[i].Name;
			}
			return names;
		}

		public bool[] GetBoneFlags(params string[] bones) {
			bool[] flags = new bool[Bones.Length];
			for(int i=0; i<bones.Length; i++) {
				Bone bone = FindBone(bones[i]);
				if(bone != null) {
					flags[bone.Index] = true;
				}
			}
			return flags;
		}

		public int[] GetBoneIndices(params string[] bones) {
			int[] indices = new int[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				Bone bone = FindBone(bones[i]);
				indices[i] = bone == null ? -1 : bone.Index;
			}
			return indices;
		}

		[System.Serializable]
		public class Bone {
			public int Index = -1;
			public string Name = "";
			public string Parent = "";
			public Vector3 Alignment = Vector3.zero;
			public Bone(int index, string name, string parent) {
				Index = index;
				Name = name;
				Parent = parent;
				Alignment = Vector3.zero;
			}
		}
	}
}
#endif