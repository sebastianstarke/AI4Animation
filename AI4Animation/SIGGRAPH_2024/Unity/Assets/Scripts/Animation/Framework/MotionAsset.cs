using UnityEngine;
using UnityEngine.SceneManagement;
using System.IO;
using System.Collections.Generic;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

namespace AI4Animation {
	public class MotionAsset : ScriptableObject {
		public Hierarchy Source = null;
		public Frame[] Frames = new Frame[0];
		public Module[] Modules = new Module[0];
		public Interval[] Sequences = new Interval[0];

		public float Framerate = 1f;

		public string Model = string.Empty;
		public Vector3 Translation = Vector3.zero;
		public Vector3 Rotation = Vector3.zero;
		public float Scale = 1f;
		public Axis MirrorAxis = Axis.XPositive;
		public bool Export = true;
		public bool Visualize = true;
		public int[] Symmetry = new int[0];

		public static bool Settings = false;

#if UNITY_EDITOR
		public static string[] Search(params string[] folders) {
			return folders.Length == 0 ? new string[0] : AssetDatabase.FindAssets("t:MotionAsset", folders);
		}
		
		public static MotionAsset Retrieve(string guid) {
				return (MotionAsset)AssetDatabase.LoadMainAssetAtPath(Utility.GetAssetPath(guid));
		}
		
		public void MarkDirty(bool asset=true, bool scene=true) {
			if(!Application.isPlaying) {
				if(asset) {
					EditorUtility.SetDirty(this);
				}
				if(scene) {
					EditorSceneManager.MarkSceneDirty(GetScene());
				}
			}
		}

		public bool IsDirty() {
			return EditorUtility.IsDirty(this) || GetScene().isDirty;
		}
		
		public void MakeCopy(string destination) {
			string source = GetDirectoryPath();
			if(!Directory.Exists(destination)) {
				Directory.CreateDirectory(destination);
				AssetDatabase.CopyAsset(source + "/" + name + ".asset", destination + "/" + name + ".asset");
				AssetDatabase.CopyAsset(source + "/" + name + ".unity", destination + "/" + name + ".unity");
				Debug.Log("Copied asset " + name + " from " + source + " to " + destination + ".");
			} else {
				Debug.Log("Failed copying asset because directory " + destination + " already exists.");
			}
		}
#endif
		public void SetModel(string name) {
			if(Model != name) {
				Model = name;
			}
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

		public void AddSequence() {
			ArrayExtensions.Append(ref Sequences, new Interval(1, GetTotalFrames()));
		}

		public void AddSequence(int start, int end) {
			ArrayExtensions.Append(ref Sequences, new Interval(start, end));
		}

		public void RemoveSequence(Interval sequence) {
			ArrayExtensions.Remove(ref Sequences, sequence);
		}

		public void SetSequence(int index, int start, int end) {
			while(Sequences.Length < (index+1)) {
				AddSequence();
			} 
			Sequences[index].SetStart(start);
			Sequences[index].SetEnd(end);
		}

		public bool InSequences(Frame frame) {
			foreach(Interval seq in Sequences) {
				if(seq.Contains(frame.Index)) {
					return true;
				}
			}
			return false;
		}

		public void ClearSequences() {
			Sequences = new Interval[0];
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
		}

		//Returns absolute timestamps around frame at framerate steps by time padding.
		public float[] SimulateTimestamps(float timestamp, float window) {
			float step = 1f/Framerate;
			float padding = window/2f;
			float start = timestamp - padding;
			float[] timestamps = new float[2*Mathf.RoundToInt(padding/step)+1];
			for(int i=0; i<timestamps.Length; i++) {
				timestamps[i] = start + i*step;
			}
			return timestamps;
		}

		//Returns absolute timestamps around frame at framerate steps by time padding.
		public float[] SimulateTimestamps(float window) {
			return SimulateTimestamps(0f, window);
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

#if UNITY_EDITOR
		public void Load(MotionEditor editor) {			
			//Check Missing Modules
			{
				bool removed = false;
				for(int i=0; i<Modules.Length; i++) {
					if(Modules[i] == null) {
						ArrayExtensions.RemoveAt(ref Modules, i);
						i--;
						removed = true;
					}
				}
				if(removed) {
					Debug.Log("Removed missing modules in asset " + name + ".");
					MarkDirty();
				}
			}

			//Check Frame References
			{
				bool restored = false;
				for(int i=0; i<Frames.Length; i++) {
					if(Frames[i].Asset != this) {
						Frames[i].Asset = this;
						restored = true;
					}
				}
				if(restored) {
					Debug.Log("Restored missing frame references in asset " + this.name + ".");
					MarkDirty();
				}
			}

			//Check Module References
			{
				bool restored = false;
				for(int i=0; i<Modules.Length; i++) {
					if(Modules[i].Asset != this) {
						Modules[i].Asset = this;
						restored = true;
					}
				}
				if(restored) {
					Debug.Log("Restored missing module references in asset " + this.name + ".");
					MarkDirty();
				}
			}

			// //Check Sequences
			// if(Sequences.Length == 0) {
			// 	AddSequence();
			// 	Debug.Log("Generated missing sequence in asset " + this.name + ".");
			// 	MarkDirty();
			// }

			//Send Load
			foreach(Module module in Modules) {
				module.Load(editor);
			}

			//Load Scene
			GetScene();
		}

		public void Unload(MotionEditor editor) {
			//Send Unload
			foreach(Module module in Modules) {
				module.Unload(editor);
			}

			//Unload Scene
			if(Application.isPlaying) {
				EditorSceneManager.UnloadSceneAsync(EditorSceneManager.GetSceneByName(name));
			} else {
				EditorSceneManager.CloseScene(EditorSceneManager.GetSceneByName(name), true);
			}
			EditorApplication.RepaintHierarchyWindow();
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

		public GameObject GetSceneGameObject(string name) {
			return GameObjectExtensions.FindRecursively(GetScene().GetRootGameObjects(), name);			
		}

		public Scene GetScene() {
			for(int i=0; i<SceneManager.sceneCount; i++) {
				if(SceneManager.GetSceneAt(i).name == name) {
					return SceneManager.GetSceneAt(i);
				}
			}
			if(Application.isPlaying) {
				if(VerifyScene()) {
					EditorSceneManager.LoadSceneInPlayMode(GetScenePath(), new LoadSceneParameters(LoadSceneMode.Additive));
				} else {
					Debug.Log("Creating temporary scene for data " + name + ".");
					SceneManager.CreateScene(name);
				}
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
				if(active.isLoaded) {
					EditorSceneManager.SetActiveScene(active);
				}
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

		private Module CreateModule(string type, string tag) {
			Module module = (Module)ScriptableObject.CreateInstance(type);
			if(module == null) {
				Debug.Log("Module of type " + type + " could not be loaded in " + name + ".");
			} else {
				ArrayExtensions.Append(ref Modules, module.Initialize(this, tag));
				AssetDatabase.AddObjectToAsset(Modules.Last(), this);
			}
			return module;
		}

		public T AddModule<T>(string tag=null) where T : Module {
			if(HasModule<T>(tag)) {
				Debug.Log("Module of type " + typeof(T).Name + (tag == null ? string.Empty : " with tag " + tag) + " already exists in " + name + ".");
				return null;
			} else {
				return CreateModule(typeof(T).ToString(), tag).ToType<T>();
			}
		}
#endif
		public T GetModule<T>(string tag=null) where T : Module {
			T[] modules = Modules.FindTypes<T>();
			List<T> subset = new List<T>();
			foreach(T m in modules) {
				if(tag == null || tag == null && m.Tag == string.Empty || tag == m.Tag) {
					subset.Add(m);
				}
			}
			modules = subset.ToArray();
			if(modules.Length == 0) {
				Debug.Log("Module of type " + typeof(T).Name + (tag == null ? string.Empty : " with tag " + tag) + " could not be found in " + name + ".");
				return null;
			}
			if(modules.Length > 1) {
				Debug.Log("Multiple modules of type " + typeof(T).Name + (tag == null ? string.Empty : " with same tag " + tag) + " found in " + name + ". Please specify a unique tag.");
				return null;
			}
			return modules[0];
		}

		public T[] GetModules<T>() where T : Module {
			return Modules.FindTypes<T>();
		}

#if UNITY_EDITOR
		public T AddOrGetModule<T>(string tag=null) where T : Module {
			return HasModule<T>(tag) ? GetModule<T>(tag) : AddModule<T>(tag);
		}
#endif
		public bool HasModule<T>(string tag=null) where T : Module {
			foreach(T m in Modules.FindTypes<T>()) {
				if(tag == null && m.Tag == string.Empty || tag == m.Tag) {
					return true;
				}
			}
			return false;
		}
#if UNITY_EDITOR
		public void RemoveModule<T>(string tag=null) where T : Module {
			Module module = GetModule<T>(tag);
			if(module != null) {
				RemoveModule(module);
			}
		}

		public void RemoveModule(Module module) {
			if(Modules.Contains(module)) {
				ArrayExtensions.Remove(ref Modules, module);
				Utility.Destroy(module);
			} else {
				Debug.Log("Given module does not exist in " + name + ".");
			}
		}

		public void RemoveAllModules() {
			while(Modules.Length > 0) {
				Utility.Destroy(Modules[0]);
				ArrayExtensions.RemoveAt(ref Modules, 0);
			}
		}

		public void RemoveAllModules<T>() where T : Module {
			foreach(T m in Modules.FindTypes<T>()) {
				RemoveModule(m);
			}
		}
#endif

		public void DetectSymmetry() {
			bool TryAssign(string value, int bone) {
				Hierarchy.Bone other = Source.FindBone(value);
				if(other != null) {
					Symmetry[bone] = other.Index;
					return true;
				} else {
					return false;
				}
			}

			Symmetry = new int[Source.Bones.Length];
			for(int i=0; i<Source.Bones.Length; i++) {
				string source = Source.Bones[i].Name;

				if(source.Contains("Left")) {
					if(TryAssign(source.Replace("Left", "Right"), i)) {continue;}
				}
				if(source.Contains("Right")) {
					if(TryAssign(source.Replace("Right", "Left"), i)) {continue;}

				}
				if(source.Contains("left")) {
					if(TryAssign(source.Replace("left", "right"), i)) {continue;}
				}
				if(source.Contains("right")) {
					if(TryAssign(source.Replace("right", "left"), i)) {continue;}
				}

				if(source.Contains("_L_")) {
					if(TryAssign(source.Replace("_L_", "_R_"), i)) {continue;}
				}
				if(source.Contains("_R_")) {
					if(TryAssign(source.Replace("_R_", "_L_"), i)) {continue;}
				}
				if(source.Contains("L_")) {
					if(TryAssign(source.Replace("L_", "R_"), i)) {continue;}
				}
				if(source.Contains("R_")) {
					if(TryAssign(source.Replace("R_", "L_"), i)) {continue;}
				}
				if(source.Contains("_L")) {
					if(TryAssign(source.Replace("_L", "_R"), i)) {continue;}
				}
				if(source.Contains("_R")) {
					if(TryAssign(source.Replace("_R", "_L"), i)) {continue;}
				}

				if(source.Contains("_l_")) {
					if(TryAssign(source.Replace("_l_", "_r_"), i)) {continue;}
				}
				if(source.Contains("_r_")) {
					if(TryAssign(source.Replace("_r_", "_l_"), i)) {continue;}
				}
				if(source.Contains("l_")) {
					if(TryAssign(source.Replace("l_", "r_"), i)) {continue;}
				}
				if(source.Contains("r_")) {
					if(TryAssign(source.Replace("r_", "l_"), i)) {continue;}
				}
				if(source.Contains("_l")) {
					if(TryAssign(source.Replace("_l", "_r"), i)) {continue;}
				}
				if(source.Contains("_r")) {
					if(TryAssign(source.Replace("_r", "_l"), i)) {continue;}
				}

				if(source.Contains("1")) {
					if(TryAssign(source.Replace("1", ""), i)) {continue;}
				}
				if(TryAssign(source += "1", i)) {continue;}
				

				Symmetry[i] = i;
			}
		}

		public void SetSymmetry(int source, int target) {
			if(Symmetry[source] != target) {
				Symmetry[source] = target;
			}
		}

#if UNITY_EDITOR
		public void Inspector(MotionEditor editor) {
			Frame frame = editor.GetCurrentFrame();

			EditorGUI.BeginChangeCheck();

			foreach(Module m in Modules) {
				m.Inspector(editor);
			}

			Utility.ResetGUIColor();
			Utility.SetGUIColor(UltiDraw.White);
			int module = EditorGUILayout.Popup(0, ArrayExtensions.Concat(new string[1]{"Add Module..."}, Module.GetTypes()));
			if(module > 0) {
				void AddModule(string type) {
					CreateModule(type, null);
				}
				AddModule(Module.GetTypes()[module-1]);
			}
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				if(Utility.GUIButton("Settings", Settings ? UltiDraw.Cyan : UltiDraw.DarkGrey, Settings ? UltiDraw.Black : UltiDraw.White)) {
					Settings = !Settings;
				}

				if(Settings) {
					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						
						SetModel(EditorGUILayout.TextField("Model", Model));
						Translation = EditorGUILayout.Vector3Field("Translation", Translation);
						Rotation = EditorGUILayout.Vector3Field("Rotation", Rotation);
						Scale = EditorGUILayout.FloatField("Scale", Scale);
						MirrorAxis = (Axis)EditorGUILayout.EnumPopup("Mirror Axis", MirrorAxis);
						if(Utility.GUIButton("Export", Export ? UltiDraw.DarkGreen : UltiDraw.DarkRed, UltiDraw.White)) {
							Export = !Export;
						}

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

						Utility.SetGUIColor(UltiDraw.White);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							if(Utility.GUIButton("Detect Symmetry", UltiDraw.DarkGrey, UltiDraw.White)) {
								DetectSymmetry();
							}
							for(int i=0; i<Source.Bones.Length; i++) {
								EditorGUI.BeginDisabledGroup(true);
								EditorGUILayout.TextField(Source.GetBoneNames()[i]);
								EditorGUI.EndDisabledGroup();
								Utility.SetGUIColor(UltiDraw.White);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									SetSymmetry(i, EditorGUILayout.Popup("Symmetry Bone", Symmetry[i], Source.GetBoneNames()));
									Source.Bones[i].Parent = EditorGUILayout.Popup("Parent Bone", Source.Bones[i].Parent+1, ArrayExtensions.Concat(new string[]{"None"}, Source.GetBoneNames())) - 1; 
									Source.Bones[i].Override = EditorGUILayout.TextField("Override Name", Source.Bones[i].Override);
									Source.Bones[i].Alignment = EditorGUILayout.Vector3Field("Alignment Rotation", Source.Bones[i].Alignment);
									Source.Bones[i].Correction = EditorGUILayout.Vector3Field("Correction Rotation", Source.Bones[i].Correction);
								}
							}
						}
					}
				}
			}

			if(EditorGUI.EndChangeCheck()) {
				editor.LoadFrame(editor.GetTimestamp());
				MarkDirty();
			}
		}

		public void OnTriggerPlay(MotionEditor editor) {
			foreach(Module m in Modules) {
				m.OnTriggerPlay(editor);
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
#endif

		[System.Serializable]
		public class Hierarchy {
			public Bone[] Bones;

			public Hierarchy() {
				Bones = new Bone[0];
			}

			public Hierarchy(int bones) {
				Bones = new Bone[bones];
				for(int i=0; i<Bones.Length; i++) {
					Bones[i] = new Bone(i, i.ToString(), "None", this);
				}
			}

			public Hierarchy(string[] bones) {
				Bones = new Bone[bones.Length];
				for(int i=0; i<Bones.Length; i++) {
					Bones[i] = new Bone(i, bones[i], "None", this);
				}
			}

			public Hierarchy(string[] bones, string[] parents) {
				Bones = new Bone[bones.Length];
				for(int i=0; i<Bones.Length; i++) {
					Bones[i] = new Bone(i, bones[i], parents[i], this);
				}
			}

			public void AddBone(string name, string parent) {
				ArrayExtensions.Append(ref Bones, new Bone(Bones.Length, name, parent, this));
			}

			public void SetBone(int index, string name, string parent) {
				Bones[index] = new Bone(index, name, parent, this);
			}

			public bool HasBone(string name) {
				return FindBone(name) != null;
			}

			public Bone FindBone(string name) {
				Bone bone = System.Array.Find(Bones, x => x != null && x.Override == name);
				// if(bone == null) {
				// 	Debug.Log("Bone " + name + " could not be found.");
				// }
				return bone != null ? bone : System.Array.Find(Bones, x => x != null && x.Name == name);
			}

			public Bone FindBoneAny(params string[] names) {
				for(int i=0; i<names.Length; i++) {
					Bone bone = FindBone(names[i]);
					if(bone != null) {
						return bone;
					}
				}
				return null;
			}
			
			public Bone FindBoneContains(string name) {
				Bone bone = System.Array.Find(Bones, x => x.Override.Contains(name));
				return bone != null ? bone : System.Array.Find(Bones, x => x.Name.Contains(name));
				// return System.Array.Find(Bones, x => x.Name.Contains(name));
			}

			public Bone FindBoneContainsAny(params string[] names) {
				Bone bone = System.Array.Find(Bones, x => x.Override.ContainsAny(names));
				return bone != null ? bone : System.Array.Find(Bones, x => x.Name.ContainsAny(names));
				// return System.Array.Find(Bones, x => x.Name.ContainsAny(names));
			}

			public void ConvertBoneNames(string from, string to) {
				foreach(Bone bone in Bones) {
					bone.Name = bone.Name.Replace(from, to);
				}
				// Names = null;
			}

			public string[] GetBoneNames() {
				string[] names = new string[Bones.Length];
				for(int i=0; i<Bones.Length; i++) {
					names[i] = Bones[i].Name;
				}
				return names;
			}

			public string[] GetBoneNames(params int[] bones) {
				string[] names = new string[bones.Length];
				for(int i=0; i<bones.Length; i++) {
					names[i] = Bones[bones[i]].Name;
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

			public int GetBoneIndex(string bone) {
				Bone result = FindBone(bone);
				return result == null ? -1 : result.Index; 
			}

			public int[] GetBoneIndices(params string[] bones) {
				int[] indices = new int[bones.Length];
				for(int i=0; i<bones.Length; i++) {
					indices[i] = GetBoneIndex(bones[i]);
				}
				return indices;
			}

			[System.Serializable]
			public class Bone {
				public int Index = -1;
				public int Parent = -1;
				public string Name = string.Empty;
				public string Override = string.Empty;
				public Vector3 Alignment = Vector3.zero;
				public Vector3 Correction = Vector3.zero;
				public Bone() {
					Index = -1;
					Name = "None";
					Parent = -1;
					Override = string.Empty;
					Alignment = Vector3.zero;
					Correction = Vector3.zero;
				}
				public Bone(int index, string name, string parent, Hierarchy hierarchy) {
					Index = index;
					Name = name;
					Parent = parent == "None" ? -1 : hierarchy.FindBone(parent).Index;
					Override = string.Empty;
					Alignment = Vector3.zero;
					Correction = Vector3.zero;
				}
				public string GetName() {
					return Override == string.Empty ? Name : Override;
				}
				public bool HasParent() {
					return Parent != -1;
				}
				public Bone GetParent(Hierarchy hierarchy) {
					return HasParent() ? hierarchy.Bones[Parent] : null;
				}
			}
		}
	}
}
