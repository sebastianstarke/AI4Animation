#if UNITY_EDITOR
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

[ExecuteInEditMode]
public class MotionEditor : MonoBehaviour {

	public string Folder = string.Empty;
	public File[] Files = new File[0];

	//public bool ShowMotion = false;
	public bool ShowMirror = false;
	public bool ShowVelocities = false;
	public bool ShowTrajectory = false;
	public bool InspectSettings = false;

	private bool AutoFocus = false;
	private float FocusHeight = 1f;
	private float FocusOffset = 0f;
	private float FocusDistance = 2.5f;
	private float FocusAngle = 0f;
	private float FocusSmoothing = 0.05f;

	private bool Playing = false;
	private float Timescale = 1f;
	private float Timestamp = 0f;
	private float Window = 1f;

	private Actor Actor = null;
	private Transform Environment = null;
	private State State = null;

	private File Instance = null;

	public float GetWindow() {
		return GetFile() == null ? 0f : Window * GetFile().Data.GetTotalTime();
	}

	public void SetMirror(bool value) {
		if(ShowMirror != value) {
			ShowMirror = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetScaling(float value) {
		if(GetFile().Data.Scaling != value) {
			GetFile().Data.Scaling = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetAutoFocus(bool value) {
		if(AutoFocus != value) {
			AutoFocus = value;
			if(!AutoFocus) {
				Vector3 position =  SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion rotation = Quaternion.Euler(0f, SceneView.lastActiveSceneView.camera.transform.rotation.eulerAngles.y, 0f);
				SceneView.lastActiveSceneView.LookAtDirect(position, rotation, 0f);
			}
		}
	}

	public Actor GetActor() {
		if(Actor == null) {
			Actor = transform.GetComponentInChildren<Actor>();
		}
		if(Actor == null) {
			Actor = CreateSkeleton();
		}
		return Actor;
	}

	public File[] GetFiles() {
		return Files;
	}

	public Transform GetEnvironment() {
		if(Environment == null) {
			Environment = transform.Find("Environment");
		}
		if(Environment == null) {
			Environment = new GameObject("Environment").transform;
			Environment.SetParent(transform);
		}
		return Environment;
	}

	public File GetFile() {
		return Instance;
	}

	public State GetState() {
		if(State == null) {
			LoadFrame(Timestamp);
		}
		return State;
	}

	public void Import() {
		string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{Folder});
		List<File> files = new List<File>();
		for(int i=0; i<assets.Length; i++) {
			File file = new File();
			file.Index = i;
			file.Data = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[i]), typeof(MotionData));
			files.Add(file);
		}
		Files = files.ToArray();
		for(int i=0; i<GetEnvironment().childCount; i++) {
			if(!System.Array.Exists(Files, x => x.Data.Name == GetEnvironment().GetChild(i).name)) {
				Utility.Destroy(GetEnvironment().GetChild(i).gameObject);
				i--;
			}
		}
		for(int i=0; i<Files.Length; i++) {
			Files[i].Environment = GetEnvironment().Find(Files[i].Data.Name);
			if(Files[i].Environment == null) {
				Files[i].Environment = new GameObject(Files[i].Data.Name).transform;
				Files[i].Environment.SetParent(GetEnvironment());
			}
		}
		//Finalise
		for(int i=0; i<Files.Length; i++) {
			Files[i].Environment.gameObject.SetActive(false);
			Files[i].Environment.SetSiblingIndex(i);
		}
		Initialise();
	}

	public void Initialise() {
		if(Instance != null) {
			if(Instance.Data == null || Instance.Environment == null) {
				Instance = null;
			}
		}
		for(int i=0; i<Files.Length; i++) {
			if(Files[i].Data == null) {
				Utility.Destroy(Files[i].Environment.gameObject);
				ArrayExtensions.RemoveAt(ref Files, i);
				i--;
			}
		}
		if(GetFile() == null && Files.Length > 0) {
			LoadFile(Files[0]);
		}
	}
	
	public void Save(File file) {
		if(file != null) {
			EditorUtility.SetDirty(file.Data);
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
		}
	}

	public void LoadFile(File file) {
		if(Instance != file) {
			if(Instance != null) {
				Instance.Environment.gameObject.SetActive(false);
				Save(Instance);
			}
			Instance = file;
			if(Instance != null) {
				Instance.Environment.gameObject.SetActive(true);
				LoadFrame(0f);
			}
		}
	}

	public void LoadFrame(State state) {
		Timestamp = state.Timestamp;
		State = state;
		if(state.Mirrored) {
			GetEnvironment().localScale = Vector3.one.GetMirror(GetFile().Data.GetAxis(GetFile().Data.MirrorAxis));
		} else {
			GetEnvironment().localScale = Vector3.one;
		}

		GetActor().GetRoot().position = GetState().Root.GetPosition();
		GetActor().GetRoot().rotation = GetState().Root.GetRotation();
		for(int i=0; i<Mathf.Min(GetActor().Bones.Length, GetState().BoneTransformations.Length); i++) {
			GetActor().Bones[i].Transform.position = GetState().BoneTransformations[i].GetPosition();
			GetActor().Bones[i].Transform.rotation = GetState().BoneTransformations[i].GetRotation();
		}

		if(AutoFocus) {
			if(SceneView.lastActiveSceneView != null) {
				Vector3 lastPosition = SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion lastRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
				Vector3 position = GetState().Root.GetPosition();
				position.y += FocusHeight;
				Quaternion rotation = GetState().Root.GetRotation();
				rotation.x = 0f;
				rotation.z = 0f;
				rotation = Quaternion.Euler(0f, ShowMirror ? Mathf.Repeat(FocusAngle + 0f, 360f) : FocusAngle, 0f) * rotation;
				position += FocusOffset * (rotation * Vector3.right);
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
			}
		}
	}

	public void LoadFrame(float timestamp) {
		LoadFrame(new State(GetFile().Data.GetFrame(timestamp), ShowMirror));
	}

	public void LoadFrame(int index) {
		LoadFrame(GetFile().Data.GetFrame(index).Timestamp);
	}

	public void LoadPreviousFrame() {
		LoadFrame(Mathf.Max(GetFile().Data.GetFrame(Timestamp).Index - 1, 1));
	}

	public void LoadNextFrame() {
		LoadFrame(Mathf.Min(GetFile().Data.GetFrame(Timestamp).Index + 1, GetFile().Data.GetTotalFrames()));
	}

	public void PlayAnimation() {
		if(Playing) {
			return;
		}
		Playing = true;
		EditorCoroutines.StartCoroutine(Play(), this);
	}

	public void StopAnimation() {
		if(!Playing) {
			return;
		}
		Playing = false;
		EditorCoroutines.StopCoroutine(Play(), this);
	}

	private IEnumerator Play() {
		System.DateTime timestamp = Utility.GetTimestamp();
		while(GetFile() != null) {
			Timestamp += Timescale * (float)Utility.GetElapsedTime(timestamp);
			if(Timestamp > GetFile().Data.GetTotalTime()) {
				Timestamp = Mathf.Repeat(Timestamp, GetFile().Data.GetTotalTime());
			}
			timestamp = Utility.GetTimestamp();
			LoadFrame(Timestamp);
			yield return new WaitForSeconds(0f);
		}
	}

	public Actor CreateSkeleton() {
		if(GetFile() == null) {
			return null;
		}
		Actor actor = new GameObject("Skeleton").AddComponent<Actor>();
		actor.transform.SetParent(transform);
		string[] names = new string[GetFile().Data.Source.Bones.Length];
		string[] parents = new string[GetFile().Data.Source.Bones.Length];
		for(int i=0; i<GetFile().Data.Source.Bones.Length; i++) {
			names[i] = GetFile().Data.Source.Bones[i].Name;
			parents[i] = GetFile().Data.Source.Bones[i].Parent;
		}
		List<Transform> instances = new List<Transform>();
		for(int i=0; i<names.Length; i++) {
			Transform instance = new GameObject(names[i]).transform;
			instance.SetParent(parents[i] == "None" ? actor.GetRoot() : actor.FindTransform(parents[i]));
			instances.Add(instance);
		}
		actor.ExtractSkeleton(instances.ToArray());
		return actor;
	}

	public void CopyHierarchy()  {
		for(int i=0; i<GetFile().Data.Source.Bones.Length; i++) {
			if(GetActor().FindBone(GetFile().Data.Source.Bones[i].Name) != null) {
				GetFile().Data.Source.Bones[i].Active = true;
			} else {
				GetFile().Data.Source.Bones[i].Active = false;
			}
		}
	}

	public void Draw() {
		if(GetFile() == null) {
			return;
		}

		//if(ShowMotion) {
		//	for(int i=0; i<GetState().PastBoneTransformations.Count; i++) {
		//		GetActor().DrawSimple(Color.Lerp(UltiDraw.Blue, UltiDraw.Cyan, 1f - (float)(i+1)/6f).Transparent(0.75f), GetState().PastBoneTransformations[i]);
		//	}
		//	for(int i=0; i<GetState().FutureBoneTransformations.Count; i++) {
		//		GetActor().DrawSimple(Color.Lerp(UltiDraw.Red, UltiDraw.Orange, (float)i/5f).Transparent(0.75f), GetState().FutureBoneTransformations[i]);
		//	}
		//}
	
		if(ShowVelocities) {
			UltiDraw.Begin();
			for(int i=0; i<GetState().BoneVelocities.Length; i++) {
				UltiDraw.DrawArrow(
					GetActor().Bones[i].Transform.position,
					GetActor().Bones[i].Transform.position + GetState().BoneVelocities[i],
					0.75f,
					0.0075f,
					0.05f,
					UltiDraw.Purple.Transparent(0.5f)
				);
			}
			UltiDraw.End();
		}

		if(ShowTrajectory) {
			GetState().Trajectory.Draw();
		}

		for(int i=0; i<GetFile().Data.Modules.Length; i++) {
			GetFile().Data.Modules[i].Draw(this);
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	[System.Serializable]
	public class File {
		public int Index;
		public MotionData Data;
		public Transform Environment;
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		private float RefreshRate = 30f;
		private System.DateTime Timestamp;

		public int Index = 0;
		public File[] Instances = new File[0];
		public string[] Names = new string[0];
		public string NameFilter = "";
		public bool ExportFilter = false;
		public bool ExcludeFilter = false;

		void Awake() {
			Target = (MotionEditor)target;
			Target.Initialise();
			Filter();
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			if(!Application.isPlaying && Target != null) {
				Target.Save(Target.GetFile());
			}
			EditorApplication.update -= EditorUpdate;
		}

		public void EditorUpdate() {
			if(Utility.GetElapsedTime(Timestamp) >= 1f/RefreshRate) {
				Repaint();
				Timestamp = Utility.GetTimestamp();
			}
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);
			Inspector();
			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
		
		private void Filter() {
			List<File> instances = new List<File>();
			if(NameFilter == string.Empty) {
				instances.AddRange(Target.Files);
			} else {
				for(int i=0; i<Target.Files.Length; i++) {
					if(Target.Files[i].Data.Name.ToLowerInvariant().Contains(NameFilter.ToLowerInvariant())) {
						instances.Add(Target.Files[i]);
					}
				}
			}
			if(ExportFilter) {
				for(int i=0; i<instances.Count; i++) {
					if(!instances[i].Data.Export) {
						instances.RemoveAt(i);
						i--;
					}
				}
			}
			if(ExcludeFilter) {
				for(int i=0; i<instances.Count; i++) {
					if(instances[i].Data.Export) {
						instances.RemoveAt(i);
						i--;
					}
				}
			}
			Instances = instances.ToArray();
			Names = new string[Instances.Length];
			for(int i=0; i<Instances.Length; i++) {
				Names[i] = Instances[i].Data.Name;
			}
			LoadFile(GetIndex());
		}

		public void SetNameFilter(string filter) {
			if(NameFilter != filter) {
				NameFilter = filter;
				Filter();
			}
		}

		public void SetExportFilter(bool value) {
			if(ExportFilter != value) {
				ExportFilter = value;
				Filter();
			}
		}

		public void SetExcludeFilter(bool value) {
			if(ExcludeFilter != value) {
				ExcludeFilter = value;
				Filter();
			}
		}

		public void LoadFile(int index) {
			if(Index != index) {
				Index = index;
				Target.LoadFile(Index >= 0 ? Target.Files[Instances[Index].Index] : null);
			}
		}

		public void Import() {
			Target.Import();
			Filter();
		}

		public int GetIndex() {
			if(Target.GetFile() == null) {
				return -1;
			}
			if(Instances.Length == Target.Files.Length) {
				return Target.GetFile().Index;
			} else {
				return System.Array.FindIndex(Instances, x => x == Target.GetFile());
			}
		}

		public void Inspector() {
			Index = GetIndex();

			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					EditorGUILayout.BeginHorizontal();
					Target.Folder = EditorGUILayout.TextField("Folder", "Assets/" + Target.Folder.Substring(Mathf.Min(7, Target.Folder.Length)));
					if(Utility.GUIButton("Import", UltiDraw.DarkGrey, UltiDraw.White)) {
						Import();
					}
					EditorGUILayout.EndHorizontal();


					Utility.SetGUIColor(Target.GetActor() == null ? UltiDraw.DarkRed : UltiDraw.White);
					Target.Actor = (Actor)EditorGUILayout.ObjectField("Actor", Target.GetActor(), typeof(Actor), true);
					Utility.ResetGUIColor();

					EditorGUILayout.ObjectField("Environment", Target.GetEnvironment(), typeof(Transform), true);

					SetNameFilter(EditorGUILayout.TextField("Name Filter", NameFilter));
					SetExportFilter(EditorGUILayout.Toggle("Export Filter", ExportFilter));
					SetExcludeFilter(EditorGUILayout.Toggle("Exclude Filter", ExcludeFilter));

					if(Instances.Length == 0) {
						LoadFile(-1);
						EditorGUILayout.LabelField("No data available.");
					} else {
						LoadFile(EditorGUILayout.Popup("Data " + "(" + Instances.Length + ")", Index, Names));
						EditorGUILayout.BeginHorizontal();
						LoadFile(EditorGUILayout.IntSlider(Index+1, 1, Instances.Length)-1);
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White)) {
							LoadFile(Mathf.Max(Index-1, 0));
						}
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White)) {
							LoadFile(Mathf.Min(Index+1, Instances.Length-1));
						}
						EditorGUILayout.EndHorizontal();
					}
				}

				if(Target.GetFile() != null) {
					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						Frame frame = Target.GetFile().Data.GetFrame(Target.Timestamp);

						Utility.SetGUIColor(UltiDraw.Mustard);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							GUILayout.FlexibleSpace();
							EditorGUILayout.LabelField(Target.GetFile().Data.Name, GUILayout.Width(100f));
							EditorGUILayout.LabelField("Frames: " + Target.GetFile().Data.GetTotalFrames(), GUILayout.Width(100f));
							EditorGUILayout.LabelField("Time: " + Target.GetFile().Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
							EditorGUILayout.LabelField("Framerate: " + Target.GetFile().Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
							EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
							Target.Timescale = EditorGUILayout.FloatField(Target.Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
							if(Utility.GUIButton("M", Target.ShowMirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.SetMirror(!Target.ShowMirror);
							}
							if(Utility.GUIButton("T", Target.ShowTrajectory ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.ShowTrajectory = !Target.ShowTrajectory;
							}
							if(Utility.GUIButton("V", Target.ShowVelocities ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.ShowVelocities = !Target.ShowVelocities;
							}
							GUILayout.FlexibleSpace();
							EditorGUILayout.EndHorizontal();
						}

						Utility.SetGUIColor(UltiDraw.DarkGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							GUILayout.FlexibleSpace();
							if(Target.Playing) {
								if(Utility.GUIButton("||", Color.red, Color.black, 20f, 20f)) {
									Target.StopAnimation();
								}
							} else {
								if(Utility.GUIButton("|>", Color.green, Color.black, 20f, 20f)) {
									Target.PlayAnimation();
								}
							}
							if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
								Target.LoadPreviousFrame();
							}
							if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
								Target.LoadNextFrame();
							}
							int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.GetFile().Data.GetTotalFrames(), GUILayout.Width(440f));
							if(index != frame.Index) {
								Target.LoadFrame(index);
							}
							EditorGUILayout.LabelField(frame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
							GUILayout.FlexibleSpace();
							EditorGUILayout.EndHorizontal();
							EditorGUILayout.BeginHorizontal();
							GUILayout.FlexibleSpace();
							Target.Window = EditorGUILayout.Slider(Target.Window, 0f, 1f);
							GUILayout.FlexibleSpace();
							EditorGUILayout.EndHorizontal();
						}
					}
					for(int i=0; i<Target.GetFile().Data.Modules.Length; i++) {
						Target.GetFile().Data.Modules[i].Inspector(Target);
					}
					string[] modules = new string[(int)Module.TYPE.Length+1];
					modules[0] = "Add Module...";
					for(int i=1; i<modules.Length; i++) {
						modules[i] = ((Module.TYPE)(i-1)).ToString();
					}
					int module = EditorGUILayout.Popup(0, modules);
					if(module > 0) {
						Target.GetFile().Data.AddModule((Module.TYPE)(module-1));
					}
					if(Utility.GUIButton("Settings", Target.InspectSettings ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.InspectSettings = !Target.InspectSettings;
					}
					if(Target.InspectSettings) {
						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							Target.GetFile().Data.Export = EditorGUILayout.Toggle("Export", Target.GetFile().Data.Export);
							Target.SetScaling(EditorGUILayout.FloatField("Scaling", Target.GetFile().Data.Scaling));
							Target.GetFile().Data.RootSmoothing = EditorGUILayout.IntField("Root Smoothing", Target.GetFile().Data.RootSmoothing);
							Target.GetFile().Data.Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.GetFile().Data.Ground), InternalEditorUtility.layers));
							Target.GetFile().Data.MirrorAxis = (MotionData.AXIS)EditorGUILayout.EnumPopup("Mirror Axis", Target.GetFile().Data.MirrorAxis);
							string[] names = new string[Target.GetFile().Data.Source.Bones.Length];
							for(int i=0; i<Target.GetFile().Data.Source.Bones.Length; i++) {
								names[i] = Target.GetFile().Data.Source.Bones[i].Name;
							}
							for(int i=0; i<Target.GetFile().Data.Source.Bones.Length; i++) {
								EditorGUILayout.BeginHorizontal();
								Target.GetFile().Data.Source.Bones[i].Active = EditorGUILayout.Toggle(Target.GetFile().Data.Source.Bones[i].Active);
								EditorGUI.BeginDisabledGroup(true);
								EditorGUILayout.TextField(names[i]);
								EditorGUI.EndDisabledGroup();
								Target.GetFile().Data.SetSymmetry(i, EditorGUILayout.Popup(Target.GetFile().Data.Symmetry[i], names));
								Target.GetFile().Data.Source.Bones[i].Alignment = EditorGUILayout.Vector3Field("", Target.GetFile().Data.Source.Bones[i].Alignment);
								EditorGUILayout.EndHorizontal();
							}

							Utility.SetGUIColor(UltiDraw.Grey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();

								Utility.SetGUIColor(UltiDraw.Cyan);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									EditorGUILayout.LabelField("Camera");
								}

								if(Utility.GUIButton("Auto Focus", Target.AutoFocus ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
									Target.SetAutoFocus(!Target.AutoFocus);
								}
								Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
								Target.FocusOffset = EditorGUILayout.FloatField("Focus Offset", Target.FocusOffset);
								Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
								Target.FocusAngle = EditorGUILayout.Slider("Focus Angle", Target.FocusAngle, 0f, 360f);
								Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
							}

							if(Utility.GUIButton("Copy Hierarchy", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.CopyHierarchy();
							}
							if(Utility.GUIButton("Detect Symmetry", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.GetFile().Data.DetectSymmetry();
							}
							if(Utility.GUIButton("Create Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.CreateSkeleton();
							}

							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("Add Export Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.GetFile().Data.AddSequence(1, Target.GetFile().Data.GetTotalFrames());
							}
							if(Utility.GUIButton("Remove Export Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.GetFile().Data.RemoveSequence();
							}
							EditorGUILayout.EndHorizontal();
							for(int i=0; i<Target.GetFile().Data.Sequences.Length; i++) {
								Utility.SetGUIColor(UltiDraw.Grey);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									
									EditorGUILayout.BeginHorizontal();
									GUILayout.FlexibleSpace();
									if(Utility.GUIButton("X", Color.cyan, Color.black, 15f, 15f)) {
										Target.GetFile().Data.Sequences[i].SetStart(Target.GetState().Index);
									}
									EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
									Target.GetFile().Data.Sequences[i].SetStart(EditorGUILayout.IntField(Target.GetFile().Data.Sequences[i].Start, GUILayout.Width(100f)));
									EditorGUILayout.LabelField("End", GUILayout.Width(50f));
									Target.GetFile().Data.Sequences[i].SetEnd(EditorGUILayout.IntField(Target.GetFile().Data.Sequences[i].End, GUILayout.Width(100f)));
									if(Utility.GUIButton("X", Color.cyan, Color.black, 15f, 15f)) {
										Target.GetFile().Data.Sequences[i].SetEnd(Target.GetState().Index);
									}
									GUILayout.FlexibleSpace();
									EditorGUILayout.EndHorizontal();
								}
							}
						}
					}
				}
			}
		}

	}

}
#endif


	/*
	private bool AutoFocus = false;
	private float FocusHeight = 1f;
	private float FocusOffset = 0f;
	private float FocusDistance = 2.5f;
	private float FocusAngle = 0f;
	private float FocusSmoothing = 0.05f;
	*/

	/*
	public void SetAutoFocus(bool value) {
		if(AutoFocus != value) {
			AutoFocus = value;
			if(!AutoFocus) {
				Vector3 position =  SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion rotation = Quaternion.Euler(0f, SceneView.lastActiveSceneView.camera.transform.rotation.eulerAngles.y, 0f);
				SceneView.lastActiveSceneView.LookAtDirect(position, rotation, 0f);
			}
		}
	}


		
		if(AutoFocus) {
			if(SceneView.lastActiveSceneView != null) {
				Vector3 lastPosition = SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion lastRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
				Vector3 position = GetState().Root.GetPosition();
				position.y += FocusHeight;
				Quaternion rotation = GetState().Root.GetRotation();
				rotation.x = 0f;
				rotation.z = 0f;
				rotation = Quaternion.Euler(0f, Mirror ? Mathf.Repeat(FocusAngle + 0f, 360f) : FocusAngle, 0f) * rotation;
				position += FocusOffset * (rotation * Vector3.right);
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
			}
		}
		
				
				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Cyan);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField("Camera");
					}

					if(Utility.GUIButton("Auto Focus", Target.AutoFocus ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.SetAutoFocus(!Target.AutoFocus);
					}
					Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
					Target.FocusOffset = EditorGUILayout.FloatField("Focus Offset", Target.FocusOffset);
					Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
					Target.FocusAngle = EditorGUILayout.Slider("Focus Angle", Target.FocusAngle, 0f, 360f);
					Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
				}

									
									for(int s=0; s<Target.GetFile().Data.Styles.Length; s++) {
										EditorGUILayout.BeginHorizontal();
										GUILayout.FlexibleSpace();
										EditorGUILayout.LabelField(Target.GetFile().Data.Styles[s], GUILayout.Width(50f));
										EditorGUILayout.LabelField("Style Copies", GUILayout.Width(100f));
										Target.GetFile().Data.Sequences[i].SetStyleCopies(s, EditorGUILayout.IntField(Target.GetFile().Data.Sequences[i].StyleCopies[s], GUILayout.Width(100f)));
										EditorGUILayout.LabelField("Transition Copies", GUILayout.Width(100f));
										Target.GetFile().Data.Sequences[i].SetTransitionCopies(s, EditorGUILayout.IntField(Target.GetFile().Data.Sequences[i].TransitionCopies[s], GUILayout.Width(100f)));
										GUILayout.FlexibleSpace();
										EditorGUILayout.EndHorizontal();
									}
									
									//for(int c=0; c<Target.GetFile().Data.Sequences[i].Copies.Length; c++) {
									//	EditorGUILayout.LabelField("Copy " + (c+1) + " - " + "Start: " + Target.GetFile().Data.Sequences[i].Copies[c].Start + " End: " + Target.GetFile().Data.Sequences[i].Copies[c].End);
									//}
	*/