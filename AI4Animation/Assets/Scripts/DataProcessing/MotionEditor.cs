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
	public MotionData[] Files = new MotionData[0];
	public Transform[] Environments = new Transform[0];

	//public bool ShowMotion = false;
	public bool ShowVelocities = false;
	public bool ShowTrajectory = false;

	/*
	private bool AutoFocus = false;
	private float FocusHeight = 1f;
	private float FocusOffset = 0f;
	private float FocusDistance = 2.5f;
	private float FocusAngle = 0f;
	private float FocusSmoothing = 0.05f;
	*/

	private bool Mirror = false;
	private bool Playing = false;
	private float Timescale = 1f;
	private float Timestamp = 0f;

	private Actor Actor = null;
	private Transform Environment = null;
	private MotionState State = null;

	private int FileID = -1;

	//public void VisualiseMotion(bool value) {
	//	ShowMotion = value;
	//}
	public void VisualiseVelocities(bool value) {
		ShowVelocities = value;
	}
	public void VisualiseTrajectory(bool value) {
		ShowTrajectory = value;
	}

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
	*/

	public void SetMirror(bool value) {
		if(Mirror != value) {
			Mirror = value;
			LoadFrame(Timestamp);
		}
	}

	public bool IsMirror() {
		return Mirror;
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

	public MotionData[] GetFiles() {
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

	public Transform[] GetEnvironments() {
		return Environments;
	}

	public int GetFileID() {
		return FileID;
	}

	public MotionData GetData() {
		if(Files.Length == 0) {
			FileID = -1;
			return null;
		}
		LoadFile(Mathf.Clamp(FileID, 0, Files.Length-1));
		return Files[FileID];
	}

	public MotionState GetState() {
		if(State == null) {
			LoadFrame(Timestamp);
		}
		return State;
	}

	public void Load() {
		string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{Folder});
		//Files
		List<MotionData> files = new List<MotionData>();
		for(int i=0; i<assets.Length; i++) {
			MotionData file = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[i]), typeof(MotionData));
			if(!files.Find(x => x.Name == file.Name)) {
				files.Add(file);
			}
		}
		Files = files.ToArray();
		//Cleanup
		for(int i=0; i<GetEnvironment().childCount; i++) {
			if(!System.Array.Find(Files, x => x.Name == GetEnvironment().GetChild(i).name)) {
				Utility.Destroy(GetEnvironment().GetChild(i).gameObject);
				i--;
			}
		}
		//Fill
		Environments = new Transform[Files.Length];
		for(int i=0; i<Environments.Length; i++) {
			Environments[i] = GetEnvironment().Find(Files[i].Name);
			if(Environments[i] == null) {
				Environments[i] = new GameObject(Files[i].Name).transform;
				Environments[i].SetParent(GetEnvironment());
			}
		}
		//Finalise
		for(int i=0; i<Environments.Length; i++) {
			Environments[i].gameObject.SetActive(i == FileID);
			Environments[i].SetSiblingIndex(i);
		}
		//Initialise
		if(GetData() != null) {
			LoadFrame(0f);
		}
	}

	public void Cleanup() {
		for(int i=0; i<Files.Length; i++) {
			if(Files[i] == null) {
				ArrayExtensions.RemoveAt(ref Files, i);
				Utility.Destroy(Environments[i].gameObject);
				ArrayExtensions.RemoveAt(ref Environments, i);
				i--;
			}
		}
	}

	public void SaveAll() {
		for(int i=0; i<Files.Length; i++) {
			EditorUtility.SetDirty(Files[i]);
		}
		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();
	}
	
	public void Save(int id) {
		if(id >= 0 && id < Files.Length) {
			EditorUtility.SetDirty(Files[id]);
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
		}
	}

	public void LoadFile(int id) {
		if(FileID != id) {
			Save(FileID);
			FileID = id;
			if(FileID < 0) {
				return;
			}
			for(int i=0; i<Environments.Length; i++) {
				Environments[i].gameObject.SetActive(i == FileID);
			}
			LoadFrame(0f);
		}
	}

	public void LoadPreviousFile() {
		LoadFile(Mathf.Max(FileID-1, 0));
	}

	public void LoadNextFile() {
		LoadFile(Mathf.Min(FileID+1, Files.Length-1));
	}

	public void LoadFrame(MotionState state) {
		Timestamp = state.Timestamp;
		State = state;
		if(state.Mirrored) {
			GetEnvironment().localScale = Vector3.one.GetMirror(GetData().GetAxis(GetData().MirrorAxis));
		} else {
			GetEnvironment().localScale = Vector3.one;
		}

		GetActor().GetRoot().position = GetState().Root.GetPosition();
		GetActor().GetRoot().rotation = GetState().Root.GetRotation();
		for(int i=0; i<GetActor().Bones.Length; i++) {
			GetActor().Bones[i].Transform.position = GetState().BoneTransformations[i].GetPosition();
			GetActor().Bones[i].Transform.rotation = GetState().BoneTransformations[i].GetRotation();
		}

		/*
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
		*/
	}

	public void LoadFrame(float timestamp) {
		LoadFrame(new MotionState(GetData().GetFrame(timestamp), Mirror));
	}

	public void LoadFrame(int index) {
		LoadFrame(GetData().GetFrame(index).Timestamp);
	}

	public void LoadPreviousFrame() {
		LoadFrame(Mathf.Max(GetData().GetFrame(Timestamp).Index - 1, 1));
	}

	public void LoadNextFrame() {
		LoadFrame(Mathf.Min(GetData().GetFrame(Timestamp).Index + 1, GetData().GetTotalFrames()));
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
		while(GetData() != null) {
			Timestamp += Timescale * (float)Utility.GetElapsedTime(timestamp);
			if(Timestamp > GetData().GetTotalTime()) {
				Timestamp = Mathf.Repeat(Timestamp, GetData().GetTotalTime());
			}
			timestamp = Utility.GetTimestamp();
			LoadFrame(Timestamp);
			yield return new WaitForSeconds(0f);
		}

		/*
		while(Data != null) {
			int next = Data.GetFrame(Timestamp).Index+1;
			if(next > Data.GetTotalFrames()) {
				next = 1;
			}
			LoadFrame(next);
			yield return new WaitForSeconds(0f);
		}
		*/
	}

	public Actor CreateSkeleton() {
		if(GetData() == null) {
			return null;
		}
		Actor actor = new GameObject("Skeleton").AddComponent<Actor>();
		actor.transform.SetParent(transform);
		string[] names = new string[GetData().Source.Bones.Length];
		string[] parents = new string[GetData().Source.Bones.Length];
		for(int i=0; i<GetData().Source.Bones.Length; i++) {
			names[i] = GetData().Source.Bones[i].Name;
			parents[i] = GetData().Source.Bones[i].Parent;
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

	public void Draw() {
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
			for(int i=0; i<GetActor().Bones.Length; i++) {
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
	}

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		private string[] Names = new string[0];

		private float RefreshRate = 30f;
		private System.DateTime Timestamp;

		void Awake() {
			Target = (MotionEditor)target;
			Target.Cleanup();
			GenerateNames();
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			if(!Application.isPlaying && Target != null) {
				Target.Save(Target.FileID);
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
		
		public void GenerateNames() {
			Names = new string[Target.Files.Length];
			for(int i=0; i<Target.Files.Length; i++) {
				Names[i] = Target.Files[i].Name;
			}
		}

		public void Inspector() {
			
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					
					EditorGUILayout.BeginHorizontal();
					Target.Folder = EditorGUILayout.TextField("Folder", "Assets/" + Target.Folder.Substring(Mathf.Min(7, Target.Folder.Length)));
					if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.Load();
						GenerateNames();
					}
					EditorGUILayout.EndHorizontal();

					Utility.SetGUIColor(Target.GetActor() == null ? UltiDraw.DarkRed : UltiDraw.White);
					Target.Actor = (Actor)EditorGUILayout.ObjectField("Actor", Target.GetActor(), typeof(Actor), true);
					Utility.ResetGUIColor();

					EditorGUILayout.ObjectField("Environment", Target.GetEnvironment(), typeof(Transform), true);

					EditorGUILayout.BeginHorizontal();
					if(Target.Files.Length == 0) {
						Target.LoadFile(-1);
						EditorGUILayout.LabelField("No data available.");
					} else {
						Target.LoadFile(EditorGUILayout.Popup("Data " + "(" + Target.Files.Length + ")", Target.FileID, Names));
						EditorGUILayout.EndHorizontal();
						Target.LoadFile(EditorGUILayout.IntSlider(Target.FileID+1, 1, Target.Files.Length)-1);
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White)) {
							Target.LoadPreviousFile();
						}
						if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White)) {
							Target.LoadNextFile();
						}
					}
					EditorGUILayout.EndHorizontal();
				}

				/*
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
				*/

				if(Target.GetData() != null) {
					//Target.GetData().Inspector(Target);
					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						Frame frame = Target.GetData().GetFrame(Target.Timestamp);

						Utility.SetGUIColor(UltiDraw.Mustard);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							GUILayout.FlexibleSpace();
							EditorGUILayout.LabelField("Frames: " + Target.GetData().GetTotalFrames(), GUILayout.Width(100f));
							EditorGUILayout.LabelField("Time: " + Target.GetData().GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
							EditorGUILayout.LabelField("Framerate: " + Target.GetData().Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
							EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
							Target.Timescale = EditorGUILayout.FloatField(Target.Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
							GUILayout.FlexibleSpace();
							EditorGUILayout.EndHorizontal();
						}

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
						int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.GetData().GetTotalFrames(), GUILayout.Width(440f));
						if(index != frame.Index) {
							Target.LoadFrame(index);
						}
						EditorGUILayout.LabelField(frame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
						GUILayout.FlexibleSpace();
						EditorGUILayout.EndHorizontal();

						EditorGUILayout.BeginHorizontal();
						//if(Utility.GUIButton("Motion", Target.ShowMotion ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						//	Target.ShowMotion = !Target.ShowMotion;
						//}
						if(Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.SetMirror(!Target.Mirror);
						}
						if(Utility.GUIButton("Trajectory", Target.ShowTrajectory ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowTrajectory = !Target.ShowTrajectory;
						}
						if(Utility.GUIButton("Velocities", Target.ShowVelocities ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowVelocities = !Target.ShowVelocities;
						}
						EditorGUILayout.EndHorizontal();

						Target.GetData().Export = EditorGUILayout.Toggle("Export", Target.GetData().Export);
						Target.GetData().Scaling = EditorGUILayout.FloatField("Scaling", Target.GetData().Scaling);
						Target.GetData().RootSmoothing = EditorGUILayout.IntField("Root Smoothing", Target.GetData().RootSmoothing);
						Target.GetData().Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.GetData().Ground), InternalEditorUtility.layers));
						Target.GetData().MirrorAxis = (MotionData.AXIS)EditorGUILayout.EnumPopup("Mirror Axis", Target.GetData().MirrorAxis);
						string[] names = new string[Target.GetData().Source.Bones.Length];
						for(int i=0; i<Target.GetData().Source.Bones.Length; i++) {
							names[i] = Target.GetData().Source.Bones[i].Name;
						}
						for(int i=0; i<Target.GetData().Source.Bones.Length; i++) {
							EditorGUILayout.BeginHorizontal();
							EditorGUI.BeginDisabledGroup(true);
							EditorGUILayout.TextField(names[i]);
							EditorGUI.EndDisabledGroup();
							Target.GetData().SetSymmetry(i, EditorGUILayout.Popup(Target.GetData().Symmetry[i], names));
							EditorGUILayout.EndHorizontal();
						}
						if(Utility.GUIButton("Detect Symmetry", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.GetData().DetectSymmetry();
						}
						if(Utility.GUIButton("Create Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.CreateSkeleton();
						}

						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Add Export Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.GetData().AddSequence(1, Target.GetData().GetTotalFrames());
						}
						if(Utility.GUIButton("Remove Export Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.GetData().RemoveSequence();
						}
						EditorGUILayout.EndHorizontal();
						for(int i=0; i<Target.GetData().Sequences.Length; i++) {
							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								
								EditorGUILayout.BeginHorizontal();
								GUILayout.FlexibleSpace();
								if(Utility.GUIButton("X", Color.cyan, Color.black, 15f, 15f)) {
									Target.GetData().Sequences[i].SetStart(Target.GetState().Index);
								}
								EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
								Target.GetData().Sequences[i].SetStart(EditorGUILayout.IntField(Target.GetData().Sequences[i].Start, GUILayout.Width(100f)));
								EditorGUILayout.LabelField("End", GUILayout.Width(50f));
								Target.GetData().Sequences[i].SetEnd(EditorGUILayout.IntField(Target.GetData().Sequences[i].End, GUILayout.Width(100f)));
								if(Utility.GUIButton("X", Color.cyan, Color.black, 15f, 15f)) {
									Target.GetData().Sequences[i].SetEnd(Target.GetState().Index);
								}
								GUILayout.FlexibleSpace();
								EditorGUILayout.EndHorizontal();

								/*
								for(int s=0; s<Target.GetData().Styles.Length; s++) {
									EditorGUILayout.BeginHorizontal();
									GUILayout.FlexibleSpace();
									EditorGUILayout.LabelField(Target.GetData().Styles[s], GUILayout.Width(50f));
									EditorGUILayout.LabelField("Style Copies", GUILayout.Width(100f));
									Target.GetData().Sequences[i].SetStyleCopies(s, EditorGUILayout.IntField(Target.GetData().Sequences[i].StyleCopies[s], GUILayout.Width(100f)));
									EditorGUILayout.LabelField("Transition Copies", GUILayout.Width(100f));
									Target.GetData().Sequences[i].SetTransitionCopies(s, EditorGUILayout.IntField(Target.GetData().Sequences[i].TransitionCopies[s], GUILayout.Width(100f)));
									GUILayout.FlexibleSpace();
									EditorGUILayout.EndHorizontal();
								}
								*/
								//for(int c=0; c<Target.GetData().Sequences[i].Copies.Length; c++) {
								//	EditorGUILayout.LabelField("Copy " + (c+1) + " - " + "Start: " + Target.GetData().Sequences[i].Copies[c].Start + " End: " + Target.GetData().Sequences[i].Copies[c].End);
								//}
							}
						}
					}
					for(int i=0; i<Target.GetData().Modules.Length; i++) {
						Target.GetData().Modules[i].Inspector(Target);
					}
					string[] modules = new string[(int)DataModule.TYPE.Length+1];
					modules[0] = "Add Module...";
					for(int i=1; i<modules.Length; i++) {
						modules[i] = ((DataModule.TYPE)(i-1)).ToString();
					}
					int module = EditorGUILayout.Popup(0, modules);
					if(module > 0) {
						Target.GetData().AddModule((DataModule.TYPE)(module-1));
					}
				}
			}
		}

	}

}
#endif
