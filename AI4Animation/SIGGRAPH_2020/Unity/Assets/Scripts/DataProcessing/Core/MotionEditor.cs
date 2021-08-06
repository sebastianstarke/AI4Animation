#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using System.Collections;
using System.Collections.Generic;

[ExecuteInEditMode]
public class MotionEditor : MonoBehaviour {

	public static MotionEditor Instance = null;

	public string[] Folders = new string[0];
	public bool[] Imports = new bool[0];
	public Actor Model = null;
	public string[] Assets = new string[0];

	public bool LoadExportable = true;
	public bool LoadUnexportable = true;
	public bool LoadTagged = false;

	public float TargetFramerate = 60f;
	public int RandomSeed = 0;

	public bool Visualize = true;
	public bool Mirror = false;
	public bool Precomputable = true;

	public bool Advanced = false;
	public bool InspectHierarchy = false;

	public int PastKeys = 6;
	public int FutureKeys = 6;
	public float PastWindow = 1f;
	public float FutureWindow = 1f;

	private bool MotionTrail = false;
	private bool CharacterMeshTrail = true;
	private bool SkeletonSketchTrail = false;

	private bool VisualizeFirstFrame = false;
	private bool VisualizeLastFrame = false;

	private bool CameraFocus = false;
	private float FocusHeight = 1f;
	private float FocusDistance = 2f;
	private Vector2 FocusAngle = new Vector2(0f, 90f);
	private float FocusSmoothing = 0.05f;

	private bool Playing = false;
	private float Timescale = 1f;
	private float Timestamp = 0f;
	private float Zoom = 1f;

	private bool Importing = false;
	private float Progress = 0f;
	private int CurrentFile = 0;
	private int TotalFiles = 0;

	private int AssetIndex = -1;

	private MotionData Data = null;
	private Actor Actor = null;

	//Precomputed
	private TimeSeries TimeSeries = null;
	private int[] BoneMapping = null;

	[UnityEditor.Callbacks.DidReloadScripts]
	public static void OnScriptsReloaded() {
		if(GetInstance() == null) {
			return;
		}
		GetInstance().LoadFrame(GetInstance().Timestamp);
	}

	void OnDestroy() {
		UnloadData();
	}

	public static MotionEditor GetInstance() {
		if(Instance == null) {
			Instance = GameObject.FindObjectOfType<MotionEditor>();
		}
		return Instance;
	}

	public static int FindIndex(string[] guids, string guid) {
		return System.Array.FindIndex(guids, x => x == guid);
	}

	public void Refresh() {
		if(Folders.Length == 0) {
			Folders = new string[1];
		}
		if(Imports.Length != Folders.Length) {
			Imports = new bool[Folders.Length];
			for(int i=0; i<Imports.Length; i++) {
				Imports[i] = true;
			}
		}
		if(Data == null && Assets.Length > 0) {
			LoadData(Assets.First());
		}
	}

	public float GetTimestamp() {
		return Timestamp;
	}

	public IEnumerator Import(System.Action callback) {
		Importing = true;

		string[] folders = GetFolders();
		if(folders.Length == 0) {
			Assets = new string[0];
		} else {
			string[] candidates = AssetDatabase.FindAssets("t:MotionData", folders);
			TotalFiles = candidates.Length;
			if(!LoadExportable || !LoadUnexportable) {
				List<string> assets = new List<string>();
				for(int i=0; i<candidates.Length; i++) {
					if(!Importing) {
						break;
					}
					CurrentFile += 1;
					MotionData asset = (MotionData)AssetDatabase.LoadMainAssetAtPath(Utility.GetAssetPath(candidates[i]));
					if((LoadExportable && asset.Export) || (LoadUnexportable && !asset.Export)) {
						if(!LoadTagged || LoadTagged && asset.Tagged) {
							assets.Add(candidates[i]);
						}
					}
					if(i % 10 == 0) {
						Resources.UnloadUnusedAssets();
					}
					Progress = (float)(i+1) / candidates.Length;
					yield return new WaitForSeconds(0f);
				}
				if(Importing) {
					Assets = assets.ToArray();
				}
			} else {
				Assets = candidates;
			}
		}
		if(Importing) {
			LoadData(null);
			callback();
		}

		CurrentFile = 0;
		TotalFiles = 0;
		Progress = 0f;
		Importing = false;
	}

	public string GetLoadStrategy() {
		if(LoadExportable && LoadUnexportable) {
			return "All";
		}
		if(LoadExportable && !LoadUnexportable) {
			return "Exportable";
		}
		if(!LoadExportable && LoadUnexportable) {
			return "Unexportable";
		}
		return "Undefined";
	}

	public void ToggleLoadStrategy() {
		if(LoadExportable && LoadUnexportable) {
			LoadExportable = true;
			LoadUnexportable = false;
			return;
		}
		if(LoadExportable && !LoadUnexportable) {
			LoadExportable = false;
			LoadUnexportable = true;
			return;
		}
		if(!LoadExportable && LoadUnexportable) {
			LoadExportable = true;
			LoadUnexportable = true;
			return;
		}
	}

	public bool IsFolderValid(int index) {
		return AssetDatabase.IsValidFolder(GetFolder(index));
	}

	public string[] GetFolders() {
		List<string> folders = new List<string>();
		for(int i=0; i<Folders.Length; i++) {
			if(IsFolderValid(i) && Imports[i]) {
				folders.Add(GetFolder(i));
			}
		}
		return folders.ToArray();
	}

	public string GetFolder(int index) {
		return "Assets/" + Folders[index];
	}

	public void ResetPrecomputation() {
		TimeSeries = null;
		BoneMapping = null;
		if(Data != null) {
			Data.ResetPrecomputation();
		}
	}

	public TimeSeries GetTimeSeries() {
		if(TimeSeries == null) {
			TimeSeries = new TimeSeries(PastKeys, FutureKeys, PastWindow, FutureWindow, 1);
		}
		return TimeSeries;
	}

	public int[] GetBoneMapping() {
		if(Data == null) {
			return null;
		}
		if(BoneMapping == null || BoneMapping.Length != GetActor().Bones.Length) {
			BoneMapping = Data.Source.GetBoneIndices(GetActor().GetBoneNames());
		}
		return BoneMapping;
	}

	public void SetPastKeys(int value) {
		if(PastKeys != value) {
			PastKeys = value;
			ResetPrecomputation();
		}
	}

	public void SetFutureKeys(int value) {
		if(FutureKeys != value) {
			FutureKeys = value;
			ResetPrecomputation();
		}
	}

	public void SetPastWindow(float value) {
		if(PastWindow != value) {
			PastWindow = value;
			ResetPrecomputation();
		}
	}

	public void SetFutureWindow(float value) {
		if(FutureWindow != value) {
			FutureWindow = value;
			ResetPrecomputation();
		}
	}

	public void SetVisualize(bool value) {
		Visualize = value;
	}

	public void SetMirror(bool value) {
		if(Mirror != value) {
			Mirror = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetPrecomputable(bool value) {
		if(Precomputable != value) {
			Precomputable = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetTargetFramerate(float value) {
		TargetFramerate = Data == null ? TargetFramerate : Mathf.Clamp(value, 1, Data.Framerate);
	}

	public void SetRandomSeed(int value) {
		RandomSeed = Mathf.Max(value, 0);
	}

	public int GetCurrentSeed() {
		if(RandomSeed == 0) {
			Frame frame = GetCurrentFrame();
			return frame == null ? 0 : frame.Index;
		} else {
			return RandomSeed;
		}
	}

	public void SetCameraFocus(bool value) {
		if(CameraFocus != value) {
			CameraFocus = value;
			LoadFrame(Timestamp);
		}
	}

	public void ApplyCameraFocus() {
		if(CameraFocus) {
			if(SceneView.lastActiveSceneView != null) {
				Vector3 currentPosition = SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion currentRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
				Vector3 target = (GetActor().Bones.Length == 0 ? GetActor().GetRoot().position : GetActor().Bones.First().Transform.position).SetY(FocusHeight);

				Vector3 position = (target + Quaternion.Euler(-FocusAngle.x, -FocusAngle.y, 0f) * new Vector3(0f, 0f, FocusDistance));
				Quaternion rotation = Quaternion.LookRotation((target - position).normalized, Vector3.up);
				
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(currentPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(currentRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
			}
		}
	}

	public void SetMotionTrail(bool value) {
		if(MotionTrail != value) {
			MotionTrail = value;
			LoadFrame(Timestamp);
		}
	}

	public void ApplyMotionTrail() {
		Transform trail = transform.Find("MotionTrail");
		if(MotionTrail) {
			if(trail == null) {
				trail = new GameObject("MotionTrail").transform;
			}
			trail.SetParent(transform);
			for(int i=0; i<GetTimeSeries().Samples.Length; i++) {
				Transform instance = trail.Find(i.ToString());
				if(instance == null) {
					instance = Instantiate(GetActor().gameObject).transform;
					instance.name = i.ToString();
					// Transparency transparency = instance.gameObject.AddComponent<Transparency>();
					// transparency.SetTransparency(i == series.Pivot ? 0f : 1f - (float)Mathf.Abs(i-series.Pivot) / (float)series.Pivot);
					instance.SetParent(trail);
				}
				Actor actor = instance.GetComponent<Actor>();
				actor.BoneColor = GetActor().BoneColor.Darken((float)Mathf.Abs(i- GetTimeSeries().Pivot) / (float) GetTimeSeries().Pivot);
				actor.DrawSkeleton = !SkeletonSketchTrail;
				actor.DrawSketch = SkeletonSketchTrail;
				foreach(Light l in instance.GetComponentsInChildren<Light>()) {
					l.enabled = false;
				}
				foreach(SkinnedMeshRenderer r in instance.GetComponentsInChildren<SkinnedMeshRenderer>()) {
					r.enabled = CharacterMeshTrail;
				}
				foreach(Renderer r in instance.GetComponentsInChildren<Renderer>()) {
					r.enabled = CharacterMeshTrail;
				}
				for(int b=0; b<actor.Bones.Length; b++) {
					Matrix4x4 bone = Data.GetFrame(GetCurrentFrame().Timestamp +  GetTimeSeries().Samples[i].Timestamp).GetBoneTransformation(actor.Bones[b].GetName(), Mirror);
					actor.Bones[b].Transform.position = bone.GetPosition();
					actor.Bones[b].Transform.rotation = bone.GetRotation();
				}
			}
		} else {
			if(trail != null) {
				Utility.Destroy(trail.gameObject);
			}
		}
	}

	public float GetWindow() {
		return Data == null ? 0f : Zoom * Data.GetTotalTime();
	}

	public Vector3Int GetView() {
		float startTime = GetCurrentFrame().Timestamp-GetWindow()/2f;
		float endTime = GetCurrentFrame().Timestamp+GetWindow()/2f;
		if(startTime < 0f) {
			endTime -= startTime;
			startTime = 0f;
		}
		if(endTime > Data.GetTotalTime()) {
			startTime -= endTime-Data.GetTotalTime();
			endTime = Data.GetTotalTime();
		}
		int start = Data.GetFrame(Mathf.Max(0f, startTime)).Index;
		int end = Data.GetFrame(Mathf.Min(Data.GetTotalTime(), endTime)).Index;
		int elements = end-start+1;
		return new Vector3Int(start, end, elements);
	}
	
	public void SetModel(Actor model) {
		if(Model == null && model != null) {
			if(Actor != null) {
				Utility.Destroy(Actor.gameObject);
				Actor = null;
			}
			Model = model;
			LoadFrame(Timestamp);
		} else {
			Model = model;
		}
	}

	public MotionData GetAsset() {
		return Data;
	}

	public MotionData GetAsset(int index) {
		return GetAsset(Assets[index]);
	}

	public MotionData GetAsset(string guid) {
		if(guid == null || guid == string.Empty) {
			return null;
		} else {
			MotionData asset = (MotionData)AssetDatabase.LoadAssetAtPath(Utility.GetAssetPath(guid), typeof(MotionData));
			asset.SetPrecomputable(Precomputable);
			return asset;
		}
	}

	public int GetAssetIndex() {
		return AssetIndex;
	}

	public Actor GetActor() {
		if(Model != null) {
			return Model;
		}
		if(Actor == null) {
			Actor = Data.CreateActor();
			Actor.transform.SetParent(transform);
		}
		return Actor;
	}

	public float RoundToTargetTime(float time) {
		return Mathf.RoundToInt(time * TargetFramerate) / TargetFramerate;	
	}

	public float CeilToTargetTime(float time) {
		return Mathf.CeilToInt(time * TargetFramerate) / TargetFramerate;	
	}

	public float FloorToTargetTime(float time) {
		return Mathf.FloorToInt(time * TargetFramerate) / TargetFramerate;	
	}


	public Frame GetCurrentFrame() {
		return Data == null ? null : Data.GetFrame(Timestamp);
	}

	public MotionData LoadData(string guid) {
		if(guid != null && Data != null && Data.GetName() == Utility.GetAssetName(guid)) {
			return Data;
		}
		MotionData data = GetAsset(guid);
		if(Data != data) {
			//Unload Previous
			UnloadData();

			//Character
			if(Model == null && Actor != null) {
				Utility.Destroy(Actor.gameObject);
			}

			//Load Next
			Data = data;
			if(Data != null) {
				Data.Load(this);
				LoadFrame(0f);
			}

			//Assign Index
			AssetIndex = FindIndex(Assets, Utility.GetAssetGUID(Data));
		}
		return Data;
	}

	public void UnloadData() {
		if(Data == null) {
			return;
		}

		//Saving
		if(Data.IsDirty()) {
			if(Application.isPlaying) {
				Debug.Log("Can not save asset " + name + " in play mode.");
			} else {
				//Debug.Log("Saving asset " + Data.GetName() + ".");
				AssetDatabase.SaveAssets();
				EditorSceneManager.SaveScene(Data.GetScene());
			}
		}

		//Unloading Scene
		Scene scene = EditorSceneManager.GetSceneByName(Data.name);
		if(Application.isPlaying) {
			SceneManager.UnloadSceneAsync(scene);
		} else {
			EditorCoroutines.StartCoroutine(RemoveScene(0.25f), this);
		}
		IEnumerator RemoveScene(float delay) {
			EditorSceneManager.CloseScene(scene, false);
			yield return new WaitForSeconds(delay);
			if(!scene.isLoaded) {
				EditorSceneManager.CloseScene(scene, true);
				EditorApplication.RepaintHierarchyWindow();
			}
		}

		//Unloading Asset
		Resources.UnloadUnusedAssets();

		//Reset Temporary
		TimeSeries = null;
		BoneMapping = null;
	}

	public void LoadFrame(float timestamp) {
		if(Data == null) {
			return;
		}
		
		Physics.autoSyncTransforms = true;
		Timestamp = timestamp;

		//Setup Precomputation
		Data.SetPrecomputable(Precomputable);

		//Apply posture on character.
		for(int i=0; i<GetActor().Bones.Length; i++) {
			if(GetBoneMapping()[i] == -1) {
				Debug.Log("Bone " + GetActor().Bones[i].GetName() + " could not be mapped.");
			} else {
				Matrix4x4 transformation = GetCurrentFrame().GetBoneTransformation(GetBoneMapping()[i], Mirror);
				Vector3 velocity = GetCurrentFrame().GetBoneVelocity(GetBoneMapping()[i], Mirror);
				GetActor().Bones[i].Transform.position = transformation.GetPosition();
				GetActor().Bones[i].Transform.rotation = transformation.GetRotation();
				GetActor().Bones[i].Velocity = velocity;
			}
		}

		//Apply scene changes.
		foreach(GameObject instance in Data.GetScene().GetRootGameObjects()) {
			instance.transform.localScale = Vector3.one.GetMirror(Mirror ? Data.MirrorAxis : Axis.None);
			foreach(SceneEvent e in instance.GetComponentsInChildren<SceneEvent>(true)) {
				e.Callback(this);
			}
		}

		//Send callbacks to all modules.
		Data.Callback(this);
		
		//Optional
		ApplyCameraFocus();
		ApplyMotionTrail();
	}

	public void LoadFrame(int index) {
		LoadFrame(Data.GetFrame(index).Timestamp);
	}

	public void LoadFrame(Frame frame) {
		LoadFrame(frame.Index);
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
		System.DateTime previous = Utility.GetTimestamp();
		while(Data != null) {
			float delta = Timescale * (float)Utility.GetElapsedTime(previous);
			if(delta > 1f/TargetFramerate) {
				previous = Utility.GetTimestamp();
				LoadFrame(Mathf.Clamp(Mathf.Repeat(Timestamp + delta, Data.GetLastValidFrame().Timestamp), Data.GetFirstValidFrame().Timestamp, Data.GetLastValidFrame().Timestamp));
			}
			yield return new WaitForSeconds(0f);
		}
	}

	void OnGUI() {
		if(Data != null && Visualize) {
			Data.GUI(this);
		}
	}

	void OnRenderObject() {
		if(Data != null && Visualize) {
			Data.Draw(this);
			if(VisualizeFirstFrame) {
				GetActor().Draw(Data.GetFirstValidFrame().GetBoneTransformations(BoneMapping, Mirror), Color.black);
			}
			if(VisualizeLastFrame) {
				GetActor().Draw(Data.GetLastValidFrame().GetBoneTransformations(BoneMapping, Mirror), Color.black);
			}
		}
	}

	public void DrawRect(Frame start, Frame end, float thickness, Color color, Rect rect) {
		Vector3 view = GetView();
		float _start = (float)(Mathf.Clamp(start.Index, view.x, view.y)-view.x) / (view.z-1);
		float _end = (float)(Mathf.Clamp(end.Index, view.x, view.y)-view.x) / (view.z-1);
		float left = rect.x + _start * rect.width;
		float right = rect.x + _end * rect.width;
		Vector3 a = new Vector3(left, rect.y, 0f);
		Vector3 b = new Vector3(right, rect.y, 0f);
		Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
		Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
		UltiDraw.Begin();
		UltiDraw.DrawTriangle(a, c, b, color);
		UltiDraw.DrawTriangle(b, c, d, color);
		UltiDraw.End();
	}

	public void DrawPivot(Rect rect, Frame frame) {
		DrawRect(
			Data.GetFrame(Mathf.Clamp(frame.Timestamp - PastWindow, 0f, Data.GetTotalTime())),
			Data.GetFrame(Mathf.Clamp(frame.Timestamp + FutureWindow, 0f, Data.GetTotalTime())),
			1f,
			UltiDraw.White.Opacity(0.1f),
			rect
		);
		Vector3 view = GetView();
		Vector3 top = new Vector3(rect.xMin + (float)(frame.Index-view.x)/(view.z-1) * rect.width, rect.yMax - rect.height, 0f);
		Vector3 bottom = new Vector3(rect.xMin + (float)(frame.Index-view.x)/(view.z-1) * rect.width, rect.yMax, 0f);
		UltiDraw.Begin();
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
		UltiDraw.End();
	}

	public void DrawPivot(Rect rect) {
		DrawPivot(rect, GetCurrentFrame());
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		private float RepaintRate = 10f;
		private System.DateTime Timestamp;

		public string[] Assets = new string[0];
		public string[] Enums = new string[0];

		public string Filter = string.Empty;

		void Awake() {
			Target = (MotionEditor)target;
			Target.Refresh();
			ApplyFilter();
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			EditorApplication.update -= EditorUpdate;
		}

		public void EditorUpdate() {
			if(Utility.GetElapsedTime(Timestamp) >= 1f/RepaintRate) {
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

			if(Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.A) {
				LoadPreviousAsset();
			}
			if(Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.D) {
				LoadNextAsset();
			}
		}

		public void LoadPreviousAsset() {
			int pivot = MotionEditor.FindIndex(Assets, Utility.GetAssetGUID(Target.Data));
			if(pivot > 0) {
				Target.LoadData(Assets[Mathf.Max(pivot-1, 0)]);
				
				// if(Target.GetAsset().GetTotalTime() < 5f) {
				// 	LoadPreviousAsset();
				// }
			}
		}

		public void LoadNextAsset() {
			int pivot = MotionEditor.FindIndex(Assets, Utility.GetAssetGUID(Target.Data));
			if(pivot < Assets.Length-1) {
				Target.LoadData(Assets[Mathf.Min(pivot+1, Assets.Length-1)]);

				// if(Target.GetAsset().GetTotalTime() < 5f) {
				// 	LoadNextAsset();
				// }
			}
		}

		public void ApplyFilter() {
			List<string> assets = new List<string>();
			List<string> enums = new List<string>();
			for(int i=0; i<Target.Assets.Length; i++) {
				if(Filter == string.Empty) {
					Add(i);
				} else {
					bool value = Utility.GetAssetName(Target.Assets[i]).ToLowerInvariant().Contains(Filter.ToLowerInvariant());
					if(value) {
						Add(i);
					}
				}
			}
			Assets = assets.ToArray();
			Enums = enums.ToArray();
			void Add(int index) {
				assets.Add(Target.Assets[index]);
				enums.Add("[" + (index+1) + "]" + " " + Utility.GetAssetName(Target.Assets[index]));
			}
		}

		public void Inspector() {
			Target.Refresh();

			//IMPORT SECTION
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Yellow);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Import Settings");
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					
					EditorGUILayout.BeginHorizontal();
					GUILayout.FlexibleSpace();
					if(Utility.GUIButton("Add Folder", UltiDraw.DarkGrey, UltiDraw.White, 100f, 18f)) {
						ArrayExtensions.Append(ref Target.Folders, string.Empty);
						ArrayExtensions.Append(ref Target.Imports, true);
					}
					if(Utility.GUIButton("Remove Folder", UltiDraw.DarkGrey, UltiDraw.White, 100f, 18f)) {
						ArrayExtensions.Shrink(ref Target.Folders);
						ArrayExtensions.Shrink(ref Target.Imports);
					}
					if(Utility.GUIButton(Target.GetLoadStrategy(), UltiDraw.LightGrey, UltiDraw.Black, 100f, 18f)) {
						Target.ToggleLoadStrategy();
					}
					if(Utility.GUIButton(Target.LoadTagged ? "Tagged" : "Any", UltiDraw.LightGrey, UltiDraw.Black, 100f, 18f)) {
						Target.LoadTagged = !Target.LoadTagged;
					}
					GUILayout.FlexibleSpace();
					EditorGUILayout.EndHorizontal();

					Utility.SetGUIColor(UltiDraw.DarkGrey);
					using(new EditorGUILayout.VerticalScope("Box")) {
						Utility.ResetGUIColor();
						for(int i=0; i<Target.Folders.Length; i++) {
							EditorGUILayout.BeginHorizontal();
							Utility.SetGUIColor(Target.IsFolderValid(i) ? (Target.Imports[i] ? UltiDraw.DarkGreen : UltiDraw.Gold) : UltiDraw.DarkRed);
							Target.Folders[i] = EditorGUILayout.TextField(Target.Folders[i]);
							Target.Imports[i] = EditorGUILayout.Toggle(Target.Imports[i], GUILayout.Width(20f));
							Utility.ResetGUIColor();
							EditorGUILayout.EndHorizontal();
						}
					}

					if(!Target.Importing) {
						if(Utility.GUIButton("Import", UltiDraw.DarkGrey, UltiDraw.White)) {
							EditorCoroutines.StartCoroutine(Target.Import(() => ApplyFilter()), this);
						}
					} else {
						EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Target.Progress * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Opacity(0.75f));
						EditorGUI.LabelField(new Rect(EditorGUILayout.GetControlRect()), Target.CurrentFile + " / " + Target.TotalFiles);
						EditorGUI.BeginDisabledGroup(!Target.Importing);
						if(Utility.GUIButton(!Target.Importing ? "Aborting" : "Stop", !Target.Importing ? UltiDraw.Gold : UltiDraw.DarkRed, UltiDraw.White)) {
							Target.Importing = false;
						}
						EditorGUI.EndDisabledGroup();
					}
				}
			}

			//DATA SECTION
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Yellow);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Data Inspector");
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					//Selection Browser
					int pivot = MotionEditor.FindIndex(Assets, Utility.GetAssetGUID(Target.Data));
					EditorGUILayout.BeginHorizontal();
					EditorGUI.BeginChangeCheck();
					int selectIndex = EditorGUILayout.Popup(pivot, Enums);
					if(EditorGUI.EndChangeCheck()) {
						if(selectIndex != -1) {
							Target.LoadData(Assets[selectIndex]);
						}
					}
					
					EditorGUI.BeginChangeCheck();
					Filter = EditorGUILayout.TextField(Filter, GUILayout.Width(200f));
					if(EditorGUI.EndChangeCheck()) {
						ApplyFilter();
					}

					if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 55f)) {
						LoadPreviousAsset();
					}
					if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 55f)) {
						LoadNextAsset();
					}

					EditorGUILayout.EndHorizontal();
					
					//Slider Browser
					EditorGUILayout.BeginHorizontal();
					if(Assets.Length == 0) {
						EditorGUILayout.IntSlider(0, 0, 0);
					} else {
						EditorGUI.BeginChangeCheck();
						int sliderIndex = EditorGUILayout.IntSlider(pivot+1, 1, Assets.Length);
						if(EditorGUI.EndChangeCheck()) {
							Target.LoadData(Assets[sliderIndex-1]);
						}
					}
					EditorGUILayout.LabelField("/ " + Assets.Length, GUILayout.Width(60f));
					EditorGUILayout.EndHorizontal();

					if(Target.Data != null) {
						Frame frame = Target.GetCurrentFrame();

						Utility.SetGUIColor(UltiDraw.Grey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();

							Utility.SetGUIColor(UltiDraw.Mustard);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();

								Utility.SetGUIColor(UltiDraw.LightGrey);
								EditorGUILayout.ObjectField(Target.Data, typeof(MotionData), true);
								Utility.ResetGUIColor();
								EditorGUILayout.HelpBox(Target.Data.GetParentDirectoryPath(), MessageType.None);
							}

							EditorGUILayout.BeginVertical(GUILayout.Height(25f));
							Rect ctrl = EditorGUILayout.GetControlRect();
							Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 25f);
							EditorGUI.DrawRect(rect, UltiDraw.Black);

							//Sequences
							for(int i=0; i<Target.Data.Sequences.Length; i++) {
								Target.DrawRect(Target.Data.GetFrame(Target.Data.Sequences[i].Start), Target.Data.GetFrame(Target.Data.Sequences[i].End), 1f, !Target.Data.Export ? UltiDraw.White.Opacity(.25f) : UltiDraw.Green.Opacity(0.25f), rect);
							}

							//Valid Window
							Target.DrawRect(Target.Data.Frames.First(), Target.Data.GetFirstValidFrame(), 1f, UltiDraw.DarkRed.Opacity(0.5f), rect);
							Target.DrawRect(Target.Data.GetLastValidFrame(), Target.Data.Frames.Last(), 1f, UltiDraw.DarkRed.Opacity(0.5f), rect);

							//Current Pivot
							Target.DrawPivot(rect);

							EditorGUILayout.EndVertical();

							Utility.SetGUIColor(UltiDraw.DarkGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								EditorGUILayout.BeginHorizontal();
								GUILayout.FlexibleSpace();
								if(Target.Playing) {
									if(Utility.GUIButton("||", Color.red, Color.black, 50f, 40f)) {
										Target.StopAnimation();
									}
								} else {
									if(Utility.GUIButton("|>", Color.green, Color.black, 50f, 40f)) {
										Target.PlayAnimation();
									}
								}
								if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 40f)) {
									Target.LoadFrame(Mathf.Max(frame.Timestamp - Target.Data.GetDeltaTime(), 0f));
								}
								if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 40f)) {
									Target.LoadFrame(Mathf.Min(frame.Timestamp + Target.Data.GetDeltaTime(), Target.Data.GetTotalTime()));
								}

								int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.Data.GetTotalFrames(), GUILayout.Height(40f));
								if(index != frame.Index) {
									Target.LoadFrame(index);
								}
								EditorGUILayout.BeginVertical();
								EditorGUILayout.LabelField("/ " + Target.Data.GetTotalFrames() + " @ " + Mathf.RoundToInt(Target.Data.Framerate) + "Hz", Utility.GetFontColor(Color.white), GUILayout.Width(80f));
								EditorGUILayout.LabelField("[" + frame.Timestamp.ToString("F2") + "s / " + Target.Data.GetTotalTime().ToString("F2") + "s]", Utility.GetFontColor(Color.white), GUILayout.Width(80f));
								EditorGUILayout.EndVertical();
								GUILayout.FlexibleSpace();
								EditorGUILayout.EndHorizontal();
							}
						}

						Target.Data.Inspector(Target);
					}
				}
			}

			//EDITOR SECTION
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Yellow);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Editor Settings");
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Target.SetModel((Actor)EditorGUILayout.ObjectField("Character", Target.Model, typeof(Actor), true));

					Target.SetTargetFramerate(EditorGUILayout.FloatField("Target Framerate", Target.TargetFramerate));

					Target.Timescale = EditorGUILayout.FloatField("Timescale", Target.Timescale);

					Target.SetRandomSeed(EditorGUILayout.IntField("Random Seed", Target.RandomSeed));

					Target.Zoom = EditorGUILayout.Slider("Zoom", Target.Zoom, 0f, 1f);

					Utility.SetGUIColor(UltiDraw.DarkGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						GUILayout.FlexibleSpace();
						EditorGUILayout.LabelField("Past Keys", Utility.GetFontColor(Color.white), GUILayout.Width(90f));
						Target.SetPastKeys(EditorGUILayout.IntField(Target.PastKeys, GUILayout.Width(50f)));
						EditorGUILayout.LabelField("Future Keys", Utility.GetFontColor(Color.white), GUILayout.Width(90f));
						Target.SetFutureKeys(EditorGUILayout.IntField(Target.FutureKeys, GUILayout.Width(50f)));
						EditorGUILayout.LabelField("Past Window", Utility.GetFontColor(Color.white), GUILayout.Width(90f));
						Target.SetPastWindow(EditorGUILayout.FloatField(Target.PastWindow ,GUILayout.Width(50f)));
						EditorGUILayout.LabelField("Future Window", Utility.GetFontColor(Color.white), GUILayout.Width(90f));
						Target.SetFutureWindow(EditorGUILayout.FloatField(Target.FutureWindow, GUILayout.Width(50f)));
						GUILayout.FlexibleSpace();
						EditorGUILayout.EndHorizontal();
					}

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Mark As First Valid Frame", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.Data.SetFirstValidFrame(Target.GetCurrentFrame());
					}
					if(Utility.GUIButton("Mark As Last Valid Frame", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.Data.SetLastValidFrame(Target.GetCurrentFrame());
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Visualize First Frame", Target.VisualizeFirstFrame ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.VisualizeFirstFrame = !Target.VisualizeFirstFrame;
					}
					if(Utility.GUIButton("Visualize Last Frame", Target.VisualizeLastFrame ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.VisualizeLastFrame = !Target.VisualizeLastFrame;
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Visualize", Target.Visualize ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.SetVisualize(!Target.Visualize);
					}
					if(Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.SetMirror(!Target.Mirror);
					}
					if(Utility.GUIButton("Precomputable", Target.Precomputable ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.SetPrecomputable(!Target.Precomputable);
					}
					EditorGUILayout.EndHorizontal();

					Utility.SetGUIColor(UltiDraw.DarkGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Camera Focus", Target.CameraFocus ? UltiDraw.Cyan : UltiDraw.DarkGrey, Target.CameraFocus ? UltiDraw.Black : UltiDraw.White)) {
							Target.SetCameraFocus(!Target.CameraFocus);
						}
						if(Target.CameraFocus) {
							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
								Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
								Target.FocusAngle.y = EditorGUILayout.Slider("Focus Angle Horizontal", Target.FocusAngle.y, -180f, 180f);
								Target.FocusAngle.x = EditorGUILayout.Slider("Focus Angle Vertical", Target.FocusAngle.x, -180f, 180f);
								Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
							}
						}
					}

					Utility.SetGUIColor(UltiDraw.DarkGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Motion Trail", Target.MotionTrail ? UltiDraw.Cyan : UltiDraw.DarkGrey, Target.MotionTrail ? UltiDraw.Black : UltiDraw.White)) {
							Target.SetMotionTrail(!Target.MotionTrail);
						}
						if(Target.MotionTrail) {
							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								Target.CharacterMeshTrail = EditorGUILayout.Toggle("Character Mesh", Target.CharacterMeshTrail);
								Target.SkeletonSketchTrail = EditorGUILayout.Toggle("Skeleton Sketch", Target.SkeletonSketchTrail);
								Target.ApplyMotionTrail();
							}
						}
					}
					
				}
			}
			//END
		}
	}
}
#endif