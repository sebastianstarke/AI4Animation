#if UNITY_EDITOR
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

[ExecuteInEditMode]
public class MotionEditor : MonoBehaviour {

	public string Folder = string.Empty;
	public Actor Character = null;
	public MotionData[] Files = new MotionData[0];
	public bool Save = true;

	public float TargetFramerate = 60f;
	public int RandomSeed = 0;

	public bool Callbacks = true;
	public bool Mirror = false;
	public bool Visualise = true;
	public bool Settings = false;

	private bool CameraFocus = false;
	private float FocusHeight = 1f;
	private float FocusDistance = 1f;
	private float FocusAngle = 90f;
	private float FocusSmoothing = 0.05f;

	private bool Playing = false;
	private float Timescale = 1f;
	private float Timestamp = 0f;
	private float Zoom = 1f;

	private int[] BoneMapping = new int[0];

	[SerializeField] private MotionData Data = null;
	[SerializeField] private Actor Actor = null;

	public static MotionEditor GetInstance() {
		return GameObject.FindObjectOfType<MotionEditor>();
	}

	public void Refresh() {
		for(int i=0; i<Files.Length; i++) {
            if(Files[i] == null) {
				Debug.Log("Removing missing file from editor.");
				ArrayExtensions.RemoveAt(ref Files, i);
				for(int j=0; j<EditorSceneManager.sceneCount; j++) {
					Scene scene = EditorSceneManager.GetSceneAt(j);
					if(scene != EditorSceneManager.GetActiveScene()) {
						if(!System.Array.Find(Files, x => x != null && x.GetName() == scene.name)) {
							EditorSceneManager.CloseScene(scene, true);
							break;
						}
					}
				}
				i--;
			}
        }
		if(Data == null && Files.Length > 0) {
			LoadData(Files[0]);
		}
	}

	public void LoadBoneMapping() {
		BoneMapping = new int[GetActor().Bones.Length];
		for(int i=0; i<GetActor().Bones.Length; i++) {
			MotionData.Hierarchy.Bone bone = Data.Source.FindBone(GetActor().Bones[i].GetName());
			BoneMapping[i] = bone == null ? -1 : bone.Index;
		}
	}

	public void UpdateBoneMapping() {
		if(Data == null) {
			BoneMapping = new int[0];
		} else {
			if(BoneMapping == null || BoneMapping.Length != GetActor().Bones.Length) {
				LoadBoneMapping();
			}
		}
	}

	public void SetCallbacks(bool value) {
		if(Callbacks != value) {
			Callbacks = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetMirror(bool value) {
		if(Mirror != value) {
			Mirror = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetOffset(Vector3 value) {
		if(Data.Offset != value) {
			Data.Offset = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetScale(float value) {
		if(Data.Scale != value) {
			Data.Scale = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetTargetFramerate(float value) {
		TargetFramerate = Data == null ? 1 : Mathf.Clamp(value, 1, Data.Framerate);
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
			if(!CameraFocus) {
				Vector3 position =  SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion rotation = Quaternion.Euler(0f, SceneView.lastActiveSceneView.camera.transform.rotation.eulerAngles.y, 0f);
				SceneView.lastActiveSceneView.LookAtDirect(position, rotation, 0f);
			}
			LoadFrame(Timestamp);
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
		startTime = Mathf.Max(0f, startTime);
		endTime = Mathf.Min(Data.GetTotalTime(), endTime);
		int start = Data.GetFrame(startTime).Index;
		int end = Data.GetFrame(endTime).Index;
		int elements = end-start+1;
		return new Vector3Int(start, end, elements);
	}
	
	public void SetCharacter(Actor character) {
		if(Character == null && character != null) {
			if(Actor != null) {
				Utility.Destroy(Actor.gameObject);
				Actor = null;
			}
			Character = character;
			LoadFrame(Timestamp);
		} else {
			Character = character;
		}
	}

	public MotionData GetData() {
		return Data;
	}

	public Actor GetActor() {
		if(Character != null) {
			return Character;
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

	public void Import() {
		LoadData((MotionData)null);
		string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{Folder});
		Files = new MotionData[assets.Length];
		for(int i=0; i<assets.Length; i++) {
			Files[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[i]), typeof(MotionData));
		}
	}

	public void LoadData(string name) {
		if(Data != null && Data.GetName() == name) {
			return;
		}
		MotionData data = System.Array.Find(Files, x => x.GetName() == name);
		if(data == null) {
			Debug.Log("Data " + name + " could not be found.");
			return;
		}
		LoadData(data);
	}

	public void LoadData(MotionData data) {
		if(Data != data) {
			if(Data != null) {
				if(Save) {
					Data.Save();
				}
				Data.Unload();
			}
			if(Character == null && Actor != null) {
				Utility.Destroy(Actor.gameObject);
			}
			Data = data;
			if(Data != null) {
				Data.Load();
				LoadBoneMapping();
				LoadFrame(0f);
			}
		}
	}

	public void LoadPreviousData() {
		if(Data == null) {
			return;
		}
		LoadData(Files[Mathf.Max(System.Array.FindIndex(Files, x => x==Data)-1, 0)]);
	}

	public void LoadNextData() {
		if(Data == null) {
			return;
		}
		LoadData(Files[Mathf.Min(System.Array.FindIndex(Files, x => x==Data)+1, Files.Length-1)]);
	}

	public void LoadFrame(float timestamp) {
		Timestamp = timestamp;
		Actor actor = GetActor();
		Scene scene = Data.GetScene();
		Frame frame = GetCurrentFrame();
		RootModule module = (RootModule)Data.GetModule(Module.ID.Root);
		Matrix4x4 root = module == null ? frame.GetBoneTransformation(0, Mirror) : module.GetRootTransformation(frame, Mirror);
		actor.transform.position = root.GetPosition();
		actor.transform.rotation = root.GetRotation();
		UpdateBoneMapping();
		for(int i=0; i<actor.Bones.Length; i++) {
			if(BoneMapping[i] == -1) {
				Debug.Log("Bone " + actor.Bones[i].GetName() + " could not be mapped.");
			} else {
				Matrix4x4 transformation = frame.GetBoneTransformation(BoneMapping[i], Mirror);
				Vector3 velocity = frame.GetBoneVelocity(BoneMapping[i], Mirror, 1f/TargetFramerate);
				Vector3 acceleration = frame.GetBoneAcceleration(BoneMapping[i], Mirror, 1f/TargetFramerate);
				Vector3 force = frame.GetBoneMass(BoneMapping[i], Mirror) * acceleration;
				actor.Bones[i].Transform.position = transformation.GetPosition();
				actor.Bones[i].Transform.rotation = transformation.GetRotation();
				actor.Bones[i].Velocity = velocity;
				actor.Bones[i].Acceleration = acceleration;
				actor.Bones[i].Force = force;
			}
		}
		foreach(GameObject instance in scene.GetRootGameObjects()) {
			instance.transform.localScale = Vector3.one.GetMirror(Mirror ? Data.MirrorAxis : Axis.None);
			foreach(SceneEvent e in instance.GetComponentsInChildren<SceneEvent>(true)) {
				if(Callbacks) {
					e.Callback(this);
				} else {
					e.Identity(this);
				}
			}
		}
		foreach(Module m in Data.Modules) {
			if(Callbacks) {
				m.Callback(this);
			}
		}
		if(CameraFocus) {
			if(SceneView.lastActiveSceneView != null) {
				/*
				Vector3 lastPosition = SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion lastRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
				Vector3 position = GetActor().GetRoot().position;
				position.y += FocusHeight;
				Quaternion rotation = GetActor().GetRoot().rotation;
				rotation.x = 0f;
				rotation.z = 0f;
				rotation = Quaternion.Euler(0f, Mirror ? Mathf.Repeat(FocusAngle + 0f, 360f) : FocusAngle, 0f) * rotation;
				position += FocusOffset * (rotation * Vector3.right);
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
				*/

				Vector3 lastPosition = SceneView.lastActiveSceneView.camera.transform.position;
				Quaternion lastRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
				Vector3 position = GetActor().GetRoot().position;
				position += Quaternion.Euler(0f, FocusAngle, 0f) * (FocusDistance * Vector3.forward);
				position.y += FocusHeight;
				Quaternion rotation = Quaternion.LookRotation(Vector3.ProjectOnPlane(GetActor().GetRoot().position - position, Vector3.up).normalized, Vector3.up);
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
			}
		}
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
				LoadFrame(Mathf.Repeat(Timestamp + delta, Data.GetTotalTime()));
			}
			yield return new WaitForSeconds(0f);
		}
	}

	public void Draw() {
		if(Data != null) {
			for(int i=0; i<Data.Modules.Length; i++) {
				Data.Modules[i].Draw(this);
			}
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

	public void DrawPivot(Rect rect) {
		UltiDraw.Begin();
		Frame frame = GetCurrentFrame();
		Vector3 view = GetView();
		Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
		Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);
		float pStart = (float)(Data.GetFrame(Mathf.Clamp(frame.Timestamp-1f, 0f, Data.GetTotalTime())).Index-view.x) / (view.z-1);
		float pEnd = (float)(Data.GetFrame(Mathf.Clamp(frame.Timestamp+1f, 0f, Data.GetTotalTime())).Index-view.x) / (view.z-1);
		float pLeft = rect.x + pStart * rect.width;
		float pRight = rect.x + pEnd * rect.width;
		Vector3 pA = new Vector3(pLeft, rect.y, 0f);
		Vector3 pB = new Vector3(pRight, rect.y, 0f);
		Vector3 pC = new Vector3(pLeft, rect.y+rect.height, 0f);
		Vector3 pD = new Vector3(pRight, rect.y+rect.height, 0f);
		UltiDraw.DrawTriangle(pA, pC, pB, UltiDraw.White.Transparent(0.1f));
		UltiDraw.DrawTriangle(pB, pC, pD, UltiDraw.White.Transparent(0.1f));
		top.x = rect.xMin + (float)(frame.Index-view.x)/(view.z-1) * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index-view.x)/(view.z-1) * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
		UltiDraw.End();
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		private float RepaintRate = 10f;
		private System.DateTime Timestamp;

		public string[] Names = new string[0];
		public string[] EnumNames = new string[0];
		public string NameFilter = "";

		void Awake() {
			Target = (MotionEditor)target;
			Target.Refresh();
			ComputeNames();
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			EditorApplication.update -= EditorUpdate;
			if(Target.Data != null) {
				Target.Data.Save();
			}
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
		}

		public void ComputeNames() {
			List<string> names = new List<string>();
			List<string> enumNames = new List<string>();
			for(int i=0; i<Target.Files.Length; i++) {
				if(Target.Files[i].GetName().ToLowerInvariant().Contains(NameFilter.ToLowerInvariant())) {
					names.Add(Target.Files[i].GetName());
					enumNames.Add("[" + (i+1) + "]" + " " + Target.Files[i].GetName());
				}
			}
			Names = names.ToArray();
			EnumNames = enumNames.ToArray();
		}

		public void SetNameFilter(string filter) {
			if(NameFilter != filter) {
				NameFilter = filter;
				ComputeNames();
			}
		}

		public void Import() {
			Target.Import();
			ComputeNames();
		}

		public void Inspector() {
			Target.Refresh();

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

					Target.SetCharacter((Actor)EditorGUILayout.ObjectField("Character", Target.Character, typeof(Actor), true));

					SetNameFilter(EditorGUILayout.TextField("Name Filter", NameFilter));

					if(Target.Data != null) {
						Frame frame = Target.GetCurrentFrame();

						Utility.SetGUIColor(UltiDraw.Grey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();

							EditorGUILayout.BeginHorizontal();
							GUILayout.FlexibleSpace();
							int selectIndex = EditorGUILayout.Popup(System.Array.FindIndex(Names, x => x==Target.Data.GetName()), EnumNames);
							if(selectIndex != -1) {
								Target.LoadData(Names[selectIndex]);
							}
							if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.LoadPreviousData();
								NameFilter = string.Empty;
							}
							if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.LoadNextData();
							}
							int sliderIndex = EditorGUILayout.IntSlider(System.Array.FindIndex(Target.Files, x => x==Target.Data)+1, 1, Target.Files.Length);
							if(Event.current.type == EventType.Used) {
								Target.LoadData(Target.Files[sliderIndex-1]);
							}
							EditorGUILayout.LabelField("/ " + Target.Files.Length, GUILayout.Width(60f));
							if(Utility.GUIButton("Save", Target.Save ? UltiDraw.Magenta : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.Save = !Target.Save;
							}
							GUILayout.FlexibleSpace();
							EditorGUILayout.EndHorizontal();

							Utility.SetGUIColor(UltiDraw.Mustard);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();

								EditorGUILayout.BeginHorizontal();
								EditorGUILayout.LabelField("Data", GUILayout.Width(50f));
								EditorGUILayout.ObjectField(Target.Data, typeof(MotionData), true);
								if(Utility.GUIButton("Export", Target.Data.Export ? UltiDraw.Cyan : UltiDraw.Grey, Target.Data.Export ? UltiDraw.Black : UltiDraw.LightGrey)) {
									Target.Data.Export = !Target.Data.Export;
								}
								if(Utility.GUIButton("Symmetric", Target.Data.Symmetric ? UltiDraw.Cyan : UltiDraw.Grey, Target.Data.Symmetric ? UltiDraw.Black : UltiDraw.LightGrey)) {
									Target.Data.Symmetric = !Target.Data.Symmetric;
								}
								EditorGUILayout.EndHorizontal();

								EditorGUILayout.BeginHorizontal();
								GUILayout.FlexibleSpace();
								EditorGUILayout.LabelField("Frames: " + Target.Data.GetTotalFrames(), GUILayout.Width(100f));
								EditorGUILayout.LabelField("Time: " + Target.Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
								EditorGUILayout.LabelField("Framerate: " + Target.Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
								EditorGUILayout.LabelField("Target Framerate: ", GUILayout.Width(100f));
								Target.SetTargetFramerate(EditorGUILayout.FloatField(Target.TargetFramerate, GUILayout.Width(40f)));
								EditorGUILayout.LabelField("Random Seed: ", GUILayout.Width(80f));
								Target.SetRandomSeed(EditorGUILayout.IntField(Target.RandomSeed, GUILayout.Width(40f)));
								GUILayout.FlexibleSpace();
								EditorGUILayout.EndHorizontal();

							}

							if(Utility.GUIButton("Add Sequence", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.AddSequence();
							}
							for(int i=0; i<Target.Data.Sequences.Length; i++) {
								Utility.SetGUIColor(UltiDraw.White);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									
									EditorGUILayout.BeginHorizontal();
									GUILayout.FlexibleSpace();
									if(Utility.GUIButton("X", Color.cyan, Color.black, 20f, 15f)) {
										Target.Data.Sequences[i].SetStart(frame.Index);
									}
									EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
									Target.Data.Sequences[i].SetStart(Mathf.Clamp(EditorGUILayout.IntField(Target.Data.Sequences[i].Start, GUILayout.Width(100f)), 1, Target.Data.GetTotalFrames()));
									EditorGUILayout.LabelField("End", GUILayout.Width(50f));
									Target.Data.Sequences[i].SetEnd(Mathf.Clamp(EditorGUILayout.IntField(Target.Data.Sequences[i].End, GUILayout.Width(100f)), 1, Target.Data.GetTotalFrames()));
									if(Utility.GUIButton("X", Color.cyan, Color.black, 20f, 15f)) {
										Target.Data.Sequences[i].SetEnd(frame.Index);
									}

									if(Utility.GUIButton("X", UltiDraw.DarkRed, Color.black, 40f, 15f)) {
										Target.Data.RemoveSequence(Target.Data.Sequences[i]);
										i--;
									}
									GUILayout.FlexibleSpace();
									EditorGUILayout.EndHorizontal();
								}
							}

							EditorGUILayout.BeginVertical(GUILayout.Height(25f));
							Rect ctrl = EditorGUILayout.GetControlRect();
							Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 25f);
							EditorGUI.DrawRect(rect, UltiDraw.Black);

							//View
							Vector3 view = Target.GetView();

							//Sequences
							UltiDraw.Begin();
							for(int i=0; i<Target.Data.Sequences.Length; i++) {
								float _start = (float)(Mathf.Clamp(Target.Data.Sequences[i].Start, view.x, view.y)-view.x) / view.z;
								float _end = (float)(Mathf.Clamp(Target.Data.Sequences[i].End, view.x, view.y)-view.x) / view.z;
								float left = rect.x + _start * rect.width;
								float right = rect.x + _end * rect.width;
								Vector3 a = new Vector3(left, rect.y, 0f);
								Vector3 b = new Vector3(right, rect.y, 0f);
								Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
								Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
								UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
								UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
							}
							UltiDraw.End();

							//Current Pivot
							Target.DrawPivot(rect);

							EditorGUILayout.EndVertical();

							Utility.SetGUIColor(UltiDraw.DarkGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								EditorGUILayout.BeginHorizontal();
								//GUILayout.FlexibleSpace();
								if(Target.Playing) {
									if(Utility.GUIButton("||", Color.red, Color.black, 50f, 20f)) {
										Target.StopAnimation();
									}
								} else {
									if(Utility.GUIButton("|>", Color.green, Color.black, 50f, 20f)) {
										Target.PlayAnimation();
									}
								}
								if(Utility.GUIButton("<<", UltiDraw.Grey, UltiDraw.White, 30f, 20f)) {
									Target.LoadFrame(Mathf.Max(frame.Timestamp - 1f, 0f));
								}
								if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
									Target.LoadFrame(Mathf.Max(frame.Timestamp - 1f/Target.TargetFramerate, 0f));
								}
								if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
									Target.LoadFrame(Mathf.Min(frame.Timestamp + 1f/Target.TargetFramerate, Target.Data.GetTotalTime()));
								}
								if(Utility.GUIButton(">>", UltiDraw.Grey, UltiDraw.White, 30f, 20f)) {
									Target.LoadFrame(Mathf.Min(frame.Timestamp + 1f, Target.Data.GetTotalTime()));
								}
								int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.Data.GetTotalFrames());
								if(index != frame.Index) {
									Target.LoadFrame(index);
								}
								EditorGUILayout.LabelField(frame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
								//GUILayout.FlexibleSpace();
								EditorGUILayout.EndHorizontal();

								EditorGUILayout.BeginHorizontal();
								EditorGUILayout.LabelField("Timescale", Utility.GetFontColor(Color.white), GUILayout.Width(60f), GUILayout.Height(20f)); 
								Target.Timescale = EditorGUILayout.FloatField(Target.Timescale, GUILayout.Width(45f), GUILayout.Height(16f));
								EditorGUILayout.LabelField("Zoom", Utility.GetFontColor(Color.white), GUILayout.Width(52f));
								Target.Zoom = EditorGUILayout.Slider(Target.Zoom, 0f, 1f);
								EditorGUILayout.LabelField((100 - Mathf.RoundToInt(100f*(1f - Target.Zoom))) + "%", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
								EditorGUILayout.EndHorizontal();
							}
							
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("Visualise", Target.Visualise ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.Visualise = !Target.Visualise;
							}
							if(Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.SetMirror(!Target.Mirror);
							}
							if(Utility.GUIButton("Callbacks", Target.Callbacks ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.SetCallbacks(!Target.Callbacks);
							}
							EditorGUILayout.EndHorizontal();

							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("Inspect All", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.InspectAll(true);
							}
							if(Utility.GUIButton("Inspect None", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.InspectAll(false);
							}
							if(Utility.GUIButton("Visualise All", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.VisualiseAll(true);
							}
							if(Utility.GUIButton("Visualise None", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.VisualiseAll(false);
							}
							EditorGUILayout.EndHorizontal();

							for(int i=0; i<Target.Data.Modules.Length; i++) {
								Target.Data.Modules[i].Inspector(Target);
							}

							Utility.ResetGUIColor();
							Utility.SetGUIColor(UltiDraw.Cyan);
							int module = EditorGUILayout.Popup(0, ArrayExtensions.Concat(new string[1]{"Add Module..."}, Module.GetIDNames()));
							if(module > 0) {
								Target.Data.AddModule(((Module.ID)(module-1)));
							}
							Utility.ResetGUIColor();

							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								if(Utility.GUIButton("Camera Focus", Target.CameraFocus ? UltiDraw.Cyan : UltiDraw.Grey, Target.CameraFocus ? UltiDraw.Black : UltiDraw.LightGrey)) {
									Target.SetCameraFocus(!Target.CameraFocus);
								}
								if(Target.CameraFocus) {
									Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
									Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
									Target.FocusAngle = EditorGUILayout.Slider("Focus Angle", Target.FocusAngle, 0f, 360f);
									Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
								}

								if(Utility.GUIButton("Settings", Target.Settings ? UltiDraw.Cyan : UltiDraw.Grey, Target.Settings ? UltiDraw.Black : UltiDraw.LightGrey)) {
									Target.Settings = !Target.Settings;
								}
								if(Target.Settings) {
									Target.SetOffset(EditorGUILayout.Vector3Field("Offset", Target.Data.Offset));
									Target.SetScale(EditorGUILayout.FloatField("Scale", Target.Data.Scale));
									Target.Data.MirrorAxis = (Axis)EditorGUILayout.EnumPopup("Mirror Axis", Target.Data.MirrorAxis);
									for(int i=0; i<Target.Data.Source.Bones.Length; i++) {
										EditorGUILayout.BeginHorizontal();
										EditorGUI.BeginDisabledGroup(true);
										EditorGUILayout.TextField(Target.Data.Source.GetBoneNames()[i]);
										EditorGUI.EndDisabledGroup();
										Target.Data.SetSymmetry(i, EditorGUILayout.Popup(Target.Data.Symmetry[i], Target.Data.Source.GetBoneNames()));
										EditorGUILayout.LabelField("Mass", GUILayout.Width(40f));
										Target.Data.Source.Bones[i].Mass = EditorGUILayout.FloatField(Target.Data.Source.Bones[i].Mass);
										EditorGUILayout.LabelField("Alignment", GUILayout.Width(60f));
										Target.Data.Source.Bones[i].Alignment = EditorGUILayout.Vector3Field("", Target.Data.Source.Bones[i].Alignment);
										EditorGUILayout.EndHorizontal();
									}
								}
								if(Utility.GUIButton("Create Actor", UltiDraw.DarkGrey, UltiDraw.White)) {
									Target.Data.CreateActor();
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