#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using System;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	[ExecuteInEditMode]
	public class MotionEditor : MonoBehaviour {
		public bool AutoSave = true;

		public List<AssetDirectory> Directories = new List<AssetDirectory>();
		public List<string> Assets = new List<string>();

		public AssetSession Session = null;
		public string Asset = null;

		public float TargetFramerate = 60f;
		public float TargetDeltaTime {get{return 1f/TargetFramerate;}}

		public int PastKeys = 6;
		public int FutureKeys = 6;
		public float PastWindow = 1f;
		public float FutureWindow = 1f;
		public int Resolution = 1;
		public float Timestamp = 0f;
		public bool Mirror {get; private set;}

		public bool ShowCharacter = true;
		private bool ShowInRootSpace = false;
		private bool PreviewMotion = false;
		private bool PreviewMesh = false;
		private int PastStep = 1;
		private int FutureStep = 1;
		private bool UpdateMesh = true;
		private float MeshTransparency = 0.5f;
		private Vector3 PreviewSpacing = Vector3.zero;
		private bool CameraFocus = false;
		private Transform FocusTransform = null;
		private float FocusHeight = 1f;
		private float FocusDistance = 2f;
		private Vector2 FocusAngle = new Vector2(0f, 90f);
		private float FocusSmoothing = 0.05f;

		private TimeSeries TimeSeries = null;
		private EditorCoroutines.EditorCoroutine Coroutine = null;
		private float Timescale = 1f;
		private float Zoom = 1f;

		private bool Settings = false;

		public static MotionEditor GetInstance() {
			return GameObject.FindObjectOfType<MotionEditor>();
		}

		[Serializable]
		public class AssetDirectory {
			public string Path = string.Empty;
			public bool Import = true;
			public int Size = 0;

			public void Refresh() {
				Size = IsValid() ? GetAssets().Length : 0;
			}

			public string GetFolder() {
				string path = Path;
				if(path.EndsWith("/.")) {
					path = path.Remove(path.Length-2, 2);
				}
				return "Assets/" + path;
			}

			public bool IsValid() {
				return AssetDatabase.IsValidFolder(GetFolder());
			}

			public void SetFolder(string path) {
				if(Path != path) {
					Path = path;
					Size = IsValid() ? GetAssets().Length : 0;
				}
			}

			public string[] GetAssets() {
				string folder = GetFolder();
				string[] assets = MotionAsset.Search(folder);
				if(Path.EndsWith("/.")) {
					List<string> filtered = new List<string>();
					for(int i=0; i<assets.Length; i++) {
						string path = Utility.GetAssetPath(assets[i]);
						string filter = path.Remove(0, folder.Length+1);
						if(filter.Count('/') == 1) {
							filtered.Add(assets[i]);
						}
					}
					assets = filtered.ToArray();
				}
				return assets;
			}
		}

		public class AssetSession {
			public MotionAsset Asset = null;

			private Actor Actor = null;
			private int[] BoneMapping = null;

			private MotionEditor Editor;

			public AssetSession(MotionEditor editor) {
				Editor = editor;
			}

			public void Load(string guid) {
				Asset = (MotionAsset)AssetDatabase.LoadAssetAtPath(Utility.GetAssetPath(guid), typeof(MotionAsset));
				Asset.Load(Editor);
				Actor = GetActor();
			}
		
			public void Unload() {
				if(Asset != null && Asset.IsDirty()) {
					if(!Application.isPlaying && Editor.AutoSave) {
						AssetDatabase.SaveAssets();
						EditorSceneManager.SaveScene(Asset.GetScene());
					}
				}
				RemoveActor(Actor);
				MotionAsset asset = Asset;
				Asset = null;
				asset.Unload(Editor);
				Resources.UnloadUnusedAssets();
			}

			public void LoadFrame() {
				//Apply posture on character
				GetActor().transform.localScale = Vector3.one * Asset.Scale;
				for(int i=0; i<GetActor().Bones.Length; i++) {
					if(GetBoneMapping()[i] != -1) {
						GetActor().Bones[i].SetTransformation(Asset.GetFrame(Editor.GetTimestamp()).GetBoneTransformation(GetBoneMapping()[i], Editor.Mirror));
						GetActor().Bones[i].SetVelocity(Asset.GetFrame(Editor.GetTimestamp()).GetBoneVelocity(GetBoneMapping()[i], Editor.Mirror));
					}
				}

				//Apply scene changes
				foreach(GameObject instance in Asset.GetScene().GetRootGameObjects()) {
					instance.transform.localScale = Vector3.one.GetMirror(Editor.Mirror ? Asset.MirrorAxis : Axis.None);
					foreach(SceneEvent e in instance.GetComponentsInChildren<SceneEvent>(true)) {
						e.Callback(Editor);
					}
				}

				//Send callbacks to all modules
				Asset.Callback(Editor);
			}

			public Actor GetActor() {
				if(Actor == null) {
					BoneMapping = null;
					{
						if(Asset.Model != string.Empty) {
							Transform t = Editor.transform.Find(Asset.Model);
							if(t != null) {
								Actor = t.GetComponent<Actor>();
								if(Actor == null) {
									Actor = t.gameObject.AddComponent<Actor>();
								}
								Actor.gameObject.SetActive(Editor.ShowCharacter);
								return Actor;
							}
						}
					}
					{
						Transform t = Editor.transform.Find(Asset.name);
						if(t != null) {
							Actor = t.GetComponent<Actor>();
							Actor.gameObject.SetActive(Editor.ShowCharacter);
							return Actor;
						}
					}
					Actor = CreateActor();
					Actor.transform.SetParent(Editor.transform);
				}
				Actor.gameObject.SetActive(Editor.ShowCharacter);
				return Actor;
			}

			public Actor CreateActor() {
				Transform root = new GameObject(Asset.name).transform;
				List<Transform> bones = new List<Transform>();
				for(int i=0; i<Asset.Source.Bones.Length; i++) {
					Transform instance = new GameObject(Asset.Source.Bones[i].Name).transform;
					MotionAsset.Hierarchy.Bone parent = Asset.Source.Bones[i].GetParent(Asset.Source);
					instance.SetParent(parent == null ? root : root.FindRecursive(parent.Name));
					Matrix4x4 matrix = Asset.Frames.First().GetBoneTransformation(i, false);
					instance.position = matrix.GetPosition();
					instance.rotation = matrix.GetRotation();
					instance.localScale = Vector3.one;
					bones.Add(instance);
				}
				root.position = new Vector3(0f, root.position.y, 0f);
				root.rotation = Quaternion.Euler(root.eulerAngles.x, 0f, root.eulerAngles.z);
				Actor actor = root.gameObject.AddComponent<Actor>();
				actor.Create(bones.ToArray());
				return actor;
			}

			public void RemoveActor(Actor actor) {
				if(actor != null && Asset != null && actor.name == Asset.name) {
					Utility.Destroy(actor.gameObject);
				}
			}

			public int[] GetBoneMapping() {
				if(BoneMapping == null || BoneMapping.Length != GetActor().Bones.Length) {
					BoneMapping = Asset.Source.GetBoneIndices(GetActor().GetBoneNames());
				}
				return BoneMapping;
			}

			public void Inspector() {
				Asset.Inspector(Editor);
				for(int i=0; i<GetBoneMapping().Length; i++) {
					if(GetBoneMapping()[i] == -1) {
						EditorGUILayout.HelpBox("Bone " + Actor.Bones[i].GetName() + " could not be mapped.", MessageType.Warning);
					}
				}
			}
		}

		void OnDestroy() {
			if(IsPlaying()) {
				Play(false);
			}
			if(this.IsInvokedDestroy()) {
				CloseSession(true);
			}
		}

		void Update() {
			if(Application.isPlaying && IsPlaying()) {
				LoadFrame(Mathf.Repeat(Timestamp + Timescale * Time.deltaTime, GetSession().Asset.Frames.Last().Timestamp));
			}
		}

		public void LoadSession(string guid, bool unloadCurrentSession=true) {
			if(Session != null) {
				CloseSession(unloadCurrentSession);
			}
			if(guid != null) {
				OpenSession(guid);
			}
		}
		
		private void OpenSession(string guid) {
			if(Session != null) {
				Debug.Log("Another session for asset " + Session.Asset.name + " is still open.");
			} else {
				foreach(Actor actor in GetComponentsInChildren<Actor>()) {
					actor.gameObject.SetActive(false);
				}
				Session = new AssetSession(this);
				Session.Load(guid);
				Asset = guid;
				LoadFrame(Mathf.Clamp(Timestamp, 0f, GetSession().Asset.GetTotalTime()));
			}
		}

		private void CloseSession(bool unloadCurrentSession) {
			if(Session == null) {
				Debug.Log("No session currently open.");
			} else {
				if(unloadCurrentSession) {
					Session.Unload();
				}
				Session = null;
				Asset = null;
			}
		}

		public AssetSession GetSession() {
			if(Session == null && Asset != null && Assets.Contains(Asset) && MotionAsset.Retrieve(Asset) != null) {
				LoadSession(Asset);
			}
			return Session;
		}

		public void Import() {
			Assets = new List<string>();
			foreach(AssetDirectory directory in Directories) {
				if(directory.IsValid() && directory.Import) {
					Assets.AddRange(directory.GetAssets());
				}
			}
			LoadSession(Assets.First());
		}

		public TimeSeries GetTimeSeries() {
			if(TimeSeries == null || TimeSeries.PastKeys != PastKeys || TimeSeries.FutureKeys != FutureKeys || TimeSeries.PastWindow != PastWindow || TimeSeries.FutureWindow != FutureWindow || TimeSeries.Resolution != Resolution) {
				TimeSeries = new TimeSeries(PastKeys, FutureKeys, PastWindow, FutureWindow, Resolution);
			}
			return TimeSeries;
		}

		public void SetMirror(bool value) {
			if(Mirror != value) {
				Mirror = value;
				LoadFrame(Timestamp);
			}
		}

		public void SetTargetFramerate(float value) {
			if(TargetFramerate != value) {
				TargetFramerate = value;
				LoadFrame(Timestamp);
			}
		}

		public float GetTimestamp() {
			return Timestamp;
		}

		public void LoadFrame(float timestamp) {
			if(IsSetup()) {
				Timestamp = timestamp;
				GetSession().LoadFrame();
				if(CameraFocus) {
					if(SceneView.lastActiveSceneView != null) {
						Vector3 currentPosition = SceneView.lastActiveSceneView.camera.transform.position;
						Quaternion currentRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
						Matrix4x4 matrix = FocusTransform == null ? (GetSession().GetActor().Bones.Length == 0 ? GetSession().GetActor().GetRoot().GetWorldMatrix() : GetSession().GetActor().Bones.First().GetTransformation()) : FocusTransform.GetWorldMatrix();
						Vector3 target = matrix.GetPosition().SetY(FocusHeight);
						Vector3 position = target + Quaternion.Euler(-FocusAngle.x, -FocusAngle.y, 0f) * new Vector3(0f, 0f, FocusDistance);
						Quaternion rotation = Quaternion.LookRotation((target - position).normalized, Vector3.up);				
						SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(currentPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(currentRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
					}
				}
				if(ShowInRootSpace) {
					Matrix4x4 root = GetSession().GetActor().transform.GetWorldMatrix();
					// Matrix4x4 root = GetSession().GetActor().Bones.First().GetTransformation();
					Matrix4x4[] transformations = GetSession().GetActor().GetBoneTransformations();
					for(int i=0; i<GetSession().GetActor().Bones.Length; i++) {
						GetSession().GetActor().Bones[i].SetTransformation(transformations[i].TransformationTo(root));
					}
				}
			}
		}

		public Frame GetCurrentFrame() {
			return GetSession().Asset.GetFrame(GetTimestamp());
		}

		public void Play(bool value) {
			if(value && IsPlaying()) {
				return;
			}
			if(!value && !IsPlaying()) {
				return;
			}
			if(value) {
				Coroutine = EditorCoroutines.StartCoroutine(Play(), this);
				GetSession().Asset.OnTriggerPlay(this);
			} else {
				EditorCoroutines.StopCoroutine(Play(), this);
				Coroutine = null;
				GetSession().Asset.OnTriggerPlay(this);
			}
		}

		public bool IsPlaying() {
			return Coroutine != null;
		}

		public bool IsSetup() {
			return GetSession() != null && GetSession().Asset != null;
		}

		private IEnumerator Play() {
			System.DateTime previous = Utility.GetTimestamp();
			while(IsSetup()) {
				if(!Application.isPlaying) {
					float delta = Timescale * (float)Utility.GetElapsedTime(previous);
					if(Timescale == 0f || delta > 1f/TargetFramerate) {
						previous = Utility.GetTimestamp();
						LoadFrame(Mathf.Repeat(Timestamp + delta, GetSession().Asset.Frames.Last().Timestamp));
					}
				}
				yield return new WaitForSeconds(0f);
			}
		}

		void OnGUI() {
			if(IsSetup()) {
				GetSession().Asset.GUI(this);
			}
		}

		void OnRenderObject() {
			Vector3 GetSpacing(TimeSeries.Sample key) {
				return Vector3.Scale(PreviewSpacing, key.Timestamp * Vector3.one);
			}

			if(IsSetup()) {
				if(PreviewMotion) {
					TimeSeries timeSeries = GetTimeSeries();
					for(int i=0; i<timeSeries.PivotKey; i++) {
						float t = timeSeries.GetKey(i).Timestamp + GetTimestamp();
						float ratio = i.Ratio(-1, timeSeries.PivotKey);
						ratio = Mathf.Sqrt(ratio);
						Matrix4x4[] transformations = GetSession().Asset.GetFrame(t).GetBoneTransformations(GetSession().GetBoneMapping(), Mirror);
						for(int j=0; j<transformations.Length; j++) {
							Matrix4x4Extensions.SetPosition(ref transformations[j], transformations[j].GetPosition() + GetSpacing(timeSeries.GetKey(i)));
						}
						GetSession().GetActor().Draw(
							transformations,
							GetSession().GetActor().BoneColor.Opacity(ratio),
							GetSession().GetActor().JointColor.Opacity(ratio),
							Actor.DRAW.Skeleton
						);
					}
					for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
						float t = timeSeries.GetKey(i).Timestamp + GetTimestamp();
						float ratio = 1f - i.Ratio(timeSeries.PivotKey, timeSeries.KeyCount);
						ratio = Mathf.Sqrt(ratio);
						Matrix4x4[] transformations = GetSession().Asset.GetFrame(t).GetBoneTransformations(GetSession().GetBoneMapping(), Mirror);
						for(int j=0; j<transformations.Length; j++) {
							Matrix4x4Extensions.SetPosition(ref transformations[j], transformations[j].GetPosition() + GetSpacing(timeSeries.GetKey(i)));
						}
						GetSession().GetActor().Draw(
							transformations,
							GetSession().GetActor().BoneColor.Opacity(ratio),
							GetSession().GetActor().JointColor.Opacity(ratio),
							Actor.DRAW.Skeleton
						);
						// GetSession().GetActor().DrawRootTransformation(
						// 	GetSession().Asset.GetModule<RootModule>("BodyWorld").GetRootTransformation(t, Mirror)
						// );
					}
				}
				if(PreviewMesh && UpdateMesh) {
					Transform instances = transform.Find("MeshInstances");
					if(instances == null) {
						instances = new GameObject("MeshInstances").transform;
						instances.SetParent(transform);
						instances.transform.localPosition = Vector3.zero;
						instances.transform.localRotation = Quaternion.identity;
					}
					List<Transform> childs = new List<Transform>(instances.GetChilds());
					Transform GetChild(int index) {
						Transform child = childs.Find(x => x.name == index.ToString());
						if(child == null) {
							child = Instantiate(GetSession().GetActor().gameObject).transform;
							child.SetParent(instances);
							child.name = index.ToString();
							foreach(Component c in child.GetComponents<MonoBehaviour>()) {
								if(!(c is Actor)) {
									Utility.Destroy(c);
								} else {
									Actor actor = c as Actor;
									actor.DrawSkeleton = false;
								}
							}
							child.gameObject.AddComponent<Transparency>();
						} else {
							childs.Remove(child);
						}
						return child;
					}
					TimeSeries timeSeries = GetTimeSeries();
					if(PastStep > 0) {
						for(int i=0; i<timeSeries.PivotKey; i+=PastStep) {
							float t = timeSeries.GetKey(i).Timestamp + GetTimestamp();
							Transform child = GetChild(i);

							float ratio = i.Ratio(-1, timeSeries.PivotKey);
							ratio = Mathf.Pow(ratio, 2f);
							child.GetComponent<Transparency>().SetTransparency(ratio.Normalize(0f, 1f, MeshTransparency, 1f));

							child.GetComponent<Actor>().SetBoneTransformations(GetSession().Asset.GetFrame(t).GetBoneTransformations(GetSession().GetBoneMapping(), Mirror));
							foreach(Actor.Bone bone in child.GetComponent<Actor>().GetRootBones()) {
								bone.SetPosition(bone.GetPosition() + GetSpacing(timeSeries.GetKey(i)));
							}
						}
					}
					if(FutureStep > 0) {
						for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i+=FutureStep) {
							float t = timeSeries.GetKey(i).Timestamp + GetTimestamp();
							Transform child = GetChild(i);

							float ratio = 1f - i.Ratio(timeSeries.PivotKey, timeSeries.KeyCount);
							ratio = Mathf.Pow(ratio, 2f);
							child.GetComponent<Transparency>().SetTransparency(ratio.Normalize(0f, 1f, MeshTransparency, 1f));

							child.GetComponent<Actor>().SetBoneTransformations(GetSession().Asset.GetFrame(t).GetBoneTransformations(GetSession().GetBoneMapping(), Mirror));
							foreach(Actor.Bone bone in child.GetComponent<Actor>().GetRootBones()) {
								bone.SetPosition(bone.GetPosition() + GetSpacing(timeSeries.GetKey(i)));
							}
						}
					}
					foreach(Transform child in childs) {
						Utility.Destroy(child.gameObject);
					}
				} else if(!PreviewMesh) {
					Transform instances = transform.Find("MeshInstances");
					if(instances != null) {
						Utility.Destroy(instances.gameObject);
					}
				}
				GetSession().Asset.Draw(this);
			}
		}

		public Actor[] GetMeshInstances() {
			Transform instances = transform.Find("MeshInstances");
			if(instances == null) {
				Debug.Log("Mesh instances could not be found.");
				return null;
			}
			return instances.GetComponentsInChildren<Actor>();
		}

		public void UpdateMeshInstance(int key, Matrix4x4[] transformations) {
			Transform instances = transform.Find("MeshInstances");
			if(instances == null) {
				Debug.Log("Mesh instances could not be found.");
				return;
			}
			Transform instance = instances.Find(key.ToString());
			if(instance == null) {
				Debug.Log("Key instance could not be found.");
				return;
			}
			instance.GetComponent<Actor>().SetBoneTransformations(transformations);
		}

		public float GetWindow() {
			return Asset == null ? 0f : Zoom * GetSession().Asset.GetTotalTime();
		}

		public Vector3Int GetView() {
			float window = Asset == null ? 0f : Zoom * GetSession().Asset.GetTotalTime();
			float startTime = GetTimestamp()-window/2f;
			float endTime = GetTimestamp()+window/2f;
			if(startTime < 0f) {
				endTime -= startTime;
				startTime = 0f;
			}
			if(endTime > GetSession().Asset.GetTotalTime()) {
				startTime -= endTime-GetSession().Asset.GetTotalTime();
				endTime = GetSession().Asset.GetTotalTime();
			}
			int start = GetSession().Asset.GetFrame(Mathf.Max(0f, startTime)).Index;
			int end = GetSession().Asset.GetFrame(Mathf.Min(GetSession().Asset.GetTotalTime(), endTime)).Index;
			int elements = end-start+1;
			return new Vector3Int(start, end, elements);
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

		public void DrawWindow(Frame frame, float window, Color color, Rect rect) {
			DrawRect(
				frame.Asset.GetFrame(frame.Timestamp-window/2f),
				frame.Asset.GetFrame(frame.Timestamp+window/2f),
				1f,
				color,
				rect
			);
		}

		public void DrawPivot(Rect rect) {
			MotionAsset asset = GetSession().Asset;
			Frame frame = GetCurrentFrame();
			DrawRect(
				asset.GetFrame(Mathf.Clamp(frame.Timestamp - PastWindow, 0f, asset.GetTotalTime())),
				asset.GetFrame(Mathf.Clamp(frame.Timestamp + FutureWindow, 0f, asset.GetTotalTime())),
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

		public void DrawFunction(float[] values, float min, float max, float height=50f, string label=null) {
			MotionAsset asset = GetSession().Asset;
            Frame frame = GetCurrentFrame();

			Utility.SetGUIColor(UltiDraw.White);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				if(label != null) {
					EditorGUILayout.LabelField(label);
				}

				UltiDraw.Begin();

				EditorGUILayout.BeginVertical(GUILayout.Height(height));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = frame.Timestamp-GetWindow()/2f;
				float endTime = frame.Timestamp+GetWindow()/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > asset.GetTotalTime()) {
					startTime -= endTime-asset.GetTotalTime();
					endTime = asset.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(asset.GetTotalTime(), endTime);
				int start = asset.GetFrame(startTime).Index;
				int end = asset.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

				start = Mathf.Clamp(start, 1, asset.Frames.Length);
				end = Mathf.Clamp(end, 1, asset.Frames.Length);

				for(int i=start; i<=end; i++) {
					int prev = i;
					int next = i+1;
					float prevValue = Mathf.Clamp(values[Mathf.Clamp(prev, start, end)-1], min, max).Normalize(min, max, 0f, 1f);
					float nextValue = Mathf.Clamp(values[Mathf.Clamp(next, start, end)-1], min, max).Normalize(min, max, 0f, 1f);
					float _start = (float)(Mathf.Clamp(prev, start, end)-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(next, start, end)-start) / (float)elements;
					float xStart = rect.x + _start * rect.width;
					float xEnd = rect.x + _end * rect.width;
					float yStart = rect.y + (1f - prevValue) * rect.height;
					float yEnd = rect.y + (1f - nextValue) * rect.height;
					UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), Color.white);
				}

				//Current Pivot
				top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				top.y = rect.yMax - rect.height;
				bottom.y = rect.yMax;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...

				EditorGUILayout.EndVertical();
				UltiDraw.End();
			}
		}

		[CustomEditor(typeof(MotionEditor))]
		public class MotionEditor_Editor : Editor {

			public MotionEditor Target;

			private float SceneViewRepaintHz = 30f;
			private DateTime SceneViewTimestamp;

			private float InspectorRepaintHz = 10f;
			private DateTime InspectorTimestamp;

			public int Index = -1;
			public string[] Assets = new string[0];
			public string[] Enums = new string[0];

			public string Filter = string.Empty;

			void Awake() {
				Target = (MotionEditor)target;
				ApplyFilter();
				Refresh();
				SceneViewTimestamp = Utility.GetTimestamp();
				InspectorTimestamp = Utility.GetTimestamp();
				EditorApplication.update += EditorUpdate;
			}

			void OnDestroy() {
				EditorApplication.update -= EditorUpdate;
			}

			public void EditorUpdate() {
				if(Utility.GetElapsedTime(SceneViewTimestamp) >= 1f/SceneViewRepaintHz) {
					SceneView.RepaintAll();
					SceneViewTimestamp = Utility.GetTimestamp();
				}
				if(Utility.GetElapsedTime(InspectorTimestamp) >= 1f/InspectorRepaintHz) {
					Repaint();
					InspectorTimestamp = Utility.GetTimestamp();
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
				if(Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.Z) {
					Target.GetSession().Asset.Sequences.First().SetStart(Target.GetCurrentFrame().Index);
					Target.GetSession().Asset.MarkDirty(true, false);
				}
				if(Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.C) {
					Target.GetSession().Asset.Sequences.First().SetEnd(Target.GetCurrentFrame().Index);
					Target.GetSession().Asset.MarkDirty(true, false);
				}
			}

			public void Refresh() {
				if(Target.Directories.Count == 0) {
					Target.Directories.Add(new AssetDirectory());
				}
				foreach(AssetDirectory directory in Target.Directories) {
					directory.Refresh();
				}
				if(Target.Session == null && Target.Assets.Contains(Target.Asset) && MotionAsset.Retrieve(Target.Asset) != null) {
					Target.LoadSession(Target.Asset);
				}
				UpdateAssetIndex();
			}

			public void LoadPreviousAsset() {
				if(Index > 0) {
					Index = Mathf.Max(Index-1, 0);
					Target.LoadSession(Assets[Index]);
				}
			}

			public void LoadNextAsset() {
				if(Index < Assets.Length-1) {
					Index = Mathf.Min(Index+1, Assets.Length-1);
					Target.LoadSession(Assets[Index]);
				}
			}

			public void ApplyFilter() {
				List<string> assets = new List<string>();
				List<string> enums = new List<string>();
				for(int i=0; i<Target.Assets.Count; i++) {
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

			private void UpdateAssetIndex() {
				if(Target.GetSession() == null) {
					Index = -1;
				} else {
					if(Index == -1 || Index >= Assets.Length || Assets[Index] != Target.Asset) {
						Index = Assets.FindIndex(Target.Asset);
					}
				}
			}

			public void Inspector() {
				EditorGUI.BeginChangeCheck();

				//EDITOR MANAGER
				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

						Utility.SetGUIColor(UltiDraw.White);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Editor Manager");
						}

						Utility.SetGUIColor(UltiDraw.DarkGrey);
						using(new EditorGUILayout.VerticalScope("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("Select All", UltiDraw.DarkGrey, UltiDraw.White)) {
								foreach(AssetDirectory directory in Target.Directories) {
									directory.Import = true;
								}
							}
							if(Utility.GUIButton("Deselect All", UltiDraw.DarkGrey, UltiDraw.White)) {
								foreach(AssetDirectory directory in Target.Directories) {
									directory.Import = false;
								}
							}
							EditorGUILayout.EndHorizontal();
							foreach(AssetDirectory directory in Target.Directories) {
								EditorGUILayout.BeginHorizontal();
								if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White, 20f, 20f)) {
									Target.Directories.Add(new AssetDirectory());
								}
								Utility.SetGUIColor(directory.IsValid() ? (directory.Import ? UltiDraw.DarkGreen : UltiDraw.Gold) : UltiDraw.DarkRed);
								directory.SetFolder(EditorGUILayout.TextField(directory.Path));
								directory.Import = EditorGUILayout.Toggle(directory.Import, GUILayout.Width(20f));
								EditorGUI.BeginDisabledGroup(true);
								EditorGUILayout.IntField(directory.Size, GUILayout.Width(50f));
								EditorGUI.EndDisabledGroup();
								Utility.ResetGUIColor();
								if(Utility.GUIButton("-", UltiDraw.DarkGrey, UltiDraw.White, 20f, 20f)) {
									Target.Directories.Remove(directory);
									EditorGUIUtility.ExitGUI();
								}
								EditorGUILayout.EndHorizontal();
							}
						}

						if(Utility.GUIButton("Import", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.Import();
							ApplyFilter();
						}						

						Utility.SetGUIColor(UltiDraw.DarkGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							if(Utility.GUIButton("Settings", Target.Settings ? UltiDraw.Cyan : UltiDraw.DarkGrey, Target.Settings ? UltiDraw.Black : UltiDraw.White)) {
								Target.Settings = !Target.Settings;
							}
							if(Target.Settings) {
								Utility.SetGUIColor(UltiDraw.LightGrey);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();

									Target.SetTargetFramerate(EditorGUILayout.FloatField("Target Framerate", Target.TargetFramerate));

									Target.Timescale = EditorGUILayout.FloatField("Timescale", Target.Timescale);

									Utility.SetGUIColor(UltiDraw.White);
									using(new EditorGUILayout.VerticalScope ("Box")) {
										Utility.ResetGUIColor();
										EditorGUILayout.LabelField("Time Series");
										Target.PastKeys = Mathf.Max(EditorGUILayout.IntField("Past Keys", Target.PastKeys), 0);
										Target.FutureKeys = Mathf.Max(EditorGUILayout.IntField("Future Keys", Target.FutureKeys), 0);
										Target.PastWindow = EditorGUILayout.FloatField("Past Window", Target.PastWindow);
										Target.FutureWindow = EditorGUILayout.FloatField("Future Window", Target.FutureWindow);
										Target.Resolution = Mathf.Max(EditorGUILayout.IntField("Resolution", Target.Resolution), 1);
										if(Utility.GUIButton("Print Time Series", UltiDraw.DarkGrey, UltiDraw.White)) {
											Debug.Log("Delta Time: " + Target.GetTimeSeries().DeltaTime);
											foreach(TimeSeries.Sample sample in Target.GetTimeSeries().Samples) {
												Debug.Log("Sample " + sample.Index + " / " + sample.Timestamp + " / " + Target.GetSession().Asset.GetFrame(Target.GetTimestamp() + sample.Timestamp).Index);
											}
										}
									}
									Target.ShowCharacter = EditorGUILayout.Toggle("Show Character", Target.ShowCharacter);
									Target.ShowInRootSpace = EditorGUILayout.Toggle("Show In Root Space", Target.ShowInRootSpace);
									Target.PreviewMotion = EditorGUILayout.Toggle("Preview Motion", Target.PreviewMotion);
									Target.PastStep = EditorGUILayout.IntField("Past Step", Target.PastStep);
									Target.FutureStep = EditorGUILayout.IntField("Future Step", Target.FutureStep);
									Target.PreviewMesh = EditorGUILayout.Toggle("Preview Mesh", Target.PreviewMesh);
									Target.UpdateMesh = EditorGUILayout.Toggle("Update Mesh", Target.UpdateMesh);
									Target.MeshTransparency = EditorGUILayout.FloatField("Mesh Transparency", Target.MeshTransparency);
									Target.PreviewSpacing = EditorGUILayout.Vector3Field("Preview Spacing", Target.PreviewSpacing);
									Target.CameraFocus = EditorGUILayout.Toggle("Camera Focus", Target.CameraFocus);
									Target.FocusTransform = EditorGUILayout.ObjectField("Focus Transform", Target.FocusTransform, typeof(Transform), true) as Transform;
									Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
									Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
									Target.FocusAngle.y = EditorGUILayout.Slider("Focus Angle Horizontal", Target.FocusAngle.y, -180f, 180f);
									Target.FocusAngle.x = EditorGUILayout.Slider("Focus Angle Vertical", Target.FocusAngle.x, -180f, 180f);
									Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
									if(Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
										Target.SetMirror(!Target.Mirror);
									}
								}
							}
						}
					}
				}

				//ASSET SECTION
				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

						Utility.SetGUIColor(UltiDraw.White);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor(); 
							EditorGUILayout.LabelField("Asset Inspector");
						}

						Utility.SetGUIColor(UltiDraw.DarkGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();

							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								EditorGUILayout.BeginHorizontal();
								EditorGUILayout.HelpBox("Assets: " + Assets.Length  + " / " + Target.Assets.Count, MessageType.None);
								EditorGUI.BeginChangeCheck();
								Filter = EditorGUILayout.TextField(Filter, GUILayout.Width(110f));
								if(EditorGUI.EndChangeCheck()) {
									ApplyFilter();
								}
								EditorGUILayout.EndHorizontal();

								//Selection Browser
								UpdateAssetIndex();
								EditorGUILayout.BeginHorizontal();
								EditorGUI.BeginChangeCheck();
								int selectIndex = EditorGUILayout.Popup(Index, Enums);
								if(EditorGUI.EndChangeCheck()) {
									if(selectIndex != -1) {
										Index = selectIndex;
										Target.LoadSession(Assets[Index]);
									}
								}
								if(Utility.GUIButton("C", UltiDraw.DarkGrey, UltiDraw.White, 25f)) {
									GUIUtility.systemCopyBuffer = Target.GetSession().Asset.name;
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
									int sliderIndex = EditorGUILayout.IntSlider(Index+1, 1, Assets.Length);
									if(EditorGUI.EndChangeCheck()) {
										Index = sliderIndex-1;
										Target.LoadSession(Assets[Index]);
									}
								}
								EditorGUILayout.LabelField("/ " + Assets.Length, GUILayout.Width(55f));
								EditorGUILayout.EndHorizontal();								
							}
						}

						if(Target.IsSetup()) {
							Frame frame = Target.GetCurrentFrame();

							Utility.SetGUIColor(UltiDraw.DarkGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();

								EditorGUILayout.BeginVertical(GUILayout.Height(25f));
								Rect ctrl = EditorGUILayout.GetControlRect();
								Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 25f);
								EditorGUI.DrawRect(rect, UltiDraw.Black);

								//Sequences
								for(int i=0; i<Target.GetSession().Asset.Sequences.Length; i++) {
									Target.DrawRect(Target.GetSession().Asset.GetFrame(Target.GetSession().Asset.Sequences[i].Start), Target.GetSession().Asset.GetFrame(Target.GetSession().Asset.Sequences[i].End), 1f, !Target.GetSession().Asset.Export ? UltiDraw.White.Opacity(.25f) : UltiDraw.Green.Opacity(0.25f), rect);
								}

								//Current Pivot
								Target.DrawPivot(rect);

								EditorGUILayout.EndVertical();

								EditorGUILayout.BeginHorizontal();

								GUILayout.FlexibleSpace();

								if(Target.IsPlaying()) {
									if(Utility.GUIButton("||", Color.red, Color.black, 50f, 40f)) {
										Target.Play(false);
									}
								} else {
									if(Utility.GUIButton("|>", Color.green, Color.black, 50f, 40f)) {
										Target.Play(true);
									}
								}
								if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 40f)) {
									Target.LoadFrame(Mathf.Max(frame.Timestamp - Target.GetSession().Asset.GetDeltaTime(), 0f));
								}
								if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 40f)) {
									Target.LoadFrame(Mathf.Min(frame.Timestamp + Target.GetSession().Asset.GetDeltaTime(), Target.GetSession().Asset.GetTotalTime()));
								}

								EditorGUILayout.BeginVertical();
								int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.GetSession().Asset.GetTotalFrames());
								if(index != frame.Index) {
									Target.LoadFrame(Target.GetSession().Asset.GetFrame(index).Timestamp);
								}
								Target.Zoom = EditorGUILayout.Slider(Target.Zoom, 0f, 1f);
								EditorGUILayout.EndVertical();
								
								EditorGUILayout.BeginVertical();
								EditorGUILayout.LabelField("/ " + Target.GetSession().Asset.GetTotalFrames() + " @ " + Mathf.RoundToInt(Target.GetSession().Asset.Framerate) + "Hz", Utility.GetFontColor(Color.white), GUILayout.Width(80f));
								EditorGUILayout.LabelField("[" + frame.Timestamp.ToString("F2") + "s / " + Target.GetSession().Asset.GetTotalTime().ToString("F2") + "s]", Utility.GetFontColor(Color.white), GUILayout.Width(80f));
								EditorGUILayout.EndVertical();

								GUILayout.FlexibleSpace();

								EditorGUILayout.EndHorizontal();
							}

							Target.GetSession().Inspector();
						} else {
							EditorGUILayout.HelpBox("No asset selected.", MessageType.None);
						}

					}
				}

				if(EditorGUI.EndChangeCheck()) {
					Refresh();
				}
			}
		}
	}
}
#endif