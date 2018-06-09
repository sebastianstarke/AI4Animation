#if UNITY_EDITOR
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

[ExecuteInEditMode]
[UnityEditor.Callbacks.DidReloadScripts]
public class MotionEditor : MonoBehaviour {

	private MotionData[] Files = new MotionData[0];
	private Transform[] Environments = new Transform[0];

	private bool AutoFocus = false;
	private float FocusHeight = 1f;
	private float FocusOffset = 0f;
	private float FocusDistance = 2.5f;
	private float FocusAngle = 0f;
	private float FocusSmoothing = 0.05f;
	private bool Mirror = false;
	private bool Playing = false;
	private float Timescale = 1f;
	private float Timestamp = 0f;

	private bool ShowMotion = false;
	private bool ShowVelocities = false;
	private bool ShowTrajectory = false;
	private bool ShowHeightMap = false;
	private bool ShowDepthMap = false;
	private bool ShowDepthImage = false;

	private bool InspectFrame = true;
	private bool InspectExport = true;
	private bool InspectSettings = true;

	private Actor Actor = null;
	private Transform Scene = null;
	private MotionState State = null;

	private int ID = -1;
	
	public void VisualiseMotion(bool value) {
		ShowMotion = value;
	}
	public void VisualiseVelocities(bool value) {
		ShowVelocities = value;
	}
	public void VisualiseTrajectory(bool value) {
		ShowTrajectory = value;
	}
	public void VisualiseHeightMap(bool value) {
		ShowHeightMap = value;
	}
	public void VisualiseDepthMap(bool value) {
		ShowDepthMap = value;
	}
	public void VisualiseDepthImage(bool value) {
		ShowDepthImage = value;
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
			Actor = GameObject.FindObjectOfType<Actor>();
		}
		if(Actor == null) {
			return CreateSkeleton();
		} else {
 			return Actor;
		}
	}

	public Transform GetScene() {
		if(Scene == null) {
			return GameObject.Find("Scene").transform;
		}
		if(Scene == null) {
			return new GameObject("Scene").transform;
		} else {
			return Scene;
		}
	}

	public MotionData GetData() {
		if(Files.Length == 0) {
			ID = -1;
			return null;
		}
		LoadFile(Mathf.Clamp(ID, 0, Files.Length-1));
		return Files[ID];
	}

	public MotionState GetState() {
		if(State == null) {
			LoadFrame(Timestamp);
		}
		return State;
	}

	public void Refresh() {
		string folder = EditorSceneManager.GetActiveScene().path.Substring(0, EditorSceneManager.GetActiveScene().path.LastIndexOf("/"));
		string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{folder});
		//Files
		Files = new MotionData[assets.Length];
		for(int i=0; i<Files.Length; i++) {
			Files[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[i]), typeof(MotionData));
		}
		//Environments
		Transform container = GetScene().Find("Environments");
		if(container == null) {
			container = new GameObject("Environments").transform;
			container.SetParent(GetScene());
		}
		//Cleanup
		for(int i=0; i<container.childCount; i++) {
			if(!System.Array.Find(Files, x => x.Name == container.GetChild(i).name)) {
				Utility.Destroy(container.GetChild(i).gameObject);
				i--;
			}
		}
		//Fill
		Environments = new Transform[assets.Length];
		for(int i=0; i<Environments.Length; i++) {
			Environments[i] = container.Find(Files[i].Name);
			if(Environments[i] == null) {
				Environments[i] = new GameObject(Files[i].Name).transform;
				Environments[i].SetParent(container);
			}
		}
		//Finalise
		for(int i=0; i<Environments.Length; i++) {
			Environments[i].gameObject.SetActive(i == ID);
			Environments[i].SetSiblingIndex(i);
		}
		//Initialise
		if(GetData() != null) {
			LoadFrame(0f);
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
		if(ID != id) {
			Save(ID);
			ID = id;
			if(ID < 0) {
				return;
			}
			for(int i=0; i<Environments.Length; i++) {
				Environments[i].gameObject.SetActive(i == ID);
			}
			LoadFrame(0f);
		}
	}

	public void LoadFrame(MotionState state) {
		Timestamp = state.Timestamp;
		State = state;
		if(state.Mirrored) {
			GetScene().localScale = Vector3.one.GetMirror(GetData().GetAxis(GetData().MirrorAxis));
		} else {
			GetScene().localScale = Vector3.one;
		}

		GetActor().GetRoot().position = GetState().Root.GetPosition();
		GetActor().GetRoot().rotation = GetState().Root.GetRotation();
		for(int i=0; i<GetActor().Bones.Length; i++) {
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
				rotation = Quaternion.Euler(0f, Mirror ? Mathf.Repeat(FocusAngle + 0f, 360f) : FocusAngle, 0f) * rotation;
				position += FocusOffset * (rotation * Vector3.right);
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
			}
		}
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
		Actor = new GameObject("Skeleton").AddComponent<Actor>();
		string[] names = new string[GetData().Source.Bones.Length];
		string[] parents = new string[GetData().Source.Bones.Length];
		for(int i=0; i<GetData().Source.Bones.Length; i++) {
			names[i] = GetData().Source.Bones[i].Name;
			parents[i] = GetData().Source.Bones[i].Parent;
		}
		List<Transform> instances = new List<Transform>();
		for(int i=0; i<names.Length; i++) {
			Transform instance = new GameObject(names[i]).transform;
			instance.SetParent(parents[i] == "None" ? GetActor().GetRoot() : GetActor().FindTransform(parents[i]));
			instances.Add(instance);
		}
		GetActor().ExtractSkeleton(instances.ToArray());
		return Actor.GetComponent<Actor>();
	}

	public void Draw() {
		if(ShowMotion) {
			for(int i=0; i<GetState().PastBoneTransformations.Count; i++) {
				GetActor().DrawSimple(Color.Lerp(UltiDraw.Blue, UltiDraw.Cyan, 1f - (float)(i+1)/6f).Transparent(0.75f), GetState().PastBoneTransformations[i]);
			}
			for(int i=0; i<GetState().FutureBoneTransformations.Count; i++) {
				GetActor().DrawSimple(Color.Lerp(UltiDraw.Red, UltiDraw.Orange, (float)i/5f).Transparent(0.75f), GetState().FutureBoneTransformations[i]);
			}
		}

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
		
		if(ShowHeightMap) {
			GetState().HeightMap.Draw();
		}
		
		if(ShowDepthMap) {
			GetState().DepthMap.Draw();
		}

		UltiDraw.Begin();
		UltiDraw.DrawGUIRectangle(Vector2.one/2f, Vector2.one, UltiDraw.Mustard);
		UltiDraw.End();
		if(ShowDepthImage) {
			UltiDraw.Begin();
			Vector2 size = new Vector2(0.5f, 0.5f*Screen.width/Screen.height);
			for(int x=0; x<GetState().DepthMap.GetResolution(); x++) {
				for(int y=0; y<GetState().DepthMap.GetResolution(); y++) {
					float distance = Vector3.Distance(GetState().DepthMap.Points[GetState().DepthMap.GridToArray(x,y)], GetState().DepthMap.Pivot.GetPosition());
					float intensity = 1f - distance / GetState().DepthMap.GetDistance();
					UltiDraw.DrawGUIRectangle(Vector2.one/2f - size/2f + new Vector2((float)x*size.x, (float)y*size.y) / (GetState().DepthMap.GetResolution()-1), size / (GetState().DepthMap.GetResolution()-1), Color.Lerp(Color.black, Color.white, intensity));
				}
			}
			UltiDraw.End();
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

		private float RefreshRate = 30f;
		private System.DateTime Timestamp;

		void Awake() {
			Target = (MotionEditor)target;
			Target.Refresh();
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			if(!Application.isPlaying && Target != null) {
				Target.Save(Target.ID);
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

		public void Inspector() {
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Cyan);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						
						Utility.SetGUIColor(Target.GetActor() == null ? UltiDraw.DarkRed : UltiDraw.White);
						Target.Actor = (Actor)EditorGUILayout.ObjectField("Actor", Target.GetActor(), typeof(Actor), true);
						Utility.ResetGUIColor();

						Utility.SetGUIColor(Target.GetScene() == null ? UltiDraw.DarkRed : UltiDraw.White);
						EditorGUILayout.ObjectField("Scene", Target.GetScene(), typeof(Transform), true);
						Utility.ResetGUIColor();

						EditorGUILayout.BeginHorizontal();
						string[] names = new string[Target.Files.Length];
						if(names.Length == 0) {
							Target.LoadFile(EditorGUILayout.Popup("Data", -1, names));
						} else {
							for(int i=0; i<names.Length; i++) {
								names[i] = Target.Files[i].name;
							}
							Target.LoadFile(EditorGUILayout.Popup("Data", Target.ID, names));
						}
						/*
						if(GUILayout.Button("+", GUILayout.Width(18f))) {
							string path = EditorUtility.OpenFilePanel("Motion Editor", Application.dataPath, "bvh");
							GUI.SetNextControlName("");
							GUI.FocusControl("");
							Target.AddData(path);
							GUIUtility.ExitGUI();
						}
						if(GUILayout.Button("-", GUILayout.Width(18f))) {
							Target.RemoveData();
						}
						if(GUILayout.Button("R", GUILayout.Width(18f))) {
							Target.RefreshData();
						}
						*/
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

					if(Target.GetData() != null) {
						MotionData.Frame frame = Target.GetData().GetFrame(Target.Timestamp);

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

						if(Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.SetMirror(!Target.Mirror);
						}

						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Motion", Target.ShowMotion ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowMotion = !Target.ShowMotion;
						}
						if(Utility.GUIButton("Trajectory", Target.ShowTrajectory ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowTrajectory = !Target.ShowTrajectory;
						}
						if(Utility.GUIButton("Velocities", Target.ShowVelocities ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowVelocities = !Target.ShowVelocities;
						}
						if(Utility.GUIButton("Height Map", Target.ShowHeightMap ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowHeightMap = !Target.ShowHeightMap;
						}
						if(Utility.GUIButton("Depth Map", Target.ShowDepthMap ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowDepthMap = !Target.ShowDepthMap;
						}
						if(Utility.GUIButton("Depth Image", Target.ShowDepthImage ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
							Target.ShowDepthImage = !Target.ShowDepthImage;
						}
						EditorGUILayout.EndHorizontal();

						Utility.SetGUIColor(UltiDraw.Grey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();

							Utility.SetGUIColor(UltiDraw.Mustard);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								Target.InspectFrame = EditorGUILayout.Toggle("Frame", Target.InspectFrame);
							}

							if(Target.InspectFrame) {
								Color[] colors = UltiDraw.GetRainbowColors(Target.GetData().Styles.Length);
								for(int i=0; i<Target.GetData().Styles.Length; i++) {
									float height = 25f;
									EditorGUILayout.BeginHorizontal();
									if(Utility.GUIButton(Target.GetData().Styles[i], !frame.StyleFlags[i] ? colors[i].Transparent(0.25f) : colors[i], UltiDraw.White, 200f, height)) {
										frame.ToggleStyle(i);
									}
									Rect c = EditorGUILayout.GetControlRect();
									Rect r = new Rect(c.x, c.y, frame.StyleValues[i] * c.width, height);
									EditorGUI.DrawRect(r, colors[i].Transparent(0.75f));
									EditorGUILayout.FloatField(frame.StyleValues[i], GUILayout.Width(50f));
									EditorGUILayout.EndHorizontal();
								}
								EditorGUILayout.BeginHorizontal();
								if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
									MotionData.Frame previous = frame.GetAnyPreviousStyleKey();
									Target.LoadFrame(previous == null ? 0f : previous.Timestamp);
								}
								EditorGUILayout.BeginVertical(GUILayout.Height(50f));
								Rect ctrl = EditorGUILayout.GetControlRect();
								Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
								EditorGUI.DrawRect(rect, UltiDraw.Black);
								UltiDraw.Begin();
								//Sequences
								for(int i=0; i<Target.GetData().Sequences.Length; i++) {
									float start = rect.x + (float)(Target.GetData().Sequences[i].Start-1)/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
									float end = rect.x + (float)(Target.GetData().Sequences[i].End-1)/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
									Vector3 a = new Vector3(start, rect.y, 0f);
									Vector3 b = new Vector3(end, rect.y, 0f);
									Vector3 c = new Vector3(start, rect.y+rect.height, 0f);
									Vector3 d = new Vector3(end, rect.y+rect.height, 0f);
									UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
									UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
								}
								//Styles
								for(int i=0; i<Target.GetData().Styles.Length; i++) {
									int x = 0;
									for(int j=1; j<Target.GetData().GetTotalFrames(); j++) {
										float val = Target.GetData().Frames[j].StyleValues[i];
										if(
											Target.GetData().Frames[x].StyleValues[i]<1f && val==1f ||
											Target.GetData().Frames[x].StyleValues[i]>0f && val==0f
											) {
											float xStart = rect.x + (float)(Mathf.Max(x-1, 0))/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
											float xEnd = rect.x + (float)j/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
											float yStart = rect.y + (1f - Target.GetData().Frames[Mathf.Max(x-1, 0)].StyleValues[i]) * rect.height;
											float yEnd = rect.y + (1f - Target.GetData().Frames[j].StyleValues[i]) * rect.height;
											UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
											x = j;
										}
										if(
											Target.GetData().Frames[x].StyleValues[i]==0f && val>0f || 
											Target.GetData().Frames[x].StyleValues[i]==1f && val<1f
											) {
											float xStart = rect.x + (float)(x)/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
											float xEnd = rect.x + (float)(j-1)/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
											float yStart = rect.y + (1f - Target.GetData().Frames[x].StyleValues[i]) * rect.height;
											float yEnd = rect.y + (1f - Target.GetData().Frames[j-1].StyleValues[i]) * rect.height;
											UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
											x = j;
										}
										if(j==Target.GetData().GetTotalFrames()-1) {
											float xStart = rect.x + (float)x/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
											float xEnd = rect.x + (float)(j-1)/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
											float yStart = rect.y + (1f - Target.GetData().Frames[x].StyleValues[i]) * rect.height;
											float yEnd = rect.y + (1f - Target.GetData().Frames[j-1].StyleValues[i]) * rect.height;
											UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
											x = j;
										}
									}
								}
								float pivot = rect.x + (float)(frame.Index-1)/(float)(Target.GetData().GetTotalFrames()-1) * rect.width;
								UltiDraw.DrawLine(new Vector3(pivot, rect.y, 0f), new Vector3(pivot, rect.y + rect.height, 0f), UltiDraw.White);
								UltiDraw.DrawWireCircle(new Vector3(pivot, rect.y, 0f), 8f, UltiDraw.Green);
								UltiDraw.DrawWireCircle(new Vector3(pivot, rect.y + rect.height, 0f), 8f, UltiDraw.Green);
								UltiDraw.End();
								EditorGUILayout.EndVertical();
								if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
									MotionData.Frame next = frame.GetAnyNextStyleKey();
									Target.LoadFrame(next == null ? Target.GetData().GetTotalTime() : next.Timestamp);
								}
								EditorGUILayout.EndHorizontal();
							}

							Utility.SetGUIColor(UltiDraw.Mustard);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								Target.InspectExport = EditorGUILayout.Toggle("Export", Target.InspectExport);
							}

							if(Target.InspectExport) {
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
										//for(int c=0; c<Target.GetData().Sequences[i].Copies.Length; c++) {
										//	EditorGUILayout.LabelField("Copy " + (c+1) + " - " + "Start: " + Target.GetData().Sequences[i].Copies[c].Start + " End: " + Target.GetData().Sequences[i].Copies[c].End);
										//}
									}
								}
								EditorGUILayout.BeginHorizontal();
								if(Utility.GUIButton("Add", UltiDraw.DarkGrey, UltiDraw.White)) {
									Target.GetData().AddSequence(1, Target.GetData().GetTotalFrames());
								}
								if(Utility.GUIButton("Remove", UltiDraw.DarkGrey, UltiDraw.White)) {
									Target.GetData().RemoveSequence();
								}
								EditorGUILayout.EndHorizontal();
							}

							Utility.SetGUIColor(UltiDraw.Mustard);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								Target.InspectSettings = EditorGUILayout.Toggle("Settings", Target.InspectSettings);
							}

							if(Target.InspectSettings) {
								string[] presets = new string[4] {"Select preset...", "Dan", "Dog", "Interaction"};
								switch(EditorGUILayout.Popup(0, presets)) {
									case 0:
									break;
									case 1:
									Target.GetData().DepthMapAxis = MotionData.Axis.ZPositive;
									Target.GetData().MirrorAxis = MotionData.Axis.XPositive;
									for(int i=0; i<Target.GetData().Corrections.Length; i++) {
										Target.GetData().SetCorrection(i, Vector3.zero);
									}
									Target.GetData().ClearStyles();
									Target.GetData().AddStyle("Idle");
									Target.GetData().AddStyle("Walk");
									Target.GetData().AddStyle("Run");
									Target.GetData().AddStyle("Jump");
									Target.GetData().AddStyle("Crouch");
									break;

									case 2:
									Target.GetData().DepthMapAxis = MotionData.Axis.XPositive;
									Target.GetData().MirrorAxis = MotionData.Axis.ZPositive;
									for(int i=0; i<Target.GetData().Corrections.Length; i++) {
										if(i==4 || i==5 || i==6 || i==11) {
											Target.GetData().SetCorrection(i, new Vector3(90f, 90f, 90f));
										} else if(i==24) {
											Target.GetData().SetCorrection(i, new Vector3(-45f, 0f, 0f));
										} else {
											Target.GetData().SetCorrection(i, new Vector3(0f, 0f, 0f));
										}
									}
									Target.GetData().ClearStyles();
									Target.GetData().AddStyle("Idle");
									Target.GetData().AddStyle("Walk");
									Target.GetData().AddStyle("Pace");
									Target.GetData().AddStyle("Trot");
									Target.GetData().AddStyle("Canter");
									Target.GetData().AddStyle("Jump");
									Target.GetData().AddStyle("Sit");
									Target.GetData().AddStyle("Stand");
									Target.GetData().AddStyle("Lie");
									break;

									case 3:
									Target.GetData().DepthMapAxis = MotionData.Axis.ZPositive;
									Target.GetData().MirrorAxis = MotionData.Axis.XPositive;							
									for(int i=0; i<Target.GetData().Corrections.Length; i++) {
										Target.GetData().SetCorrection(i, Vector3.zero);
									}
									Target.GetData().ClearStyles();
									Target.GetData().AddStyle("Idle");
									Target.GetData().AddStyle("Walk");
									Target.GetData().AddStyle("Run");
									Target.GetData().AddStyle("Jump");
									Target.GetData().AddStyle("Crouch");
									Target.GetData().AddStyle("Sit");
									break;
								}

								Utility.SetGUIColor(UltiDraw.LightGrey);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									if(Utility.GUIButton("Create Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
										Target.CreateSkeleton();
									}
									if(Utility.GUIButton("Refresh Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
										Utility.Destroy(Target.GetActor().gameObject);
										Target.CreateSkeleton();
									}
								}

								Utility.SetGUIColor(UltiDraw.LightGrey);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									EditorGUILayout.LabelField("General");

									Target.GetData().Scaling = EditorGUILayout.FloatField("Scaling", Target.GetData().Scaling);
									Target.GetData().RootSmoothing = EditorGUILayout.IntField("Root Smoothing", Target.GetData().RootSmoothing);
									
									Target.GetData().GroundMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.GetData().GroundMask), InternalEditorUtility.layers));
									Target.GetData().ObjectMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Object Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.GetData().ObjectMask), InternalEditorUtility.layers));
									
									string[] names = new string[Target.GetData().Source.Bones.Length];
									for(int i=0; i<Target.GetData().Source.Bones.Length; i++) {
										names[i] = Target.GetData().Source.Bones[i].Name;
									}
									Target.GetData().HeightMapSensor = EditorGUILayout.Popup("Height Map Sensor", Target.GetData().HeightMapSensor, names);
									Target.GetData().HeightMapSize = EditorGUILayout.Slider("Height Map Size", Target.GetData().HeightMapSize, 0f, 1f);
									Target.GetData().DepthMapSensor = EditorGUILayout.Popup("Depth Map Sensor", Target.GetData().DepthMapSensor, names);
									Target.GetData().DepthMapAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Depth Map Axis", Target.GetData().DepthMapAxis);
									Target.GetData().DepthMapResolution = EditorGUILayout.IntField("Depth Map Resolution", Target.GetData().DepthMapResolution);
									Target.GetData().DepthMapSize = EditorGUILayout.FloatField("Depth Map Size", Target.GetData().DepthMapSize);
									Target.GetData().DepthMapDistance = EditorGUILayout.FloatField("Depth Map Distance", Target.GetData().DepthMapDistance);
									
									Target.GetData().SetStyleTransition(EditorGUILayout.Slider("Style Transition", Target.GetData().StyleTransition, 0.1f, 1f));
									for(int i=0; i<Target.GetData().Styles.Length; i++) {
										EditorGUILayout.BeginHorizontal();
										Target.GetData().Styles[i] = EditorGUILayout.TextField("Style " + (i+1), Target.GetData().Styles[i]);
										EditorGUILayout.EndHorizontal();
									}
									EditorGUILayout.BeginHorizontal();
									if(Utility.GUIButton("Add Style", UltiDraw.DarkGrey, UltiDraw.White)) {
										Target.GetData().AddStyle("Style");
									}
									if(Utility.GUIButton("Remove Style", UltiDraw.DarkGrey, UltiDraw.White)) {
										Target.GetData().RemoveStyle();
									}
									EditorGUILayout.EndHorizontal();
								}

								Utility.SetGUIColor(UltiDraw.LightGrey);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									EditorGUILayout.LabelField("Geometry");
									if(Utility.GUIButton("Detect Symmetry", UltiDraw.DarkGrey, UltiDraw.White)) {
										Target.GetData().DetectSymmetry();
									}
									Target.GetData().MirrorAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Mirror Axis", Target.GetData().MirrorAxis);
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
										Target.GetData().SetCorrection(i, EditorGUILayout.Vector3Field("", Target.GetData().Corrections[i]));
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

}
#endif
