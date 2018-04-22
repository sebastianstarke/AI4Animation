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

	public string Path = string.Empty;

	public Actor Actor;
	public Transform Scene;
	
	public MotionData Data = null;

	public bool AutoFocus = true;
	public float FocusDistance = 2.5f;
	public float FocusAngle = 180f;
	public float FocusSmoothing = 0.5f;

	public float Timestamp = 0f;
	public bool Playing = false;
	public float Timescale = 1f;
	public bool Mirror = false;

	public bool ShowMotion = false;
	public bool ShowVelocities = false;
	public bool ShowTrajectory = true;
	public bool ShowHeightMap = false;
	public bool ShowDepthMap = false;
	public bool ShowDepthImage = false;

	private FrameState State;

	public void SetActor(Actor actor) {
		if(Actor != actor) {
			Actor = actor;
			CheckActor();
		}
	}

	public void SetScene(Transform scene) {
		Scene = scene;
	}

	public FrameState GetState() {
		return State;
	}

	public void LoadFile() {
		if(!File.Exists(Path)) {
			Debug.Log("File at path " + Path + " does not exist.");
			return;
		}
		Data = ScriptableObject.CreateInstance<MotionData>();
		Data.Load(Path);
		Timestamp = 0f;
		StopAnimation();
		Timescale = 1f;
		Mirror = false;
		ShowMotion = false;
		ShowVelocities = false;
		ShowTrajectory = false;
		State = null;
		CheckActor();
		AssetDatabase.RenameAsset(UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, Path.Substring(Path.LastIndexOf("/")+1));
	}

	public void UnloadFile() {
		Data = null;
		Timestamp = 0f;
		StopAnimation();
		Timescale = 1f;
		Mirror = false;
		ShowMotion = false;
		ShowVelocities = false;
		ShowTrajectory = false;
		State = null;
		CheckActor();
		AssetDatabase.RenameAsset(UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, "None");
	}

	public void LoadFrame(float timestamp) {
		CheckActor();
		CheckScene();
		Timestamp = timestamp;
		State = new FrameState(Data.GetFrame(Timestamp), Mirror);
		Actor.GetRoot().position = State.Root.GetPosition();
		Actor.GetRoot().rotation = State.Root.GetRotation();
		for(int i=0; i<Actor.Bones.Length; i++) {
			Actor.Bones[i].Transform.position = State.BoneTransformations[i].GetPosition();
			Actor.Bones[i].Transform.rotation = State.BoneTransformations[i].GetRotation();
		}
	}

	public void LoadFrame(int index) {
		LoadFrame(Data.GetFrame(index).Timestamp);
	}

	public void LoadPreviousFrame() {
		LoadFrame(Mathf.Max(Data.GetFrame(Timestamp).Index - 1, 1));
	}

	public void LoadNextFrame() {
		LoadFrame(Mathf.Min(Data.GetFrame(Timestamp).Index + 1, Data.GetTotalFrames()));
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
		while(true) {
			Timestamp += Timescale * (float)Utility.GetElapsedTime(timestamp);
			if(Timestamp > Data.GetTotalTime()) {
				Timestamp = Mathf.Repeat(Timestamp, Data.GetTotalTime());
			}
			timestamp = Utility.GetTimestamp();
			LoadFrame(Timestamp);
			yield return new WaitForSeconds(0f);
		}
	}

	public void CheckActor() {
		if(Data == null) {
			if(Actor != null) {
				if(Actor.transform.parent == transform) {
					Utility.Destroy(Actor.gameObject);
					return;
				}
			}
		}
		if(Actor == null) {
			Transform actor = transform.Find("Skeleton");
			if(actor != null) {
				Actor = actor.GetComponent<Actor>();
			} else {
				Actor = new GameObject("Skeleton").AddComponent<Actor>();
				Actor.transform.SetParent(transform);
				string[] names = new string[Data.Source.Bones.Length];
				string[] parents = new string[Data.Source.Bones.Length];
				for(int i=0; i<Data.Source.Bones.Length; i++) {
					names[i] = Data.Source.Bones[i].Name;
					parents[i] = Data.Source.Bones[i].Parent;
				}
				List<Transform> instances = new List<Transform>();
				for(int i=0; i<names.Length; i++) {
					Transform instance = new GameObject(names[i]).transform;
					instance.SetParent(parents[i] == "None" ? Actor.GetRoot() : Actor.FindTransform(parents[i]));
					instances.Add(instance);
				}
				Actor.ExtractSkeleton(instances.ToArray());
			}
		} else {
			if(Actor.transform.parent != transform) {
				Transform actor = transform.Find("Skeleton");
				if(actor != null) {
					Utility.Destroy(actor.gameObject);
				}
			}
		}
	}

	public void CheckScene() {
		if(Scene == null) {
			return;
		}
		if(Mirror) {
			Scene.localScale = Vector3.one.GetMirror(Data.GetAxis(Data.MirrorAxis));
		} else {
			Scene.localScale = Vector3.one;
		}
	}

	public void Draw() {
		if(State == null) {
			return;
		}
		
		if(ShowMotion) {
			for(int i=0; i<6; i++) {
				MotionData.Frame previous = Data.GetFrame(Mathf.Clamp(State.Timestamp - 1f + (float)i/6f, 0f, Data.GetTotalTime()));
				Actor.DrawSimple(Color.Lerp(UltiDraw.Blue, UltiDraw.Cyan, 1f - (float)(i+1)/6f).Transparent(0.75f), previous.GetBoneTransformations(Mirror));
			}
			for(int i=1; i<=5; i++) {
				MotionData.Frame future = Data.GetFrame(Mathf.Clamp(State.Timestamp + (float)i/5f, 0f, Data.GetTotalTime()));
				Actor.DrawSimple(Color.Lerp(UltiDraw.Red, UltiDraw.Orange, (float)(i+1)/5f).Transparent(0.75f), future.GetBoneTransformations(Mirror));

			}
		}
		if(ShowVelocities) {
			UltiDraw.Begin();
			for(int i=0; i<Actor.Bones.Length; i++) {
				UltiDraw.DrawArrow(
					Actor.Bones[i].Transform.position,
					Actor.Bones[i].Transform.position + State.BoneVelocities[i],
					0.75f,
					0.0075f,
					0.05f,
					UltiDraw.Purple.Transparent(0.5f)
				);
			}
			UltiDraw.End();
		}
		if(ShowTrajectory) {
			State.Trajectory.Draw();
		}
		
		if(ShowHeightMap) {
			State.HeightMap.Draw();
		}

		if(ShowDepthMap) {
			State.DepthMap.Draw();
		}
		
		if(ShowDepthImage) {
			UltiDraw.Begin();
			Vector2 position = new Vector2(0.5f, 0.5f);
			Vector2 size = new Vector2(0.5f, 0.5f*Screen.width/Screen.height);
			UltiDraw.DrawGUIRectangle(position, Vector2.one, UltiDraw.Brown);
			for(int x=0; x<State.DepthMap.GetResolution(); x++) {
				for(int y=0; y<State.DepthMap.GetResolution(); y++) {
					float distance = Vector3.Distance(State.DepthMap.Points[State.DepthMap.GridToArray(x,y)], State.DepthMap.Pivot.GetPosition());
					float intensity = 1f - distance / State.DepthMap.GetDistance();
					//intensity = Utility.TanH(intensity);
					UltiDraw.DrawGUIRectangle(position - size/2f + new Vector2((float)x*size.x, (float)y*size.y) / (State.DepthMap.GetResolution()-1), size / (State.DepthMap.GetResolution()-1), Color.Lerp(Color.black, Color.white, intensity));
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

	public class FrameState {
		public int Index;
		public float Timestamp;
		public Matrix4x4 Root;
		public Vector3 RootMotion;
		public Matrix4x4[] BoneTransformations;
		public Vector3[] BoneVelocities;
		public Trajectory Trajectory;
		public HeightMap HeightMap;
		public DepthMap DepthMap;
		public FrameState(MotionData.Frame frame, bool mirrored) {
			Index = frame.Index;
			Timestamp = frame.Timestamp;
			Root = frame.GetRoot(mirrored);
			RootMotion = frame.GetRootMotion(mirrored);
			BoneTransformations = frame.GetBoneTransformations(mirrored);
			BoneVelocities = frame.GetBoneVelocities(mirrored);
			Trajectory = frame.GetTrajectory(mirrored);
			HeightMap = frame.GetHeightMap(mirrored);
			DepthMap = frame.GetDepthMap(mirrored);
		}
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		void Awake() {
			Target = (MotionEditor)target;
		}

		void OnDestroy() {
			if(!Application.isPlaying && Target != null) {
				EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
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
			InspectImporter();
			InspectEditor();
		}

		private void InspectImporter() {
			if(Target.Data != null) {
				return;
			}
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					using(new EditorGUILayout.VerticalScope ("Box")) {
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Path", GUILayout.Width(50));
						Target.Path = EditorGUILayout.TextField(Target.Path);
						GUI.skin.button.alignment = TextAnchor.MiddleCenter;
						if(GUILayout.Button("O", GUILayout.Width(20))) {
							Target.Path = EditorUtility.OpenFilePanel("Motion Editor", Target.Path == string.Empty ? Application.dataPath : Target.Path.Substring(0, Target.Path.LastIndexOf("/")), "bvh");
							GUI.SetNextControlName("");
							GUI.FocusControl("");
						}
						EditorGUILayout.EndHorizontal();
					}
					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.LoadFile();
						}
					}
				}
			}
		}

		private void InspectEditor() {
			if(Target.Data == null) {
				return;
			}

			MotionData.Frame frame = Target.Data.GetFrame(Target.Timestamp);

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField("Name: " + Target.Data.Name);
						Target.SetActor((Actor)EditorGUILayout.ObjectField("Actor", Target.Actor, typeof(Actor), true));
						Target.SetScene((Transform)EditorGUILayout.ObjectField("Scene", Target.Scene, typeof(Transform), true));
					}

					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						GUILayout.FlexibleSpace();
						EditorGUILayout.LabelField("Frames: " + Target.Data.GetTotalFrames(), GUILayout.Width(100f));
						EditorGUILayout.LabelField("Time: " + Target.Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
						EditorGUILayout.LabelField("Framerate: " + Target.Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
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
					int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.Data.GetTotalFrames(), GUILayout.Width(440f));
					if(index != frame.Index) {
						Target.LoadFrame(index);
					}
					EditorGUILayout.LabelField(frame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
					GUILayout.FlexibleSpace();
					EditorGUILayout.EndHorizontal();

					if(Utility.GUIButton("Mirror", Target.Mirror ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
						Target.Mirror = !Target.Mirror;
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
							EditorGUILayout.LabelField("Frame");
						}

						Color[] colors = UltiDraw.GetRainbowColors(Target.Data.Styles.Length);
						for(int i=0; i<Target.Data.Styles.Length; i++) {
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton(Target.Data.Styles[i], !frame.StyleFlags[i] ? colors[i].Transparent(0.25f) : colors[i], UltiDraw.White)) {
								frame.ToggleStyle(i);
							}
							EditorGUI.BeginDisabledGroup(true);
							EditorGUILayout.Slider(frame.StyleValues[i], 0f, 1f);
							EditorGUI.EndDisabledGroup();
							EditorGUILayout.EndHorizontal();
						}
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
							MotionData.Frame previous = frame.GetAnyPreviousStyleKey();
							Target.Timestamp = previous == null ? 0f : previous.Timestamp;
						}
						EditorGUILayout.BeginVertical(GUILayout.Height(50f));
						Rect ctrl = EditorGUILayout.GetControlRect();
						Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
						EditorGUI.DrawRect(rect, UltiDraw.Black);
						UltiDraw.Begin();
						//Sequences
						for(int i=0; i<Target.Data.Sequences.Length; i++) {
							float start = rect.x + (float)(Target.Data.Sequences[i].Start-1)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
							float end = rect.x + (float)(Target.Data.Sequences[i].End-1)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
							Vector3 a = new Vector3(start, rect.y, 0f);
							Vector3 b = new Vector3(end, rect.y, 0f);
							Vector3 c = new Vector3(start, rect.y+rect.height, 0f);
							Vector3 d = new Vector3(end, rect.y+rect.height, 0f);
							UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
							UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
						}
						//Styles
						for(int i=0; i<Target.Data.Styles.Length; i++) {
							int x = 0;
							for(int j=1; j<Target.Data.GetTotalFrames(); j++) {
								float val = Target.Data.Frames[j].StyleValues[i];
								if(
									Target.Data.Frames[x].StyleValues[i]<1f && val==1f ||
									Target.Data.Frames[x].StyleValues[i]>0f && val==0f
									) {
									float xStart = rect.x + (float)(x-1)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
									float xEnd = rect.x + (float)j/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
									float yStart = rect.y + (1f - Target.Data.Frames[x-1].StyleValues[i]) * rect.height;
									float yEnd = rect.y + (1f - Target.Data.Frames[j].StyleValues[i]) * rect.height;
									UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
									x = j;
								}
								if(
									Target.Data.Frames[x].StyleValues[i]==0f && val>0f || 
									Target.Data.Frames[x].StyleValues[i]==1f && val<1f
									) {
									float xStart = rect.x + (float)(x)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
									float xEnd = rect.x + (float)(j-1)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
									float yStart = rect.y + (1f - Target.Data.Frames[x].StyleValues[i]) * rect.height;
									float yEnd = rect.y + (1f - Target.Data.Frames[j-1].StyleValues[i]) * rect.height;
									UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
									x = j;
								}
								if(j==Target.Data.GetTotalFrames()-1) {
									float xStart = rect.x + (float)x/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
									float xEnd = rect.x + (float)(j-1)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
									float yStart = rect.y + (1f - Target.Data.Frames[x].StyleValues[i]) * rect.height;
									float yEnd = rect.y + (1f - Target.Data.Frames[j-1].StyleValues[i]) * rect.height;
									UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
									x = j;
								}
							}
						}
						float pivot = rect.x + (float)(frame.Index-1)/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
						UltiDraw.DrawLine(new Vector3(pivot, rect.y, 0f), new Vector3(pivot, rect.y + rect.height, 0f), UltiDraw.White);
						UltiDraw.DrawWireCircle(new Vector3(pivot, rect.y, 0f), 8f, UltiDraw.Green);
						UltiDraw.DrawWireCircle(new Vector3(pivot, rect.y + rect.height, 0f), 8f, UltiDraw.Green);
						UltiDraw.End();
						EditorGUILayout.EndVertical();
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
							MotionData.Frame next = frame.GetAnyNextStyleKey();
							Target.Timestamp = next == null ? Target.Data.GetTotalTime() : next.Timestamp;
						}
						EditorGUILayout.EndHorizontal();
					}

					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

						Utility.SetGUIColor(UltiDraw.Mustard);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Sequences");
						}

						for(int i=0; i<Target.Data.Sequences.Length; i++) {
						Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								Target.Data.Sequences[i].Start = EditorGUILayout.IntSlider("Start", Target.Data.Sequences[i].Start, 1, Target.Data.GetTotalFrames());
								Target.Data.Sequences[i].End = EditorGUILayout.IntSlider("End", Target.Data.Sequences[i].End, 1, Target.Data.GetTotalFrames());
							}
						}
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Add", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.Data.AddSequence(1, Target.Data.GetTotalFrames());
						}
						if(Utility.GUIButton("Remove", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.Data.RemoveSequence();
						}
						EditorGUILayout.EndHorizontal();

					}

					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

						Utility.SetGUIColor(UltiDraw.Mustard);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Settings");
						}

						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							Target.Data.SetUnitScale(EditorGUILayout.FloatField("Unit Scale", Target.Data.UnitScale));
							Target.Data.SetStyleTransition(EditorGUILayout.Slider("Style Transition", Target.Data.StyleTransition, 0.1f, 1f));
							Target.Data.SetMotionSmoothing(EditorGUILayout.Slider("Motion Smoothing", Target.Data.MotionSmoothing, 0f, 1f));
						}

						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Styles");
							string[] presets = new string[4] {"Select preset...", "Dan", "Dog", "Interaction"};
							switch(EditorGUILayout.Popup(0, presets)) {
								case 0:
								break;
								case 1:
								Target.Data.ClearStyles();
								Target.Data.AddStyle("Idle");
								Target.Data.AddStyle("Walk");
								Target.Data.AddStyle("Run");
								Target.Data.AddStyle("Jump");
								Target.Data.AddStyle("Crouch");
								break;
								case 2:
								Target.Data.ClearStyles();
								Target.Data.AddStyle("Idle");
								Target.Data.AddStyle("Move");
								Target.Data.AddStyle("Jump");
								Target.Data.AddStyle("Sit");
								Target.Data.AddStyle("Stand");
								Target.Data.AddStyle("Lie");
								break;
								case 3:
								Target.Data.ClearStyles();
								Target.Data.AddStyle("Idle");
								Target.Data.AddStyle("Walk");
								Target.Data.AddStyle("Run");
								Target.Data.AddStyle("Jump");
								Target.Data.AddStyle("Crouch");
								Target.Data.AddStyle("Sit");
								Target.Data.AddStyle("Open Door");
								Target.Data.AddStyle("PickUp");
								break;
							}
							for(int i=0; i<Target.Data.Styles.Length; i++) {
								EditorGUILayout.BeginHorizontal();
								Target.Data.Styles[i] = EditorGUILayout.TextField("Style " + (i+1), Target.Data.Styles[i]);
								EditorGUILayout.EndHorizontal();
							}
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("Add", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.AddStyle("Style");
							}
							if(Utility.GUIButton("Remove", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.Data.RemoveStyle();
							}
							EditorGUILayout.EndHorizontal();
						}

						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Sensors");
							Target.Data.GroundMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.Data.GroundMask), InternalEditorUtility.layers));
							Target.Data.ObjectMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Object Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.Data.ObjectMask), InternalEditorUtility.layers));
							string[] names = new string[Target.Data.Source.Bones.Length];
							for(int i=0; i<Target.Data.Source.Bones.Length; i++) {
								names[i] = Target.Data.Source.Bones[i].Name;
							}
							Target.Data.HeightMapRadius = EditorGUILayout.FloatField("Height Map Radius", Target.Data.HeightMapRadius);
							Target.Data.DepthMapSensor = EditorGUILayout.Popup("Depth Map Sensor", Target.Data.DepthMapSensor, names);
							Target.Data.DepthMapAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Depth Map Axis", Target.Data.DepthMapAxis);
							Target.Data.DepthMapResolution = EditorGUILayout.IntField("Depth Map Resolution", Target.Data.DepthMapResolution);
							Target.Data.DepthMapSize = EditorGUILayout.FloatField("Depth Map Size", Target.Data.DepthMapSize);
							Target.Data.DepthMapDistance = EditorGUILayout.FloatField("Depth Map Distance", Target.Data.DepthMapDistance);
						}

						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Mirroring");
							string[] presets = new string[4] {"Select preset...", "Dan", "Dog", "Interaction"};
							switch(EditorGUILayout.Popup(0, presets)) {
								case 0:
								break;
								case 1:
								for(int i=0; i<Target.Data.Corrections.Length; i++) {
									Target.Data.SetCorrection(i, Vector3.zero);
								}
								break;
								case 2:
								for(int i=0; i<Target.Data.Corrections.Length; i++) {
									if(i==4 || i==5 || i==6 || i==11) {
										Target.Data.SetCorrection(i, new Vector3(90f, 90f, 90f));
									} else if(i==24) {
										Target.Data.SetCorrection(i, new Vector3(-45f, 0f, 0f));
									} else {
										Target.Data.SetCorrection(i, new Vector3(0f, 0f, 0f));
									}
								}
								break;
								case 3:
								for(int i=0; i<Target.Data.Corrections.Length; i++) {
									Target.Data.SetCorrection(i, Vector3.zero);
								}
								break;
							}
							Target.Data.MirrorAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Axis", Target.Data.MirrorAxis);
							string[] names = new string[Target.Data.Source.Bones.Length];
							for(int i=0; i<Target.Data.Source.Bones.Length; i++) {
								names[i] = Target.Data.Source.Bones[i].Name;
							}
							for(int i=0; i<Target.Data.Source.Bones.Length; i++) {
								EditorGUILayout.BeginHorizontal();
								EditorGUI.BeginDisabledGroup(true);
								EditorGUILayout.TextField(names[i]);
								EditorGUI.EndDisabledGroup();
								Target.Data.SetSymmetry(i, EditorGUILayout.Popup(Target.Data.Symmetry[i], names));
								Target.Data.SetCorrection(i, EditorGUILayout.Vector3Field("", Target.Data.Corrections[i]));
								EditorGUILayout.EndHorizontal();
							}
						}
						
					}

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Unload", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.UnloadFile();
						}
					}

				}
			}
		}

	}

}
#endif
