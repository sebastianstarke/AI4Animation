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
	public MotionData Data = null;
	
	private bool Playing = false;
	private float PlayTime = 0f;
	private System.DateTime Timestamp;
	private float Timescale = 1f;
	private bool ShowMotion = false;
	private bool ShowVelocities = false;
	private bool ShowTrajectory = false;
	private bool ShowMirrored = false;

	private bool InspectFrame = true;
	private bool InspectSettings = true;

	private bool Spinning = false;

	void Awake() {
		StartSpin();
	}

	void OnDestroy() {
		StopSpin();
	}

	public void StartSpin() {
		if(Spinning) {
			return;
		}
		EditorApplication.update += EditorUpdate;
		Spinning = true;
	}

	public void StopSpin() {
		if(!Spinning) {
			return;
		}
		EditorApplication.update -= EditorUpdate;
		Spinning = false;
	}

	void EditorUpdate() {
		if(Data == null) {
			return;
		}

		if(Playing) {
			PlayTime += Timescale * (float)Utility.GetElapsedTime(Timestamp);
			if(PlayTime > Data.GetTotalTime()) {
				PlayTime -= Data.GetTotalTime();
			}
			Timestamp = Utility.GetTimestamp();
		}
		
		CheckSkeleton();

		MotionData.Frame frame = GetCurrentFrame();
		Matrix4x4 root = frame.GetRoot(ShowMirrored);
		Actor.GetRoot().position = root.GetPosition();
		Actor.GetRoot().rotation = root.GetRotation();
		for(int i=0; i<Actor.Bones.Length; i++) {
			Matrix4x4 transformation = frame.GetBoneTransformation(i, ShowMirrored);
			Actor.Bones[i].Transform.position = transformation.GetPosition();
			Actor.Bones[i].Transform.rotation = transformation.GetRotation();
		}
		SceneView.RepaintAll();
	}

	void Play() {
		if(!Playing) {
			Timestamp = Utility.GetTimestamp();
			Playing = true;
		}
	}

	void Stop() {
		if(Playing) {
			Playing = false;
		}
	}

	public void Draw() {
		if(Data == null) {
			return;
		}

		MotionData.Frame frame = GetCurrentFrame();

		if(ShowMotion) {
			for(int i=0; i<6; i++) {
				MotionData.Frame previous = Data.GetFrame(Mathf.Clamp(frame.Timestamp - 1f + (float)i/6f, 0f, Data.GetTotalTime()));
				Actor.DrawSimple(Color.Lerp(UltiDraw.Blue, UltiDraw.Cyan, 1f - (float)(i+1)/6f).Transparent(0.75f), previous.GetBoneTransformations(ShowMirrored));
			}
			for(int i=1; i<=5; i++) {
				MotionData.Frame future = Data.GetFrame(Mathf.Clamp(frame.Timestamp + (float)i/5f, 0f, Data.GetTotalTime()));
				Actor.DrawSimple(Color.Lerp(UltiDraw.Red, UltiDraw.Orange, (float)(i+1)/5f).Transparent(0.75f), future.GetBoneTransformations(ShowMirrored));

			}
		}

		if(ShowVelocities) {
			UltiDraw.Begin();
			for(int i=0; i<Actor.Bones.Length; i++) {
				UltiDraw.DrawArrow(
					Actor.Bones[i].Transform.position,
					Actor.Bones[i].Transform.position + frame.GetBoneVelocity(i, ShowMirrored),
					0.75f,
					0.0075f,
					0.05f,
					UltiDraw.Purple.Transparent(0.5f)
				);
			}
			UltiDraw.End();
		}

		if(ShowTrajectory) {
			frame.GetTrajectory(ShowMirrored).Draw();
		}

		frame.GetHeightMap(ShowMirrored).Draw();

		frame.GetDepthMap(ShowMirrored).Draw();
	}

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	public void LoadFile() {
		if(!File.Exists(Path)) {
			Debug.Log("File at path " + Path + " does not exist.");
			return;
		}
		Data = ScriptableObject.CreateInstance<MotionData>();
		Data.Load(Path);
		Playing = false;
		PlayTime = 0f;
		Timestamp = Utility.GetTimestamp();
		Timescale = 1f;
		CheckSkeleton();
	}

	public void UnloadFile() {
		Data = null;
		Playing = false;
		PlayTime = 0f;
		Timestamp = Utility.GetTimestamp();
		Timescale = 1f;
		CheckSkeleton();
	}

	public MotionData.Frame GetCurrentFrame() {
		return Data.GetFrame(PlayTime);
	}

	public void CheckSkeleton() {
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

	public void Inspector() {
		if(EditorApplication.isCompiling) {
			StopSpin();
		} else {
			StartSpin();
		}
		InspectImporter();
		InspectEditor();
	}

	private void InspectImporter() {
		if(Data != null) {
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
					Path = EditorGUILayout.TextField(Path);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Path = EditorUtility.OpenFilePanel("Motion Editor", Path == string.Empty ? Application.dataPath : Path.Substring(0, Path.LastIndexOf("/")), "bvh");
						GUI.SetNextControlName("");
						GUI.FocusControl("");
					}
					EditorGUILayout.EndHorizontal();
				}
				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
						LoadFile();
					}
				}
			}
		}
	}

	private void InspectEditor() {
		if(Data == null) {
			return;
		}
		Utility.SetGUIColor(UltiDraw.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Name: " + Data.Name);
					Actor = (Actor)EditorGUILayout.ObjectField("Actor", Actor, typeof(Actor), true);
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					GUILayout.FlexibleSpace();
					EditorGUILayout.LabelField("Frames: " + Data.GetTotalFrames(), GUILayout.Width(100f));
					EditorGUILayout.LabelField("Time: " + Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
					EditorGUILayout.LabelField("Framerate: " + Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
					EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
					Timescale = EditorGUILayout.FloatField(Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
					GUILayout.FlexibleSpace();
					EditorGUILayout.EndHorizontal();
				}

				EditorGUILayout.BeginHorizontal();
				GUILayout.FlexibleSpace();
				if(Playing) {
					if(Utility.GUIButton("||", Color.red, Color.black, 20f, 20f)) {
						Stop();
					}
				} else {
					if(Utility.GUIButton("|>", Color.green, Color.black, 20f, 20f)) {
						Play();
					}
				}
				if(Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
					PlayTime = Data.GetFrame(Mathf.Clamp(GetCurrentFrame().Index-1, 1, Data.GetTotalFrames())).Timestamp;
				}
				if(Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 20f)) {
					PlayTime = Data.GetFrame(Mathf.Clamp(GetCurrentFrame().Index+1, 1, Data.GetTotalFrames())).Timestamp;
				}
				int current = GetCurrentFrame().Index;
				int index = EditorGUILayout.IntSlider(current, 1, Data.GetTotalFrames(), GUILayout.Width(440f));
				if(index != current) {
					PlayTime = Data.GetFrame(index).Timestamp;
				}
				EditorGUILayout.LabelField(GetCurrentFrame().Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
				GUILayout.FlexibleSpace();
				EditorGUILayout.EndHorizontal();

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Motion", ShowMotion ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
					ShowMotion = !ShowMotion;
				}
				if(Utility.GUIButton("Trajectory", ShowTrajectory ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
					ShowTrajectory = !ShowTrajectory;
				}
				if(Utility.GUIButton("Velocities", ShowVelocities ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
					ShowVelocities = !ShowVelocities;
				}
				if(Utility.GUIButton("Mirror", ShowMirrored ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
					ShowMirrored = !ShowMirrored;
				}
				EditorGUILayout.EndHorizontal();

				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						InspectFrame = EditorGUILayout.Toggle("Frame", InspectFrame);
					}

					if(InspectFrame) {
						MotionData.Frame frame = GetCurrentFrame();

						Color[] colors = UltiDraw.GetRainbowColors(Data.Styles.Length);
						for(int i=0; i<Data.Styles.Length; i++) {
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton(Data.Styles[i], !frame.StyleFlags[i] ? colors[i].Transparent(0.25f) : colors[i], UltiDraw.White)) {
								frame.ToggleStyle(i);
							}
							EditorGUI.BeginDisabledGroup(true);
							EditorGUILayout.Slider(frame.StyleValues[i], 0f, 1f);
							EditorGUI.EndDisabledGroup();
							EditorGUILayout.EndHorizontal();
						}
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
							MotionData.Frame previous = GetCurrentFrame().GetAnyPreviousStyleKey();
							PlayTime = previous == null ? 0f : previous.Timestamp;
						}
						EditorGUILayout.BeginVertical(GUILayout.Height(50f));
						Rect ctrl = EditorGUILayout.GetControlRect();
						Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
						EditorGUI.DrawRect(rect, UltiDraw.Black);
						UltiDraw.Begin();
						for(int i=0; i<Data.Styles.Length; i++) {
							int x = 0;
							for(int j=1; j<Data.GetTotalFrames(); j++) {
								float val = Data.Frames[j].StyleValues[i];
								if(
									Data.Frames[x].StyleValues[i]<1f && val==1f ||
									Data.Frames[x].StyleValues[i]>0f && val==0f
									) {
									float xStart = rect.x + (float)(x-1)/(float)(Data.GetTotalFrames()-1) * rect.width;
									float xEnd = rect.x + (float)j/(float)(Data.GetTotalFrames()-1) * rect.width;
									float yStart = rect.y + (1f - Data.Frames[x-1].StyleValues[i]) * rect.height;
									float yEnd = rect.y + (1f - Data.Frames[j].StyleValues[i]) * rect.height;
									UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
									x = j;
								}
								if(
									Data.Frames[x].StyleValues[i]==0f && val>0f || 
									Data.Frames[x].StyleValues[i]==1f && val<1f
									) {
									float xStart = rect.x + (float)(x)/(float)(Data.GetTotalFrames()-1) * rect.width;
									float xEnd = rect.x + (float)(j-1)/(float)(Data.GetTotalFrames()-1) * rect.width;
									float yStart = rect.y + (1f - Data.Frames[x].StyleValues[i]) * rect.height;
									float yEnd = rect.y + (1f - Data.Frames[j-1].StyleValues[i]) * rect.height;
									UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
									x = j;
								}
								if(j==Data.GetTotalFrames()-1) {
									float xStart = rect.x + (float)x/(float)(Data.GetTotalFrames()-1) * rect.width;
									float xEnd = rect.x + (float)(j-1)/(float)(Data.GetTotalFrames()-1) * rect.width;
									float yStart = rect.y + (1f - Data.Frames[x].StyleValues[i]) * rect.height;
									float yEnd = rect.y + (1f - Data.Frames[j-1].StyleValues[i]) * rect.height;
									UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
									x = j;
								}
							}
						}
						float pivot = rect.x + (float)(frame.Index-1)/(float)(Data.GetTotalFrames()-1) * rect.width;
						UltiDraw.DrawLine(new Vector3(pivot, rect.y, 0f), new Vector3(pivot, rect.y + rect.height, 0f), UltiDraw.White);
						UltiDraw.DrawWireCircle(new Vector3(pivot, rect.y, 0f), 8f, UltiDraw.Green);
						UltiDraw.DrawWireCircle(new Vector3(pivot, rect.y + rect.height, 0f), 8f, UltiDraw.Green);
						UltiDraw.End();
						EditorGUILayout.EndVertical();
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
							MotionData.Frame next = GetCurrentFrame().GetAnyNextStyleKey();
							PlayTime = next == null ? Data.GetTotalTime() : next.Timestamp;
						}
						EditorGUILayout.EndHorizontal();
					}
				}				

				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						InspectSettings = EditorGUILayout.Toggle("Settings", InspectSettings);
					}

					if(InspectSettings) {
						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							Data.SetUnitScale(EditorGUILayout.FloatField("Unit Scale", Data.UnitScale));
							Data.SetStyleTransition(EditorGUILayout.FloatField("Style Transition", Data.StyleTransition));
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
								Data.ClearStyles();
								Data.AddStyle("Idle");
								Data.AddStyle("Walk");
								Data.AddStyle("Run");
								Data.AddStyle("Jump");
								Data.AddStyle("Crouch");
								break;
								case 2:
								Data.ClearStyles();
								Data.AddStyle("Idle");
								Data.AddStyle("Move");
								Data.AddStyle("Jump");
								Data.AddStyle("Sit");
								Data.AddStyle("Stand");
								Data.AddStyle("Lie");
								break;
								case 3:
								Data.ClearStyles();
								Data.AddStyle("Idle");
								Data.AddStyle("Walk");
								Data.AddStyle("Run");
								Data.AddStyle("Jump");
								Data.AddStyle("Crouch");
								Data.AddStyle("Sit");
								Data.AddStyle("OpenDoor");
								Data.AddStyle("PickUp");
								break;
							}
							for(int i=0; i<Data.Styles.Length; i++) {
								EditorGUILayout.BeginHorizontal();
								Data.Styles[i] = EditorGUILayout.TextField("Style " + (i+1), Data.Styles[i]);
								EditorGUILayout.EndHorizontal();
							}
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("Add", UltiDraw.DarkGrey, UltiDraw.White)) {
								Data.AddStyle("Style");
							}
							if(Utility.GUIButton("Remove", UltiDraw.DarkGrey, UltiDraw.White)) {
								Data.RemoveStyle();
							}
							EditorGUILayout.EndHorizontal();
						}

						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Sensors");
							Data.GroundMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Data.GroundMask), InternalEditorUtility.layers));
							Data.ObjectMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Object Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Data.ObjectMask), InternalEditorUtility.layers));
							string[] names = new string[Data.Source.Bones.Length];
							for(int i=0; i<Data.Source.Bones.Length; i++) {
								names[i] = Data.Source.Bones[i].Name;
							}
							Data.HeightMapSensor = EditorGUILayout.Popup("Height Map Sensor", Data.HeightMapSensor, names);
							Data.DepthMapSensor = EditorGUILayout.Popup("Depth Map Sensor", Data.DepthMapSensor, names);
							Data.DepthMapAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Depth Map Axis", Data.DepthMapAxis);
						}

						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.LabelField("Mirroring");
							Data.MirrorAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Axis", Data.MirrorAxis);
							string[] names = new string[Data.Source.Bones.Length];
							for(int i=0; i<Data.Source.Bones.Length; i++) {
								names[i] = Data.Source.Bones[i].Name;
							}
							for(int i=0; i<Data.Source.Bones.Length; i++) {
								EditorGUILayout.BeginHorizontal();
								EditorGUI.BeginDisabledGroup(true);
								EditorGUILayout.TextField(names[i]);
								EditorGUI.EndDisabledGroup();
								Data.Symmetry[i] = EditorGUILayout.Popup(Data.Symmetry[i], names);
								EditorGUILayout.EndHorizontal();
							}
						}
					}
				}

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Utility.GUIButton("Unload", UltiDraw.DarkGrey, UltiDraw.White)) {
						UnloadFile();
					}
				}

			}
		}
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		void Awake() {
			Target = (MotionEditor)target;
			EditorApplication.update += Update;
		}

		void OnDestroy() {
   		 	EditorApplication.update -= Update;
			if(!Application.isPlaying) {
				EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
			}
		}

		void Update() {
			Repaint();
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Target.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}

}
#endif
