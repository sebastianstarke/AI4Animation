#if UNITY_EDITOR
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[ExecuteInEditMode]
[RequireComponent(typeof(Actor))]
public class MotionEditor : MonoBehaviour {

	public string Path = string.Empty;

	public MotionData Data = null;
	
	private bool Playing = false;
	private float PlayTime = 0f;
	private System.DateTime Timestamp;
	private float Timescale = 1f;

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
		if(Playing) {
			PlayTime += Timescale * (float)Utility.GetElapsedTime(Timestamp);
			if(PlayTime > Data.GetTotalTime()) {
				PlayTime -= Data.GetTotalTime();
			}
			Timestamp = Utility.GetTimestamp();
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
		Matrix4x4 root = frame.GetRoot();
		GetActor().GetRoot().position = root.GetPosition();
		GetActor().GetRoot().rotation = root.GetRotation();
		for(int i=0; i<GetActor().Bones.Length; i++) {
			GetActor().Bones[i].Transform.position = frame.World[i].GetPosition();
			GetActor().Bones[i].Transform.rotation = frame.World[i].GetRotation();
			UltiDraw.Begin();
			UltiDraw.DrawArrow(
				GetActor().Bones[i].Transform.position,
				GetActor().Bones[i].Transform.position + frame.GetBoneVelocity(i),
				0.75f,
				0.0075f,
				0.05f,
				UltiDraw.Purple.Transparent(0.5f)
			);
			UltiDraw.End();
		}
		frame.GetTrajectory().Draw(10);
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
		CreateActor();
	}

	public void UnloadFile() {
		Data = null;
		Playing = false;
		PlayTime = 0f;
		Timestamp = Utility.GetTimestamp();
		Timescale = 1f;
		ClearActor();
	}

	public MotionData.Frame GetCurrentFrame() {
		return Data.GetFrame(PlayTime);
	}

	public void CreateActor() {
		string[] names = new string[Data.Source.Bones.Length];
		string[] parents = new string[Data.Source.Bones.Length];
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			names[i] = Data.Source.Bones[i].Name;
			parents[i] = Data.Source.Bones[i].Parent;
		}
		List<Transform> instances = new List<Transform>();
		for(int i=0; i<names.Length; i++) {
			Transform instance = new GameObject(names[i]).transform;
			instance.SetParent(parents[i] == "None" ? GetActor().GetRoot() : GetActor().FindTransform(parents[i]));
			instances.Add(instance);
		}
		GetActor().ExtractSkeleton(instances.ToArray());
	}

	public void ClearActor() {
		Transform root = GetActor().GetRoot();
		while(root.childCount > 0) {
			Utility.Destroy(root.GetChild(0).gameObject);
		}
		GetActor().ExtractSkeleton();
	}

	public Actor GetActor() {
		return GetComponent<Actor>();
	}

	public void Inspector() {
		StartSpin();
		InspectImporter();
		InspectEditor();
	}

	private void InspectImporter() {
		Utility.SetGUIColor(UltiDraw.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.LabelField("Importer");
			}

			if(Data == null) {
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
				if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
					LoadFile();
				}
			} else {
				if(Utility.GUIButton("Unload", UltiDraw.DarkRed, UltiDraw.White)) {
					UnloadFile();
				}
			}

		}
	}

	private void InspectEditor() {
		Utility.SetGUIColor(UltiDraw.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Orange);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.LabelField("Editor");
			}

			if(Data == null) {
				EditorGUILayout.LabelField("No file loaded.");
			} else {
				EditorGUILayout.LabelField("Name: " + Data.Name);

				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Frames: " + Data.GetTotalFrames(), GUILayout.Width(100f));
						EditorGUILayout.LabelField("Time: " + Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
						EditorGUILayout.LabelField("Framerate: " + Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
						EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
						Timescale = EditorGUILayout.FloatField(Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
						EditorGUILayout.EndHorizontal();
					}

					EditorGUILayout.BeginHorizontal();
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
					
					EditorGUILayout.EndHorizontal();
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
