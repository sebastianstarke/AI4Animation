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

	public static string Path = string.Empty;
	
	public MotionData Data = null;

	private bool AutoFocus = true;
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

	private bool InspectCamera = true;
	private bool InspectFrame = true;
	private bool InspectExport = true;
	private bool InspectSettings = true;

	private Actor Actor = null;
	private Transform Scene = null;
	private MotionState State;
	
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
		if(Data == null) {
			return;
		}
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
		Mirror = value;
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

	public MotionState GetState() {
		if(State == null) {
			LoadFrame(Timestamp);
		}
		return State;
	}

	public void LoadFile(string path) {
		if(!File.Exists(path)) {
			Debug.Log("File at path " + path + " does not exist.");
			return;
		}
		Data = ScriptableObject.CreateInstance<MotionData>().Create(path, EditorSceneManager.GetActiveScene().path.Substring(0, EditorSceneManager.GetActiveScene().path.LastIndexOf("/")+1));
		Data.Scene = AssetDatabase.LoadAssetAtPath<SceneAsset>(EditorSceneManager.GetActiveScene().path);
		AssetDatabase.RenameAsset(UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, Data.name);
	}

	public void UnloadFile() {
		AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(Data));
		AssetDatabase.RenameAsset(UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, "Empty");
	}

	public void LoadFrame(MotionState state) {
		Timestamp = state.Timestamp;
		State = state;
		if(state.Mirrored) {
			GetScene().localScale = Vector3.one.GetMirror(Data.GetAxis(Data.MirrorAxis));
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
		Timestamp = timestamp;
		State = new MotionState(Data.GetFrame(Timestamp), Mirror);
		
		if(Mirror) {
			GetScene().localScale = Vector3.one.GetMirror(Data.GetAxis(Data.MirrorAxis));
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
		while(Data != null) {
			Timestamp += Timescale * (float)Utility.GetElapsedTime(timestamp);
			if(Timestamp > Data.GetTotalTime()) {
				Timestamp = Mathf.Repeat(Timestamp, Data.GetTotalTime());
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
					//intensity = Utility.TanH(intensity);
					UltiDraw.DrawGUIRectangle(Vector2.one/2f - size/2f + new Vector2((float)x*size.x, (float)y*size.y) / (GetState().DepthMap.GetResolution()-1), size / (GetState().DepthMap.GetResolution()-1), Color.Lerp(Color.black, Color.white, intensity));
				}
			}
			UltiDraw.End();
		}

		//Motion Function
		/*
		MotionData.Frame[] frames = Data.GetFrames(Mathf.Clamp(GetState().Timestamp-1f, 0f, Data.GetTotalTime()), Mathf.Clamp(GetState().Timestamp+1f, 0f, Data.GetTotalTime()));
		float[] values = new float[frames.Length];
		for(int i=0; i<frames.Length; i++) {
			values[i] = frames[i].GetBoneVelocity(0, Mirror).magnitude;
		}
		Debug.Log(values[0]);
		UltiDraw.Begin();
		UltiDraw.DrawGUIFunction(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), values, -2f, 2f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.Green);
		UltiDraw.DrawGUILine(new Vector2(0.5f, 1f), new Vector2(0.5f, 0f), 0.0025f, UltiDraw.IndianRed);
		UltiDraw.End();
		*/
		/*
		//Bone Velocities
		MotionData.Frame[] frames = Data.GetFrames(Mathf.Clamp(GetState().Timestamp-1f, 0f, Data.GetTotalTime()), Mathf.Clamp(GetState().Timestamp+1f, 0f, Data.GetTotalTime()));
		List<float[]> values = new List<float[]>();
		for(int i=0; i<Actor.Bones.Length; i++) {
			values.Add(new float[frames.Length]);
		}
		for(int i=0; i<frames.Length; i++) {
			for(int j=0; j<Actor.Bones.Length; j++) {
				values[j][i] = frames[i].GetBoneVelocity(j, Mirror).magnitude;
			}
		}
		UltiDraw.Begin();
		UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), values, 0f, 2f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
		UltiDraw.DrawGUILine(new Vector2(0.5f, 1f), new Vector2(0.5f, 0f), 0.0025f, UltiDraw.Green);
		UltiDraw.End();
		*/
		
		/*
		//Trajectory Motion
		MotionData.Frame[] frames = Data.GetFrames(Mathf.Clamp(GetState().Timestamp-1f, 0f, Data.GetTotalTime()), Mathf.Clamp(GetState().Timestamp+1f, 0f, Data.GetTotalTime()));
		List<float[]> values = new List<float[]>(3);
		for(int i=0; i<6; i++) {
			values.Add(new float[frames.Length]);
		}
		for(int i=0; i<frames.Length; i++) {
			Vector3 motion = frames[i].GetRootMotion(Mirror);
			values[0][i] = motion.x;
			values[1][i] = motion.y / 180f;
			values[2][i] = motion.z;
			Vector3 velocity = frames[i].GetRootVelocity(Mirror);
			values[3][i] = velocity.x;
			values[4][i] = velocity.y;
			values[5][i] = velocity.z;
		}
		UltiDraw.Begin();
		UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), values, -2f, 2f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
		UltiDraw.DrawGUILine(new Vector2(0.5f, 1f), new Vector2(0.5f, 0f), 0.0025f, UltiDraw.Green);
		UltiDraw.End();
		*/		
		
		//Agility Function
		/*
		MotionData.Frame[] frames = Data.GetFrames(Mathf.Clamp(GetState().Timestamp-1f, 0f, Data.GetTotalTime()), Mathf.Clamp(GetState().Timestamp+1f, 0f, Data.GetTotalTime()));
		List<float[]> values = new List<float[]>();
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			values.Add(new float[frames.Length]);
			for(int j=0; j<frames.Length; j++) {
				values[i][j] = frames[j].GetAgility(i, Mirror);
			}
		}
		UltiDraw.Begin();
		UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), values, -1f, 1f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
		UltiDraw.DrawGUILine(new Vector2(0.5f, 1f), new Vector2(0.5f, 0f), 0.0025f, UltiDraw.Green);
		UltiDraw.End();
		*/
	}

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	[MenuItem("Assets/Create/Motion Capture")]
	public static string CreateMotionCapture() {
		string source = Application.dataPath + "/Project/MotionCapture/Setup.unity";
		string destination = AssetDatabase.GetAssetPath(Selection.activeObject) + "/Empty.unity";
		int index = 0;
		while(File.Exists(destination)) {
			index += 1;
			destination = AssetDatabase.GetAssetPath(Selection.activeObject) + "/Empty (" + index +").unity";
		}
		if(!File.Exists(source)) {
			Debug.Log("Source file at path " + source + " does not exist.");
		} else {
			FileUtil.CopyFileOrDirectory(source, destination);
		}
		return destination;
	}

	public static string CreateMotionCapture(string path, string name) {
		string source = Application.dataPath + "/Project/MotionCapture/Setup.unity";
		string destination = (path ==  "" ? AssetDatabase.GetAssetPath(Selection.activeObject) : path) + "/" + name + ".unity";
		int index = 0;
		while(File.Exists(destination)) {
			index += 1;
			destination = (path ==  "" ? AssetDatabase.GetAssetPath(Selection.activeObject) : path) + "/" + name + "(" + index +").unity";
		}
		if(!File.Exists(source)) {
			Debug.Log("Source file at path " + source + " does not exist.");
		} else {
			FileUtil.CopyFileOrDirectory(source, destination);
		}
		return destination;
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor {

		public MotionEditor Target;

		private float RefreshRate = 30f;
		private System.DateTime Timestamp;

		void Awake() {
			Target = (MotionEditor)target;
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			if(!Application.isPlaying && Target != null) {
				EditorUtility.SetDirty(Target);
				if(Target.Data != null) {
					EditorUtility.SetDirty(Target.Data);
				}
				EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
			}
			EditorApplication.update -= EditorUpdate;
		}

		public void Save() {
			if(!Application.isPlaying && Target != null) {
				if(Target.Data != null) {
					EditorUtility.SetDirty(Target.Data);
					AssetDatabase.SaveAssets();
					AssetDatabase.Refresh();
				}
				EditorUtility.SetDirty(Target);
				EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
			}
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
						MotionEditor.Path = EditorGUILayout.TextField(MotionEditor.Path);
						GUI.skin.button.alignment = TextAnchor.MiddleCenter;
						if(GUILayout.Button("O", GUILayout.Width(20))) {
							MotionEditor.Path = EditorUtility.OpenFilePanel("Motion Editor", MotionEditor.Path == string.Empty ? Application.dataPath : MotionEditor.Path.Substring(0, MotionEditor.Path.LastIndexOf("/")), "bvh");
							GUI.SetNextControlName("");
							GUI.FocusControl("");
						}
						EditorGUILayout.EndHorizontal();
					}
					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.LoadFile(MotionEditor.Path);
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
						
						Utility.SetGUIColor(Target.GetActor() == null ? UltiDraw.DarkRed : UltiDraw.White);
						EditorGUILayout.ObjectField("Actor", Target.GetActor(), typeof(Actor), true);
						Utility.ResetGUIColor();

						Utility.SetGUIColor(Target.GetScene() == null ? UltiDraw.DarkRed : UltiDraw.White);
						EditorGUILayout.ObjectField("Scene", Target.GetScene(), typeof(Transform), true);
						Utility.ResetGUIColor();

						EditorGUILayout.ObjectField("Data", Target.Data, typeof(MotionData), true);
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
							Target.InspectCamera = EditorGUILayout.Toggle("Camera", Target.InspectCamera);
						}

						if(Target.InspectCamera) {
							if(Utility.GUIButton("Auto Focus", Target.AutoFocus ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black)) {
								Target.SetAutoFocus(!Target.AutoFocus);
							}
							Target.FocusHeight = EditorGUILayout.FloatField("Focus Height", Target.FocusHeight);
							Target.FocusOffset = EditorGUILayout.FloatField("Focus Offset", Target.FocusOffset);
							Target.FocusDistance = EditorGUILayout.FloatField("Focus Distance", Target.FocusDistance);
							Target.FocusAngle = EditorGUILayout.Slider("Focus Angle", Target.FocusAngle, 0f, 360f);
							Target.FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", Target.FocusSmoothing, 0f, 1f);
						}
					}

					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

						Utility.SetGUIColor(UltiDraw.Mustard);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							Target.InspectFrame = EditorGUILayout.Toggle("Frame", Target.InspectFrame);
						}

						if(Target.InspectFrame) {
							Color[] colors = UltiDraw.GetRainbowColors(Target.Data.Styles.Length);
							for(int i=0; i<Target.Data.Styles.Length; i++) {
								float height = 25f;
								EditorGUILayout.BeginHorizontal();
								if(Utility.GUIButton(Target.Data.Styles[i], !frame.StyleFlags[i] ? colors[i].Transparent(0.25f) : colors[i], UltiDraw.White, 200f, height)) {
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
										float xStart = rect.x + (float)(Mathf.Max(x-1, 0))/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
										float xEnd = rect.x + (float)j/(float)(Target.Data.GetTotalFrames()-1) * rect.width;
										float yStart = rect.y + (1f - Target.Data.Frames[Mathf.Max(x-1, 0)].StyleValues[i]) * rect.height;
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
								Target.LoadFrame(next == null ? Target.Data.GetTotalTime() : next.Timestamp);
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
							Target.InspectExport = EditorGUILayout.Toggle("Export", Target.InspectExport);
						}

						if(Target.InspectExport) {
							for(int i=0; i<Target.Data.Sequences.Length; i++) {
							Utility.SetGUIColor(UltiDraw.LightGrey);
								using(new EditorGUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									
									EditorGUILayout.BeginHorizontal();
									GUILayout.FlexibleSpace();
									if(Utility.GUIButton("X", Color.cyan, Color.black, 15f, 15f)) {
										Target.Data.Sequences[i].SetStart(Target.GetState().Index);
									}
									EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
									Target.Data.Sequences[i].SetStart(EditorGUILayout.IntField(Target.Data.Sequences[i].Start, GUILayout.Width(100f)));
									EditorGUILayout.LabelField("End", GUILayout.Width(50f));
									Target.Data.Sequences[i].SetEnd(EditorGUILayout.IntField(Target.Data.Sequences[i].End, GUILayout.Width(100f)));
									if(Utility.GUIButton("X", Color.cyan, Color.black, 15f, 15f)) {
										Target.Data.Sequences[i].SetEnd(Target.GetState().Index);
									}
									GUILayout.FlexibleSpace();
									EditorGUILayout.EndHorizontal();

									for(int s=0; s<Target.Data.Styles.Length; s++) {
										EditorGUILayout.BeginHorizontal();
										GUILayout.FlexibleSpace();
										EditorGUILayout.LabelField(Target.Data.Styles[s], GUILayout.Width(50f));
										EditorGUILayout.LabelField("Style Copies", GUILayout.Width(100f));
										Target.Data.Sequences[i].SetStyleCopies(s, EditorGUILayout.IntField(Target.Data.Sequences[i].StyleCopies[s], GUILayout.Width(100f)));
										EditorGUILayout.LabelField("Transition Copies", GUILayout.Width(100f));
										Target.Data.Sequences[i].SetTransitionCopies(s, EditorGUILayout.IntField(Target.Data.Sequences[i].TransitionCopies[s], GUILayout.Width(100f)));
										GUILayout.FlexibleSpace();
										EditorGUILayout.EndHorizontal();
									}
									//for(int c=0; c<Target.Data.Sequences[i].Copies.Length; c++) {
									//	EditorGUILayout.LabelField("Copy " + (c+1) + " - " + "Start: " + Target.Data.Sequences[i].Copies[c].Start + " End: " + Target.Data.Sequences[i].Copies[c].End);
									//}
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

					}

					Utility.SetGUIColor(UltiDraw.Grey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

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
								Target.Data.DepthMapAxis = MotionData.Axis.ZPositive;
								Target.Data.SetUnitScale(10f);
								Target.Data.MirrorAxis = MotionData.Axis.XPositive;
								for(int i=0; i<Target.Data.Corrections.Length; i++) {
									Target.Data.SetCorrection(i, Vector3.zero);
								}
								Target.Data.ClearStyles();
								Target.Data.AddStyle("Idle");
								Target.Data.AddStyle("Walk");
								Target.Data.AddStyle("Run");
								Target.Data.AddStyle("Jump");
								Target.Data.AddStyle("Crouch");
								break;

								case 2:
								Target.Data.DepthMapAxis = MotionData.Axis.XPositive;
								Target.Data.SetUnitScale(100f);
								Target.Data.MirrorAxis = MotionData.Axis.ZPositive;
								for(int i=0; i<Target.Data.Corrections.Length; i++) {
									if(i==4 || i==5 || i==6 || i==11) {
										Target.Data.SetCorrection(i, new Vector3(90f, 90f, 90f));
									} else if(i==24) {
										Target.Data.SetCorrection(i, new Vector3(-45f, 0f, 0f));
									} else {
										Target.Data.SetCorrection(i, new Vector3(0f, 0f, 0f));
									}
								}
								Target.Data.ClearStyles();
								Target.Data.AddStyle("Idle");
								Target.Data.AddStyle("Walk");
								Target.Data.AddStyle("Pace");
								Target.Data.AddStyle("Trot");
								Target.Data.AddStyle("Canter");
								Target.Data.AddStyle("Jump");
								Target.Data.AddStyle("Sit");
								Target.Data.AddStyle("Stand");
								Target.Data.AddStyle("Lie");
								break;

								case 3:
								Target.Data.DepthMapAxis = MotionData.Axis.ZPositive;
								Target.Data.SetUnitScale(100f);
								Target.Data.MirrorAxis = MotionData.Axis.XPositive;							
								for(int i=0; i<Target.Data.Corrections.Length; i++) {
									Target.Data.SetCorrection(i, Vector3.zero);
								}
								Target.Data.ClearStyles();
								Target.Data.AddStyle("Idle");
								Target.Data.AddStyle("Walk");
								Target.Data.AddStyle("Run");
								Target.Data.AddStyle("Jump");
								Target.Data.AddStyle("Crouch");
								Target.Data.AddStyle("Sit");
								break;
							}

							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								if(Utility.GUIButton("Create Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
									Target.CreateSkeleton();
								}
							}

							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								EditorGUILayout.LabelField("General");

								Target.Data.SetUnitScale(EditorGUILayout.FloatField("Unit Scale", Target.Data.UnitScale));
								Target.Data.RootSmoothing = EditorGUILayout.IntField("Root Smoothing", Target.Data.RootSmoothing);
								
								Target.Data.GroundMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.Data.GroundMask), InternalEditorUtility.layers));
								Target.Data.ObjectMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Object Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.Data.ObjectMask), InternalEditorUtility.layers));
								
								string[] names = new string[Target.Data.Source.Bones.Length];
								for(int i=0; i<Target.Data.Source.Bones.Length; i++) {
									names[i] = Target.Data.Source.Bones[i].Name;
								}
								Target.Data.HeightMapSensor = EditorGUILayout.Popup("Height Map Sensor", Target.Data.HeightMapSensor, names);
								Target.Data.HeightMapSize = EditorGUILayout.Slider("Height Map Size", Target.Data.HeightMapSize, 0f, 1f);
								Target.Data.DepthMapSensor = EditorGUILayout.Popup("Depth Map Sensor", Target.Data.DepthMapSensor, names);
								Target.Data.DepthMapAxis = (MotionData.Axis)EditorGUILayout.EnumPopup("Depth Map Axis", Target.Data.DepthMapAxis);
								Target.Data.DepthMapResolution = EditorGUILayout.IntField("Depth Map Resolution", Target.Data.DepthMapResolution);
								Target.Data.DepthMapSize = EditorGUILayout.FloatField("Depth Map Size", Target.Data.DepthMapSize);
								Target.Data.DepthMapDistance = EditorGUILayout.FloatField("Depth Map Distance", Target.Data.DepthMapDistance);
								
								Target.Data.SetStyleTransition(EditorGUILayout.Slider("Style Transition", Target.Data.StyleTransition, 0.1f, 1f));
								for(int i=0; i<Target.Data.Styles.Length; i++) {
									EditorGUILayout.BeginHorizontal();
									Target.Data.Styles[i] = EditorGUILayout.TextField("Style " + (i+1), Target.Data.Styles[i]);
									EditorGUILayout.EndHorizontal();
								}
								EditorGUILayout.BeginHorizontal();
								if(Utility.GUIButton("Add Style", UltiDraw.DarkGrey, UltiDraw.White)) {
									Target.Data.AddStyle("Style");
								}
								if(Utility.GUIButton("Remove Style", UltiDraw.DarkGrey, UltiDraw.White)) {
									Target.Data.RemoveStyle();
								}
								EditorGUILayout.EndHorizontal();
							}

							Utility.SetGUIColor(UltiDraw.LightGrey);
							using(new EditorGUILayout.VerticalScope ("Box")) {
								Utility.ResetGUIColor();
								EditorGUILayout.LabelField("Mirroring");
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
