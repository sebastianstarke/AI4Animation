using UnityEngine;
using UnityEditor;

public class BVHEditor : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;
	public static System.DateTime Timestamp;
	public static int RefreshRate = 60;
	
	public bool AutoFocus = true;
	public float FocusDistance = 2.5f;
	public float FocusAngle = 180f;
	public float FocusSmoothing = 0.5f;

	public string Path = string.Empty;
	public BVHAnimation Animation = null;

	[MenuItem ("Addons/BVH Editor")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHEditor));
		Scroll = Vector3.zero;
		Timestamp = Utility.GetTimestamp();
	}

	void OnFocus() {
		SceneView.onSceneGUIDelegate -= this.OnSceneGUI;
		SceneView.onSceneGUIDelegate += this.OnSceneGUI;

		if(Animation != null) {
			Animation.Timestamp = Utility.GetTimestamp();
		}
	}

	void OnDestroy() {
		SceneView.onSceneGUIDelegate -= this.OnSceneGUI;
		Save();
	}

	void Update() {
		if(Animation == null) {
			return;
		}
		
		Animation.EditorUpdate();
		SceneView.RepaintAll();

		if(Utility.GetElapsedTime(Timestamp) > 1f/(float)RefreshRate) {
			Repaint();
			Timestamp = Utility.GetTimestamp();
		}
	}

	void OnGUI() {
		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Importer");
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					RefreshRate = EditorGUILayout.IntField("Refresh Rate", RefreshRate);
					SetAutoFocus(EditorGUILayout.Toggle("Auto Focus", AutoFocus));
					FocusDistance = EditorGUILayout.FloatField("Focus Distance", FocusDistance);
					FocusAngle = EditorGUILayout.Slider("Focus Angle", FocusAngle, 0f, 360f);
					FocusSmoothing = EditorGUILayout.Slider("Focus Smoothing", FocusSmoothing, 0f, 1f);
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Path", GUILayout.Width(50));
					Path = EditorGUILayout.TextField(Path);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Path = EditorUtility.OpenFilePanel("BVH Editor", Path == string.Empty ? Application.dataPath : Path.Substring(0, Path.LastIndexOf("/")), "bvh");
						GUI.SetNextControlName("");
						GUI.FocusControl("");
					}
					EditorGUILayout.EndHorizontal();
				}
				if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
					Load();
				}
			}

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.DarkGreen);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					Load((BVHAnimation)EditorGUILayout.ObjectField("Animation", Animation, typeof(BVHAnimation), false));
				}

				if(Animation != null) {
					Scroll = EditorGUILayout.BeginScrollView(Scroll);
					Animation.Inspector();
					EditorGUILayout.EndScrollView();
				}

			}
		}
		Timestamp = Utility.GetTimestamp();
	}

	void OnSceneGUI(SceneView view) {
		if(Animation != null) {
			Animation.Draw();
			if(AutoFocus) {
				Vector3 lastPosition = view.camera.transform.position;
				Quaternion lastRotation = view.camera.transform.rotation;
				Vector3 position = Animation.ShowMirrored ? Animation.CurrentFrame.World[0].GetPosition().GetMirror(Animation.GetMirrorAxis()) : Animation.CurrentFrame.World[0].GetPosition();
				Quaternion rotation = Animation.ShowMirrored ? Animation.CurrentFrame.World[0].GetRotation().GetMirror(Animation.GetMirrorAxis()) : Animation.CurrentFrame.World[0].GetRotation();
				rotation.x = 0f;
				rotation.z = 0f;
				rotation = Quaternion.Euler(0f, Animation.ShowMirrored ? Mathf.Repeat(FocusAngle + 0f, 360f) : FocusAngle, 0f) * rotation;
				SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, FocusSmoothing), FocusDistance*FocusSmoothing);
			}
		}
	}

	private void SetAutoFocus(bool value) {
		if(AutoFocus != value) {
			AutoFocus = value;
			if(!AutoFocus) {
				Vector3 position = Animation.ShowMirrored ? Animation.CurrentFrame.World[0].GetPosition().GetMirror(Animation.GetMirrorAxis()) : Animation.CurrentFrame.World[0].GetPosition();
				Quaternion rotation = Quaternion.Euler(0f, Mathf.Repeat(FocusAngle + 180f, 360f), 0f);
				SceneView.lastActiveSceneView.LookAtDirect(position, rotation, FocusDistance);
			}
		}
	}
	
	private void Load() {
		Load(ScriptableObject.CreateInstance<BVHAnimation>().Create(this));
	}

	private void Load(BVHAnimation animation) {
		if(Animation != animation) {
			Save();
			Animation = animation;
		}
	}

	private void Save() {
		if(Animation != null) {
			Animation.Stop();
			EditorUtility.SetDirty(Animation);
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
		}
	}

}
