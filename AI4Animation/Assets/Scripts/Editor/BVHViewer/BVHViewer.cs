using UnityEngine;
using UnityEditor;

public class BVHViewer : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;
	public static System.DateTime Timestamp;
	public static int RefreshRate = 30;

	public float UnitScale = 10f;
	public string Path = string.Empty;
	public BVHAnimation Animation = null;

	[MenuItem ("Addons/BVH Viewer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHViewer));
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
			System.GC.Collect();
		}
	}

	void OnGUI() {
		Utility.SetGUIColor(Utility.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Importer");
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					RefreshRate = EditorGUILayout.IntField("Refresh Rate", RefreshRate);
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					UnitScale = EditorGUILayout.FloatField("Unit Scale", UnitScale);
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Path", GUILayout.Width(30));
					Path = EditorGUILayout.TextField(Path);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Path = EditorUtility.OpenFilePanel("BVH Viewer", Path == string.Empty ? Application.dataPath : Path.Substring(0, Path.LastIndexOf("/")), "bvh");
						GUI.SetNextControlName("");
						GUI.FocusControl("");
					}
					EditorGUILayout.EndHorizontal();
				}
				if(Utility.GUIButton("Load", Utility.DarkGrey, Utility.White)) {
					Load();
				}
			}

			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
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
