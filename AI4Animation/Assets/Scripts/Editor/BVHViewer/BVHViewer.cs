using UnityEngine;
using UnityEditor;

public class BVHViewer : EditorWindow {

	public static EditorWindow Window;

	public float UnitScale = 10f;
	public string Path = string.Empty;
	public BVHAnimation Animation = null;

	[MenuItem ("Addons/BVH Viewer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHViewer));
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

		if(Animation != null) {
			Animation.Save();
		}
	}

	void Update() {
		if(Animation == null) {
			return;
		}

		Animation.EditorUpdate();
		SceneView.RepaintAll();
		Repaint();
	}

	void OnGUI() {
		Utility.SetGUIColor(Utility.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(Utility.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Importer");
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
				if(Utility.GUIButton("Load", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Load();
				}
			}

			Utility.SetGUIColor(Utility.Teal);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					Load((BVHAnimation)EditorGUILayout.ObjectField("Animation", Animation, typeof(BVHAnimation), false));
				}

				if(Animation == null) {
					return;
				}

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Name", GUILayout.Width(150f));
				string newName = EditorGUILayout.TextField(Animation.name);
				if(newName != Animation.name) {
					AssetDatabase.RenameAsset(AssetDatabase.GetAssetPath(Animation), newName);
				}
				EditorGUILayout.EndHorizontal();

				Animation.Character.Inspector();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Frames: " + Animation.TotalFrames, GUILayout.Width(100f));
				EditorGUILayout.LabelField("Time: " + Animation.TotalTime.ToString("F3") + "s", GUILayout.Width(100f));
				EditorGUILayout.LabelField("Time/Frame: " + Animation.FrameTime.ToString("F3") + "s" + " (" + (1f/Animation.FrameTime).ToString("F1") + "Hz)", GUILayout.Width(175f));
				EditorGUILayout.LabelField("Preview:", GUILayout.Width(50f), GUILayout.Height(20f)); 
				Animation.Preview = EditorGUILayout.Toggle(Animation.Preview, GUILayout.Width(20f), GUILayout.Height(20f));
				EditorGUILayout.LabelField("Timescale:", GUILayout.Width(65f), GUILayout.Height(20f)); 
				Animation.Timescale = EditorGUILayout.FloatField(Animation.Timescale, GUILayout.Width(30f), GUILayout.Height(20f));
				EditorGUILayout.EndHorizontal();

				Utility.SetGUIColor(Utility.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					EditorGUILayout.BeginHorizontal();
					if(Animation.Playing) {
						if(Utility.GUIButton("||", Color.red, Color.black, TextAnchor.MiddleCenter, 20f, 20f)) {
							Animation.Stop();
						}
					} else {
						if(Utility.GUIButton("|>", Color.green, Color.black, TextAnchor.MiddleCenter, 20f, 20f)) {
							Animation.Play();
						}
					}
					if(Utility.GUIButton("<", Color.grey, Color.white, TextAnchor.MiddleCenter, 20f, 20f)) {
						Animation.LoadPreviousFrame();
					}
					if(Utility.GUIButton(">", Color.grey, Color.white, TextAnchor.MiddleCenter, 20f, 20f)) {
						Animation.LoadNextFrame();
					}
					BVHAnimation.BVHFrame frame = Animation.GetFrame(EditorGUILayout.IntSlider(Animation.CurrentFrame.Index, 1, Animation.TotalFrames, GUILayout.Width(440f)));
					if(Animation.CurrentFrame != frame) {
						Animation.PlayTime = frame.Timestamp;
						Animation.LoadFrame(frame);
					}
					EditorGUILayout.LabelField(Animation.CurrentFrame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
					EditorGUILayout.EndHorizontal();

				}

				Animation.PhaseFunction.Inspector();

				Animation.StyleFunction.Inspector();

				Utility.SetGUIColor(Utility.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(Utility.Orange);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField("Trajectory");
					}

					Animation.ForwardOrientation = EditorGUILayout.Vector3Field("Forward Orientation", Animation.ForwardOrientation);
				}

				Utility.SetGUIColor(Utility.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(Utility.Orange);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField("Export");
					}

					if(Utility.GUIButton("Skeleton", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						Animation.ExportSkeleton(Animation.Character.GetRoot(), null);
					}

					if(Utility.GUIButton("Data", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						//Animation.ExportData(Animation.Character.GetRoot(), null);
					}
					
				}

				//if(Utility.GUIButton("Compute Foot Contacts", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					//ComputeFootContacts(); //TODO
				//}

			}

		}
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
		AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
	}

}
