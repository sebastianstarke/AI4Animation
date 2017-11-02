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
		Utility.SetGUIColor(Utility.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			using(new EditorGUILayout.VerticalScope ("Box")) {

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
					Animation = ScriptableObject.CreateInstance<BVHAnimation>().Create(this);
				}
			}

			using(new EditorGUILayout.VerticalScope ("Box")) {

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					Animation = (BVHAnimation)EditorGUILayout.ObjectField("Animation", Animation, typeof(BVHAnimation), false);
				}

				if(Animation == null) {
					return;
				}

				Animation.Character.Inspector();
				Animation.ForwardOrientation = EditorGUILayout.Vector3Field("Forward Orientation", Animation.ForwardOrientation);

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Total Frames: " + Animation.TotalFrames, GUILayout.Width(140f));
				EditorGUILayout.LabelField("Total Time: " + Animation.TotalTime + "s", GUILayout.Width(140f));
				EditorGUILayout.LabelField("Frame Time: " + Animation.FrameTime + "s" + " (" + (1f/Animation.FrameTime).ToString("F1") + "Hz)", GUILayout.Width(200f));
				EditorGUILayout.LabelField("Preview:", GUILayout.Width(50f), GUILayout.Height(20f)); 
				Animation.Preview = EditorGUILayout.Toggle(Animation.Preview, GUILayout.Width(20f), GUILayout.Height(20f));
				EditorGUILayout.EndHorizontal();

				EditorGUILayout.BeginHorizontal();
				if(Animation.Playing) {
					if(Utility.GUIButton("||", Color.grey, Color.white, TextAnchor.MiddleCenter, 20f, 20f)) {
						Animation.Stop();
					}
				} else {
					if(Utility.GUIButton("|>", Color.grey, Color.white, TextAnchor.MiddleCenter, 20f, 20f)) {
						Animation.Play();
					}
				}
				int newIndex = EditorGUILayout.IntSlider(Animation.CurrentFrame.Index, 1, Animation.TotalFrames, GUILayout.Width(470f));
				if(newIndex != Animation.CurrentFrame.Index) {
					BVHAnimation.BVHFrame frame = Animation.GetFrame(newIndex);
					Animation.PlayTime = frame.Timestamp;
					Animation.LoadFrame(frame.Index);
				}
				EditorGUILayout.LabelField(Animation.CurrentFrame.Timestamp.ToString() + "s", GUILayout.Width(100f));
				EditorGUILayout.EndHorizontal();

				Animation.DrawPhaseFunction();

				Animation.DrawStyleFunction();

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Previous Keyframe", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					BVHAnimation.BVHFrame prev = Animation.GetPreviousKeyframe(Animation.CurrentFrame);
					if(prev != null) {
						Animation.LoadFrame(prev);
					}
				}
				if(Utility.GUIButton("Next Keyframe", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					BVHAnimation.BVHFrame next = Animation.GetNextKeyframe(Animation.CurrentFrame);
					if(next != null) {
						Animation.LoadFrame(next);
					}
				}
				EditorGUILayout.EndHorizontal();

				Animation.CurrentFrame.SetKeyframe(EditorGUILayout.Toggle("Keyframe", Animation.CurrentFrame.IsKeyframe));

				if(!Animation.CurrentFrame.IsKeyframe) {
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.Slider("Phase", Animation.CurrentFrame.Phase, 0f, 1f);
					EditorGUI.EndDisabledGroup();
				} else {
					Animation.CurrentFrame.Interpolate(EditorGUILayout.Slider("Phase", Animation.CurrentFrame.Phase, 0f, 1f));
				}

				if(Utility.GUIButton("Export Skeleton", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					Animation.ExportSkeleton(Animation.Character.GetRoot(), null);
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

}
