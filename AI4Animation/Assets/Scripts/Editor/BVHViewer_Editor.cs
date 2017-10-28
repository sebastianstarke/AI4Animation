using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(BVHViewer))]
public class BVHViewer_Editor : Editor {

		public BVHViewer Target;

		void Awake() {
			Target = (BVHViewer)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {
			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				using(new EditorGUILayout.VerticalScope ("Box")) {

					using(new EditorGUILayout.VerticalScope ("Box")) {
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Path", GUILayout.Width(30));
						Target.Path = EditorGUILayout.TextField(Target.Path);
						GUI.skin.button.alignment = TextAnchor.MiddleCenter;
						if(GUILayout.Button("O", GUILayout.Width(20))) {
							Target.Path = EditorUtility.OpenFilePanel("BVH Viewer", Target.Path == string.Empty ? Application.dataPath : Target.Path.Substring(0, Target.Path.LastIndexOf("/")), "bvh");
							//if(path.Length != 0) {
								/*
								if(path.Contains("Assets/")) {
									Target.Path = path.Substring(path.IndexOf("Assets/")+7);
								} else {
									Debug.Log("Please specify a path inside the Assets folder.");
								}
								*/
							//	GUI.SetNextControlName("");
							//	GUI.FocusControl("");
							//}
						}
						if(GUILayout.Button("X", GUILayout.Width(20))) {
							Target.Unload();
						}
						EditorGUILayout.EndHorizontal();
					}
					
					if(Utility.GUIButton("Import", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						Target.Load(Target.Path);
					}

				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					if(Application.isPlaying) {
						EditorGUILayout.LabelField("This tool can only be used in edit mode.");
						return;
					}
					if(Target.IsLoaded()) {
						EditorGUILayout.LabelField("Total Frames: " + Target.GetTotalFrames());
						EditorGUILayout.LabelField("Current Frame: " + Target.GetCurrentKeyframe().Index);
						EditorGUILayout.LabelField("Total Time: " + Target.GetTotalTime());
						EditorGUILayout.LabelField("Current Time: " + Target.GetCurrentKeyframe().Timestamp);
						EditorGUILayout.LabelField("Frame Time: " + Target.GetFrameTime());
						Target.SetTimescale(EditorGUILayout.FloatField("Timescale", Target.GetTimescale()));
						ControlKeyframe();
						if(Target.IsPlaying()) {
							if(Utility.GUIButton("Stop", Color.white, Color.grey, TextAnchor.MiddleCenter)) {
								Target.Stop();
							}
						} else {
							if(Utility.GUIButton("Play", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
								Target.Play();
							}
						}
						Target.DrawCapture(EditorGUILayout.Toggle("Show Capture", Target.IsDrawingCapture()));
					} else {
						EditorGUILayout.LabelField("No animation loaded.");
					}
				}
			}
			
		}

		private void ControlKeyframe() {
			int newIndex = EditorGUILayout.IntSlider("Keyframe", Target.GetCurrentKeyframe().Index, 1, Target.GetTotalFrames());
			if(newIndex != Target.GetCurrentKeyframe().Index) {
				BVHViewer.Keyframe frame = Target.GetKeyframe(newIndex);
				Target.SetPlayTime(frame.Timestamp);
				Target.ShowKeyframe(frame.Index);
			}
		}

}