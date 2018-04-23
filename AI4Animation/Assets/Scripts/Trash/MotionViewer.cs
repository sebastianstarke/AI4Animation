/*
#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using UnityEditor.SceneManagement;

public class MotionViewer : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Path = string.Empty;
	public MotionData Data = null;

	[MenuItem ("Addons/Motion Viewer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionViewer));
		Scroll = Vector3.zero;
	}
	
	void OnGUI() {
		Scroll = EditorGUILayout.BeginScrollView(Scroll);
		
		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Editor");
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
								//LoadFile();
							}
						}
					}
				}

			}
		}

		EditorGUILayout.EndScrollView();
	}

	void OnSceneGUI(SceneView view) {
		UltiDraw.Begin();
		UltiDraw.DrawLine(Vector3.zero, Vector3.one, 10f, Color.red);
		UltiDraw.End();
	}

}
#endif
*/