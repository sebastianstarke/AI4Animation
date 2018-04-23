#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using UnityEditor.SceneManagement;

public class MotionProcessor : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public bool[] Process = new bool[0];
	public MotionData[] Data = new MotionData[0];

	[MenuItem ("Addons/Motion Processor")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionProcessor));
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
					EditorGUILayout.LabelField("Processor");
				}

				if(Utility.GUIButton("Process Data", UltiDraw.DarkGrey, UltiDraw.White)) {
					ProcessData();
				}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
					for(int i=0; i<Process.Length; i++) {
						Process[i] = true;
					}
				}
				if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
					for(int i=0; i<Process.Length; i++) {
						Process[i] = false;
					}
				}
				EditorGUILayout.EndHorizontal();

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45f));
					LoadDirectory(EditorGUILayout.TextField(Directory));
					EditorGUILayout.EndHorizontal();

					for(int i=0; i<Data.Length; i++) {
						if(Process[i]) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Process[i] = EditorGUILayout.Toggle(Process[i], GUILayout.Width(20f));
							Data[i] = (MotionData)EditorGUILayout.ObjectField(Data[i], typeof(MotionData), true);
							EditorGUILayout.EndHorizontal();
						}
					}
				}

			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory(string directory) {
		if(Directory != directory) {
			Directory = directory;
			Data = new MotionData[0];
			Process = new bool[0];
			string path = "Assets/"+Directory;
			if(AssetDatabase.IsValidFolder(path)) {
				string[] files = AssetDatabase.FindAssets("t:MotionData", new string[1]{path});
				Data = new MotionData[files.Length];
				Process = new bool[files.Length];
				for(int i=0; i<files.Length; i++) {
					Data[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(files[i]), typeof(MotionData));
					Process[i] = true;
				}
			}
		}
	}

	private void ProcessData() {
        for(int i=0; i<Data.Length; i++) {
            if(Process[i]) {
				//Data[i].Sequences[0].Start = 63;
				//Data[i].HeightMapSize = 0.25f;
				//Data[i].GroundMask = LayerMaskExtensions.NamesToMask("Ground");
             	EditorUtility.SetDirty(Data[i]);
            }
		}
		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();
	}

}
#endif