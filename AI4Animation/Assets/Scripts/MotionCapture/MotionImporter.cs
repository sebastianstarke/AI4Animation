#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using UnityEditor.SceneManagement;

public class MotionImporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Source = string.Empty;
	public string Destination = string.Empty;
	public bool[] Import = new bool[0];
	public FileInfo[] Files = new FileInfo[0];

	private bool Importing = false;

	[MenuItem ("Addons/Motion Importer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionImporter));
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
					EditorGUILayout.LabelField("Importer");
				}
				
                if(!Importing) {
                    if(Utility.GUIButton("Import Data", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StartCoroutine(ImportFiles());
                    }

                    EditorGUILayout.BeginHorizontal();
                    if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
                        for(int i=0; i<Import.Length; i++) {
                            Import[i] = true;
                        }
                    }
                    if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
                        for(int i=0; i<Import.Length; i++) {
                            Import[i] = false;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                } else {
                    if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) {
                        this.StopAllCoroutines();
                        Importing = false;
                    }
                }

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Source", GUILayout.Width(50));
					LoadDirectory(EditorGUILayout.TextField(Source));
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Source = EditorUtility.OpenFilePanel("Motion Importer", Source == string.Empty ? Application.dataPath : Source.Substring(0, Source.LastIndexOf("/")), "");
						GUI.SetNextControlName("");
						GUI.FocusControl("");
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
					Destination = EditorGUILayout.TextField(Destination);
					EditorGUILayout.EndHorizontal();

					for(int i=0; i<Files.Length; i++) {
						if(Import[i]) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Import[i] = EditorGUILayout.Toggle(Import[i], GUILayout.Width(20f));
							EditorGUILayout.LabelField(Files[i].Name);
							EditorGUILayout.EndHorizontal();
						}
					}
					
				}
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory(string source) {
		if(Source != source) {
			Source = source;
			Import = new bool[0];
			Files = new FileInfo[0];
			if(Directory.Exists(Source)) {
				DirectoryInfo info = new DirectoryInfo(Source);
				Files = info.GetFiles("*.bvh");
				Import = new bool[Files.Length];
				for(int i=0; i<Files.Length; i++) {
					Import[i] = true;
				}
			}
		}
	}

	private IEnumerator ImportFiles() {
        Importing = true;

        for(int i=0; i<Files.Length; i++) {
            if(Import[i]) {
				string scene = MotionEditor.CreateMotionCapture(Application.dataPath + "/" + Destination, Files[i].Name);
                EditorSceneManager.OpenScene(scene);
				yield return new WaitForSeconds(0f);
                MotionEditor editor = FindObjectOfType<MotionEditor>();
                if(editor == null) {
                    Debug.Log("No motion editor found in scene " + scene + ".");
                } else {
					foreach(Actor actor in GameObject.FindObjectsOfType<Actor>()) {
						Utility.Destroy(actor.gameObject);
					}
					editor.LoadFile(Files[i].FullName);
					EditorUtility.SetDirty(editor);
					EditorUtility.SetDirty(editor.Data);
					EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
				}
				yield return new WaitForSeconds(0f);
       		}
		}

        yield return new WaitForSeconds(0f);
        
		Importing = false;
	}

}
#endif