using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using UnityEditor.SceneManagement;

public class MotionExporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public int Framerate = 60;
	public bool[] Export = new bool[0];
	public SceneAsset[] Animations = new SceneAsset[0];

    private bool Exporting = false;

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Addons/Motion Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporter));
		Scroll = Vector3.zero;
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
					EditorGUILayout.LabelField("Exporter");
				}

                if(!Exporting) {
                    if(Utility.GUIButton("Export Labels", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StartCoroutine(ExportLabels());
                    }
                    if(Utility.GUIButton("Export Data", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StartCoroutine(ExportData());
                    }

                    EditorGUILayout.BeginHorizontal();
                    if(Utility.GUIButton("Enable All", UltiDraw.Grey, UltiDraw.White)) {
                        for(int i=0; i<Export.Length; i++) {
                            Export[i] = true;
                        }
                    }
                    if(Utility.GUIButton("Disable All", UltiDraw.Grey, UltiDraw.White)) {
                        for(int i=0; i<Export.Length; i++) {
                            Export[i] = false;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                } else {
                    if(Utility.GUIButton("Stop", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StopAllCoroutines();
                        Exporting = false;
                    }
                }
				
				Framerate = EditorGUILayout.IntField("Framerate", Framerate);				

				Scroll = EditorGUILayout.BeginScrollView(Scroll);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45f));
					LoadDirectory(EditorGUILayout.TextField(Directory));
					EditorGUILayout.EndHorizontal();

					for(int i=0; i<Animations.Length; i++) {
						if(Export[i]) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Export[i] = EditorGUILayout.Toggle(Export[i], GUILayout.Width(20f));
							Animations[i] = (SceneAsset)EditorGUILayout.ObjectField(Animations[i], typeof(SceneAsset), true);
							EditorGUILayout.EndHorizontal();
						}
					}
					
				}
				EditorGUILayout.EndScrollView();
			}
		}
	}

	private void LoadDirectory(string directory) {
		if(Directory != directory) {
			Directory = directory;
			Animations = new SceneAsset[0];
			Export = new bool[0];
			string path = "Assets/"+Directory;
			if(AssetDatabase.IsValidFolder(path)) {
				string[] files = AssetDatabase.FindAssets("t:SceneAsset", new string[1]{path});
				Animations = new SceneAsset[files.Length];
				Export = new bool[files.Length];
				for(int i=0; i<files.Length; i++) {
					Animations[i] = (SceneAsset)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(files[i]), typeof(SceneAsset));
					Export[i] = true;
				}
			}
		}
	}

	private IEnumerator ExportLabels() {
        Exporting = true;

        yield return new WaitForSeconds(0f);

        Exporting = false;
	}

	private IEnumerator ExportData() {
        Exporting = true;

        for(int i=0; i<Animations.Length; i++) {
            if(Export[i]) {
                EditorSceneManager.OpenScene(AssetDatabase.GetAssetPath(Animations[i]));
                MotionEditor editor = FindObjectOfType<MotionEditor>();
                if(editor == null) {
                    Debug.Log("No motion editor found in scene " + (i+1) + ".");
                } else {
                    float time = editor.Data.GetTotalTime();
                    for(float t=0f; t<=time; t+=1f/Framerate) {
                        editor.LoadFrame(t);
                        yield return new WaitForSeconds(0f);
                    }
                }
            }
        }

        yield return new WaitForSeconds(0f);
        
        Exporting = false;
	}

	private string FormatString(string value) {
		return value + Separator;
	}

	private string FormatValue(float value) {
		return value.ToString(Accuracy) + Separator;
	}

	private string FormatArray(float[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			format += array[i].ToString(Accuracy) + Separator;
		}
		return format;
	}

	private string FormatArray(bool[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			float value = array[i] ? 1f : 0f;
			format += value.ToString(Accuracy) + Separator;
		}
		return format;
	}

	private string FormatVector3(Vector3 vector) {
		return vector.x.ToString(Accuracy) + Separator + vector.y.ToString(Accuracy) + Separator + vector.z.ToString(Accuracy) + Separator;
	}

	private string FormatQuaternion(Quaternion quaternion, bool imaginary, bool real) {
		string output = string.Empty;
		if(imaginary) {
			output += quaternion.x + Separator + quaternion.y + Separator + quaternion.z + Separator;
		}
		if(real) {
			output += quaternion.w + Separator;
		}
		return output;
	}

}