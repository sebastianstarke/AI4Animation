#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class MotionTools : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public bool[] Active = new bool[0];
	public MotionData[] Data = new MotionData[0];

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Addons/Motion Tools")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionTools));
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
					EditorGUILayout.LabelField("Tools");
				}

				if(Utility.GUIButton("Verify Data", UltiDraw.DarkGrey, UltiDraw.White)) {
					VerifyData();
				}

				if(Utility.GUIButton("Examine Data", UltiDraw.DarkGrey, UltiDraw.White)) {
					ExamineData();
				}

				if(Utility.GUIButton("Process Data", UltiDraw.DarkGrey, UltiDraw.White)) {
					ProcessData();
				}

				//if(Utility.GUIButton("Print Velocity Profiles", UltiDraw.DarkGrey, UltiDraw.White)) {
				//	PrintVelocityProfiles();
				//}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
					for(int i=0; i<Active.Length; i++) {
						Active[i] = true;
					}
				}
				if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
					for(int i=0; i<Active.Length; i++) {
						Active[i] = false;
					}
				}
				EditorGUILayout.EndHorizontal();

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45f));
					LoadDirectory(EditorGUILayout.TextField(Directory));
					EditorGUILayout.EndHorizontal();

					for(int i=0; i<Data.Length; i++) {
						if(Active[i]) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Active[i] = EditorGUILayout.Toggle(Active[i], GUILayout.Width(20f));
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
			Active = new bool[0];
			string path = "Assets/"+Directory;
			if(AssetDatabase.IsValidFolder(path)) {
				string[] files = AssetDatabase.FindAssets("t:MotionData", new string[1]{path});
				Data = new MotionData[files.Length];
				Active = new bool[files.Length];
				for(int i=0; i<files.Length; i++) {
					Data[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(files[i]), typeof(MotionData));
					Active[i] = true;
				}
			}
		}
	}

	private void VerifyData() {
		int errors = 0;

		//Default Values
		float unitScale = Data[0].UnitScale;
		MotionData.Axis mirrorAxis = Data[0].MirrorAxis;
		LayerMask groundMask = Data[0].GroundMask;
		LayerMask objectMask = Data[0].ObjectMask;
		//

		for(int i=0; i<Data.Length; i++) {
			if(Active[i]) {
				for(int f=0; f<Data[i].GetTotalFrames(); f++) {
					float style = 0f;
					for(int s=0; s<Data[i].Frames[f].StyleValues.Length; s++) {
						style += Data[i].Frames[f].StyleValues[s];
					}
					if(style != 1f) {
						Debug.Log("One-hot failed in file " + Data[i].name + " at frame " + (f+1) + "!");
						errors += 1;
					}
				}

				if(Data[i].UnitScale != unitScale) {
					errors += 1;
				}
				if(Data[i].MirrorAxis != mirrorAxis) {
					errors += 1;
				}
				if(Data[i].GroundMask != groundMask) {
					errors += 1;
				}
				if(Data[i].ObjectMask != objectMask) {
					errors += 1;
				}
			}
		}
		Debug.Log("Errors: " + errors);
	}

	private void ExamineData() {
		int sequences = 0;
		int frames = 0;
		int[] styles = new int[Data[0].Styles.Length];
		for(int i=0; i<Data.Length; i++) {
			if(Active[i]) {
				for(int m=1; m<=2; m++) {
					for(int s=0; s<Data[i].Sequences.Length; s++) {

						MotionData.Sequence.Interval[] intervals = Data[i].Sequences[s].GetIntervals();
						for(int interval=0; interval<intervals.Length; interval++) {
							sequences += 1;
							for(int f=intervals[interval].Start; f<=intervals[interval].End; f++) {
								frames += 1;
								for(int index=0; index<Data[i].Frames[f].StyleValues.Length; index++) {
									if(Data[i].Frames[f].StyleFlags[index]) {
										styles[index] += 1;
									}
								}
							}
						}
						
					}
				}
			}
		}

		Debug.Log("Sequences: " + sequences);
		Debug.Log("Frames: " + frames);
		Debug.Log("Time: " + (float)frames/(float)Data[0].Framerate+"s");
		for(int i=0; i<styles.Length; i++) {
			Debug.Log(Data[0].Styles[i] + " -> " + (float)styles[i] / (float)frames + "%" + " (" + styles[i] + " frames; " + (float)styles[i]/(float)Data[0].Framerate + "s)");
		}
	}

	private void ProcessData() {
        for(int i=0; i<Data.Length; i++) {
        	if(Active[i]) {
				//Data[i].HeightMapSize = 0.25f;
				//Data[i].DepthMapResolution = 20;
				//Data[i].DepthMapSize = 10f;
				//Data[i].DepthMapDistance = 10f;
				for(int s=0; s<Data[i].Sequences.Length; s++) {
					//Trot
					Data[i].Sequences[s].SetStyleCopies(3, 10);
					Data[i].Sequences[s].SetTransitionCopies(3, 10);

					//Canter
					Data[i].Sequences[s].SetStyleCopies(4, 2);
					Data[i].Sequences[s].SetTransitionCopies(4, 2);

					//Jump
					Data[i].Sequences[s].SetStyleCopies(5, 10);
					Data[i].Sequences[s].SetTransitionCopies(5, 10);

					//Sit
					Data[i].Sequences[s].SetStyleCopies(6, 0);
					Data[i].Sequences[s].SetTransitionCopies(6, 5);

					//Stand
					Data[i].Sequences[s].SetStyleCopies(7, 0);
					Data[i].Sequences[s].SetTransitionCopies(7, 10);

					//Lie
					Data[i].Sequences[s].SetStyleCopies(8, 0);
					Data[i].Sequences[s].SetTransitionCopies(8, 10);
				}
             	EditorUtility.SetDirty(Data[i]);
            }
		}
		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();
	}

	/*
	private void PrintVelocityProfiles() {
		if(Data.Length == 0) {
			return;
		}
		List<float>[] profiles = new List<float>[Data[0].Styles.Length];
		for(int i=0; i<profiles.Length; i++) {
			profiles[i] = new List<float>();
		}
		for(int i=0; i<Data.Length; i++) {
			if(Active[i]) {
				for(int m=1; m<=2; m++) {
					for(int s=0; s<Data[i].Sequences.Length; s++) {
						//for(int e=0; e<Data[i].Sequences[s].Export; e++) {
							int start = Data[i].Sequences[s].Start;
							int end = Data[i].Sequences[s].End;
							for(int f=start; f<=end; f++) {
								MotionData.Frame frame = Data[i].GetFrame(f);
								for(int v=0; v<frame.StyleValues.Length; v++) {
									if(frame.StyleValues[v] > 0.5f) {
										profiles[v].Add(frame.GetSpeed(m==2));
									}
								}
							}
						//}
					}
				}
			}
		}

		for(int i=0; i<profiles.Length; i++) {
			ExportFile(Data[0].Styles[i], FormatArray(profiles[i].ToArray()));
			Debug.Log(Data[0].Styles[i] + " - " + Utility.ComputeMean(profiles[i].ToArray()));
		}
	}
	*/

	private void ExportFile(string name, string text) {
		string filename = string.Empty;
		string folder = Application.dataPath + "/../../../Export/";
		if(!File.Exists(folder+name+".txt")) {
			filename = folder+name;
		} else {
			int i = 1;
			while(File.Exists(folder+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = folder+name+" ("+i+")";
		}
		StreamWriter data = File.CreateText(filename+".txt");
		data.Write(text);
		data.Close();
	}

	private string FormatArray(float[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			format += array[i].ToString(Accuracy) + Separator;
		}
		return format;
	}

}
#endif