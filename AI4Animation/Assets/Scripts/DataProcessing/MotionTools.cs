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

	public MotionEditor[] Editors = new MotionEditor[0];
	public bool[] Active = new bool[0];

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Data Processing/Motion Tools")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionTools));
		Scroll = Vector3.zero;
		((MotionTools)Window).Load();
	}

	public void Load() {
		Editors = GameObject.FindObjectsOfType<MotionEditor>();
		Active = new bool[Editors.Length];
		for(int i=0; i<Active.Length; i++) {
			Active[i] = true;
		}
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

				if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
					Load();
				}

				/*
				if(Utility.GUIButton("Examine Data", UltiDraw.DarkGrey, UltiDraw.White)) {
					ExamineData();
				}

				if(Utility.GUIButton("Search Style", UltiDraw.DarkGrey, UltiDraw.White)) {
					SearchStyle();
				}
				*/

				if(Utility.GUIButton("Batch Process", UltiDraw.DarkGrey, UltiDraw.White)) {
					BatchProcess();
				}

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

				for(int i=0; i<Editors.Length; i++) {
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
						EditorGUILayout.ObjectField(Editors[i], typeof(MotionEditor), true);
						EditorGUILayout.EndHorizontal();
					}
				}

			}
		}

		EditorGUILayout.EndScrollView();
	}

	/*
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
								for(int index=0; index<Data[i].GetFrame(f).StyleValues.Length; index++) {
									if(Data[i].GetFrame(f).StyleFlags[index]) {
										styles[index] += 1;
									}
									//if(Data[i].Frames[f].StyleValues[index] > 0f) {
									//	styles[index] += 1;
									//}
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
	*/

	/*
	private void SearchStyle() {
		/*
		int style = System.Array.FindIndex(Data[0].Styles, x => x == "Sit");
		for(int i=0; i<Data.Length; i++) {
			for(int s=0; s<Data[i].Sequences.Length; s++) {
				for(int f=Data[i].Sequences[s].Start; f<=Data[i].Sequences[s].End; f++) {
					if((Data[i].GetFrame(f).IsStyleKey(style) || f==Data[i].Sequences[s].Start) && Data[i].GetFrame(f).StyleFlags[style]) {
						Debug.Log("Style at frame " + f + " in file " + Data[i]);
					}
				}
			}
		}
	}
	*/

	private void BatchProcess() {
        for(int i=0; i<Editors.Length; i++) {
        	if(Active[i]) {
				for(int j=0; j<Editors[i].Files.Length; j++) {
					MotionData data = Editors[i].Files[j].Data;
					/*
					for(int s=0; s<Data[i].Sequences.Length; s++) {
						//Idle
						Data[i].Sequences[s].SetStyleCopies("Idle", 0);
						Data[i].Sequences[s].SetTransitionCopies("Idle", 0);

						//Walk
						Data[i].Sequences[s].SetStyleCopies("Walk", 0);
						Data[i].Sequences[s].SetTransitionCopies("Walk", 0);

						//Pace
						Data[i].Sequences[s].SetStyleCopies("Pace", 0);
						Data[i].Sequences[s].SetTransitionCopies("Pace", 0);

						//Trot
						Data[i].Sequences[s].SetStyleCopies("Trot", 6);
						Data[i].Sequences[s].SetTransitionCopies("Trot", 6);

						//Canter
						Data[i].Sequences[s].SetStyleCopies("Canter", 1);
						Data[i].Sequences[s].SetTransitionCopies("Canter", 1);

						//Jump
						Data[i].Sequences[s].SetStyleCopies("Jump", 9);
						Data[i].Sequences[s].SetTransitionCopies("Jump", 9);

						//Sit
						Data[i].Sequences[s].SetStyleCopies("Sit", 0);
						Data[i].Sequences[s].SetTransitionCopies("Sit", 0);

						//Stand
						Data[i].Sequences[s].SetStyleCopies("Stand", 0);
						Data[i].Sequences[s].SetTransitionCopies("Stand", 5);

						//Lie
						Data[i].Sequences[s].SetStyleCopies("Lie", 0);
						Data[i].Sequences[s].SetTransitionCopies("Lie", 5);
					}
					*/

					//string path = AssetDatabase.GetAssetPath(Data[i]);
					//path = path.Substring(0, path.LastIndexOf(".")) + ".unity";
					//SceneAsset scene = AssetDatabase.LoadAssetAtPath<SceneAsset>(path);
					//Data[i].Scene = scene;
					//Data[i].Sequences[0].SetStart(1);
					//Data[i].Sequences[0].SetEnd(Data[i].GetTotalFrames());
					//Data[i].RootSmoothing = 10;

					//StyleModule styleModule = (StyleModule)data.GetModule(Module.TYPE.Style);
					//styleModule.RemoveStyle("Move");
					//System.Array.Find(styleModule.Functions, x => x.Name == "Move").ToggleStyle(data.GetFirstFrame());

					//Editors[i].LoadFile(Editors[i].Files[j]);

					//data.Scaling = 7.5f;

					//Editors[i].CopyHierarchy();

					//ContactModule contact = (ContactModule)data.GetModule(Module.TYPE.Contact);
					//for(int f=0; f<contact.Functions.Length; f++) {
					//	contact.Functions[f].SetFilterWidth(5);
					//	contact.Functions[f].SetDistanceThreshold(0.025f);
					//	contact.Functions[f].SetVelocityThreshold(1f);
					//}

					//data.Export = true;

					//data.RemoveModule(Module.TYPE.HeightMap);
					//((HeightMapModule)data.GetModule(Module.TYPE.HeightMap)).Size = 1f;

					//data.Repair(Editors[i]);
					((KeypointModule)data.GetModule(Module.TYPE.Keypoint)).Size = 1.5f;

					EditorUtility.SetDirty(data);
				}
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