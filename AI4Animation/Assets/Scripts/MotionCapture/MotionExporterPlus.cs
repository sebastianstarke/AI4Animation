#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class MotionExporterPlus : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public int Framerate = 60;
	public int BatchSize = 10;
	public bool[] Export = new bool[0];
	public MotionData[] Animations = new MotionData[0];
	public StyleFilter[] StyleFilters = new StyleFilter[0];

    private bool Exporting = false;
	private float Generating = 0f;
	private float Writing = 0f;

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[System.Serializable]
	public class StyleFilter {
		public string Name;
		public int[] Indices;

		public StyleFilter(string name, int[] indices) {
			Name = name;
			Indices = indices;
		}
	}

	[MenuItem ("Addons/Motion Exporter Plus")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporterPlus));
		Scroll = Vector3.zero;
	}
	
	public void OnInspectorUpdate() {
		Repaint();
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
					EditorGUILayout.LabelField("Exporter");
				}

                if(!Exporting) {
					EditorGUILayout.BeginHorizontal();
                    if(Utility.GUIButton("Export Input Labels", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StartCoroutine(ExportInputLabels());
                    }
                    if(Utility.GUIButton("Export Output Labels", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StartCoroutine(ExportOutputLabels());
                    }
					EditorGUILayout.EndHorizontal();

                    if(Utility.GUIButton("Export Data", UltiDraw.DarkGrey, UltiDraw.White)) {
                        this.StartCoroutine(ExportData());
                    }

                    EditorGUILayout.BeginHorizontal();
                    if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
                        for(int i=0; i<Export.Length; i++) {
                            Export[i] = true;
                        }
                    }
                    if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
                        for(int i=0; i<Export.Length; i++) {
                            Export[i] = false;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                } else {
					EditorGUILayout.LabelField("Generating");
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Generating * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Transparent(0.75f));

					EditorGUILayout.LabelField("Writing");
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Writing * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.IndianRed.Transparent(0.75f));

                    if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) {
                        this.StopAllCoroutines();
                        Exporting = false;
                    }
                }
				
				Framerate = EditorGUILayout.IntField("Framerate", Framerate);
				BatchSize = Mathf.Max(1, EditorGUILayout.IntField("Batch Size", BatchSize));

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45f));
					LoadDirectory(EditorGUILayout.TextField(Directory));
					EditorGUILayout.EndHorizontal();

					for(int i=0; i<StyleFilters.Length; i++) {
						Utility.SetGUIColor(UltiDraw.Cyan);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();					
							StyleFilters[i].Name = EditorGUILayout.TextField("Name", StyleFilters[i].Name);
							for(int j=0; j<StyleFilters[i].Indices.Length; j++) {
								StyleFilters[i].Indices[j] = EditorGUILayout.IntField("ID", StyleFilters[i].Indices[j]);
							}
							EditorGUILayout.BeginHorizontal();
							if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White)) {
								ArrayExtensions.Expand(ref StyleFilters[i].Indices);
							}
							if(Utility.GUIButton("-", UltiDraw.DarkGrey, UltiDraw.White)) {
								ArrayExtensions.Shrink(ref StyleFilters[i].Indices);
							}
							EditorGUILayout.EndHorizontal();
							if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White)) {
								ArrayExtensions.RemoveAt(ref StyleFilters, i);
							}
						}
					}

					for(int i=0; i<Animations.Length; i++) {

						if(Exporting && Animations[i].name == EditorSceneManager.GetActiveScene().name) {
							Utility.SetGUIColor(UltiDraw.Mustard);
						} else {
							if(Export[i]) {
								Utility.SetGUIColor(UltiDraw.DarkGreen);
							} else {
								Utility.SetGUIColor(UltiDraw.DarkRed);
							}
						}

						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Export[i] = EditorGUILayout.Toggle(Export[i], GUILayout.Width(20f));
							Animations[i] = (MotionData)EditorGUILayout.ObjectField(Animations[i], typeof(MotionData), true);
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
			Export = new bool[0];
			Animations = new MotionData[0];
			StyleFilters = new StyleFilter[0];
			string path = "Assets/"+Directory;
			if(AssetDatabase.IsValidFolder(path)) {
				string[] files = AssetDatabase.FindAssets("t:MotionData", new string[1]{path});
				Export = new bool[files.Length];
				Animations = new MotionData[files.Length];
				for(int i=0; i<files.Length; i++) {
					Export[i] = true;
					Animations[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(files[i]), typeof(MotionData));
				}
				
				StyleFilters = new StyleFilter[Animations[0].Styles.Length];
				for(int i=0; i<StyleFilters.Length; i++) {
					StyleFilters[i] = new StyleFilter(Animations[0].Styles[i], new int[1]{i});
				}
			}
		}
	}

	private StreamWriter CreateFile(string name) {
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
		return File.CreateText(filename+".txt");
	}

	private IEnumerator ExportInputLabels() {
        Exporting = true;

		string name = "InputLabels";
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
		StreamWriter file = File.CreateText(filename+".txt");

		int index = 0;
		for(int i=1; i<=12; i++) {
			file.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
			//file.WriteLine(index + " " + "TrajectoryVelocityX"+i); index += 1;
			//file.WriteLine(index + " " + "TrajectoryVelocityZ"+i); index += 1;
			//file.WriteLine(index + " " + "TrajectorySpeed"+i); index += 1;
			//for(int j=1; j<=StyleFilters.Length; j++) {
			//	file.WriteLine(index + " " + StyleFilters[j-1].Name + i); index += 1;
			//}
		}
		for(int i=0; i<Animations[0].Source.Bones.Length; i++) {
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "PositionX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "PositionY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "PositionZ"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "ForwardX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "ForwardY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "ForwardZ"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "UpX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "UpY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "UpZ"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "VelocityX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "VelocityY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "VelocityZ"+(i+1)); index += 1;
		}
		//for(int i=1; i<=120; i++) {
		//	file.WriteLine(index + " " + "HeightMap"+i); index += 1;
		//}

		//file.WriteLine(index + " " + "RootMotionX"); index += 1;
		//file.WriteLine(index + " " + "RootMotionY"); index += 1;
		//file.WriteLine(index + " " + "RootMotionZ"); index += 1;

		//for(int i=0; i<Animations[0].Source.Bones.Length; i++) {
		//	file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Agility"+(i+1)); index += 1;
		//}

		//file.WriteLine(index + " " + "Phase"); index += 1;
		//file.WriteLine(index + " " + "PhaseUpdate"); index += 1;

		/*
		for(int k=0; k<6; k++) {
			for(int i=0; i<Animations[0].Source.Bones.Length; i++) {
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"PositionX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"PositionY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"PositionZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"ForwardX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"ForwardY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"ForwardZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"UpX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"UpY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"UpZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"VelocityX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"VelocityY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"VelocityZ"+(i+1)); index += 1;
			}
		}
		*/

        yield return new WaitForSeconds(0f);

		file.Close();

        Exporting = false;
	}

	private IEnumerator ExportOutputLabels() {
        Exporting = true;

		string name = "OutputLabels";
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
		StreamWriter file = File.CreateText(filename+".txt");

		int index = 0;
		for(int i=7; i<=12; i++) {
			file.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
			//file.WriteLine(index + " " + "TrajectoryVelocityX"+i); index += 1;
			//file.WriteLine(index + " " + "TrajectoryVelocityZ"+i); index += 1;
			//file.WriteLine(index + " " + "TrajectorySpeed"+i); index += 1;
			//for(int j=1; j<=StyleFilters.Length; j++) {
			//	file.WriteLine(index + " " + StyleFilters[j-1].Name + i); index += 1;
			//}
		}
		for(int i=0; i<Animations[0].Source.Bones.Length; i++) {
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "PositionX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "PositionY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "PositionZ"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "ForwardX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "ForwardY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "ForwardZ"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "UpX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "UpY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "UpZ"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "VelocityX"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "VelocityY"+(i+1)); index += 1;
			file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "VelocityZ"+(i+1)); index += 1;
		}
		//for(int i=1; i<=120; i++) {
		//	file.WriteLine(index + " " + "HeightMap"+i); index += 1;
		//}

		file.WriteLine(index + " " + "RootMotionX"); index += 1;
		file.WriteLine(index + " " + "RootMotionY"); index += 1;
		file.WriteLine(index + " " + "RootMotionZ"); index += 1;

		//for(int i=0; i<Animations[0].Source.Bones.Length; i++) {
		//	file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Agility"+(i+1)); index += 1;
		//}

		//file.WriteLine(index + " " + "Phase"); index += 1;
		//file.WriteLine(index + " " + "PhaseUpdate"); index += 1;

		/*
		for(int k=0; k<6; k++) {
			for(int i=0; i<Animations[0].Source.Bones.Length; i++) {
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"PositionX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"PositionY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"PositionZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"ForwardX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"ForwardY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"ForwardZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"UpX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"UpY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"UpZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"VelocityX"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"VelocityY"+(i+1)); index += 1;
				file.WriteLine(index + " " + Animations[0].Source.Bones[i].Name + "Past"+k+"VelocityZ"+(i+1)); index += 1;
			}
		}
		*/

        yield return new WaitForSeconds(0f);

		file.Close();

        Exporting = false;
	}

	private IEnumerator ExportData() {
        Exporting = true;

		int items = 0;
        for(int i=0; i<Animations.Length; i++) {
            if(Export[i]) {
                EditorSceneManager.OpenScene(AssetDatabase.GetAssetPath(Animations[i].Scene));
				yield return new WaitForSeconds(0f);
                MotionEditor editor = FindObjectOfType<MotionEditor>();
                if(editor == null) {
                    Debug.Log("No motion editor found in scene " + Animations[i].name + ".");
                } else {
					editor.VisualiseTrajectory(true);
					editor.VisualiseVelocities(true);
					for(int m=1; m<=2; m++) {
						if(m==1) {
							editor.SetMirror(false);
						}
						if(m==2) {
							editor.SetMirror(true);
						}

						StreamWriter input = CreateFile(Animations[i].name + "_" + (m==1 ? "Default" : "Mirror") + "_" + "Input");
						StreamWriter output = CreateFile(Animations[i].name + "_" + (m==1 ? "Default" : "Mirror") + "_" + "Output");

						for(int s=0; s<editor.Data.Sequences.Length; s++) {
							MotionData.Sequence.Interval[] intervals = editor.Data.Sequences[s].GetIntervals();
							for(int interval=0; interval<intervals.Length; interval++) {
								Generating = 0f;
								Writing = 0f;

								List<MotionState> states = new List<MotionState>();
								float start = editor.Data.GetFrame(intervals[interval].Start).Timestamp;
								float end = editor.Data.GetFrame(intervals[interval].End).Timestamp;

								for(float t=start; t<=end; t+=1f/Framerate) {
									Generating = (t-start) / (end-start-1f/Framerate);
									editor.LoadFrame(t);
									states.Add(editor.GetState());
									//Spin
									items += 1;
									if(items == BatchSize) {
										items = 0;
										yield return new WaitForSeconds(0f);
									}
								}

								for(int state=1; state<states.Count-1; state++) {
									Writing = (float)(state) / (float)(states.Count-2);
									MotionState previous = states[state-1];
									MotionState next = states[state+1];
									MotionState current = states[state];
									editor.LoadFrame(current);

									//Input
									string inputLine = string.Empty;
									for(int k=0; k<12; k++) {
										Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
										inputLine += FormatValue(position.x);
										inputLine += FormatValue(position.z);
										inputLine += FormatValue(direction.x);
										inputLine += FormatValue(direction.z);
									}
									for(int k=0; k<previous.BoneTransformations.Length; k++) {
										Vector3 position = previous.BoneTransformations[k].GetPosition().GetRelativePositionTo(previous.Root);
										Vector3 forward = previous.BoneTransformations[k].GetForward().GetRelativeDirectionTo(previous.Root);
										Vector3 up = previous.BoneTransformations[k].GetUp().GetRelativeDirectionTo(previous.Root);
										Vector3 velocity = previous.BoneVelocities[k].GetRelativeDirectionTo(previous.Root);
										inputLine += FormatVector3(position);
										inputLine += FormatVector3(forward);
										inputLine += FormatVector3(up);
										inputLine += FormatVector3(velocity / Framerate);
									}
									inputLine = inputLine.Remove(inputLine.Length-1);
									inputLine = inputLine.Replace(",",".");
									input.WriteLine(inputLine);

									//Output
									string outputLine = string.Empty;
									for(int k=6; k<12; k++) {
										Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(next.Root);
										Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(next.Root);
										outputLine += FormatValue(position.x);
										outputLine += FormatValue(position.z);
										outputLine += FormatValue(direction.x);
										outputLine += FormatValue(direction.z);
									}
									for(int k=0; k<current.BoneTransformations.Length; k++) {
										Vector3 position = current.BoneTransformations[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 forward = current.BoneTransformations[k].GetForward().GetRelativeDirectionTo(current.Root);
										Vector3 up = current.BoneTransformations[k].GetUp().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = current.BoneVelocities[k].GetRelativeDirectionTo(current.Root);
										outputLine += FormatVector3(position);
										outputLine += FormatVector3(forward);
										outputLine += FormatVector3(up);
										outputLine += FormatVector3(velocity / Framerate);
									}
									outputLine += FormatVector3(next.RootMotion / Framerate);
									outputLine = outputLine.Remove(outputLine.Length-1);
									outputLine = outputLine.Replace(",",".");
									output.WriteLine(outputLine);

									//Spin
									items += 1;
									if(items == BatchSize) {
										items = 0;
										yield return new WaitForSeconds(0f);
									}
								}
							}
						}
						input.Close();
						output.Close();
					}
                }
       		}
		}

        yield return new WaitForSeconds(0f);


        Exporting = false;
	}

	/*
	private IEnumerator ExportData() {
        Exporting = true;

		StreamWriter input = CreateFile("Input");
		StreamWriter output = CreateFile("Output");

		int items = 0;
        for(int i=0; i<Animations.Length; i++) {
            if(Export[i]) {
                EditorSceneManager.OpenScene(AssetDatabase.GetAssetPath(Animations[i].Scene));
				yield return new WaitForSeconds(0f);
                MotionEditor editor = FindObjectOfType<MotionEditor>();
                if(editor == null) {
                    Debug.Log("No motion editor found in scene " + Animations[i].name + ".");
                } else {
					editor.VisualiseTrajectory(true);
					editor.VisualiseVelocities(true);
					for(int m=1; m<=2; m++) {
						if(m==1) {
							editor.SetMirror(false);
						}
						if(m==2) {
							editor.SetMirror(true);
						}
						for(int s=0; s<editor.Data.Sequences.Length; s++) {
							MotionData.Sequence.Interval[] intervals = editor.Data.Sequences[s].GetIntervals();
							for(int interval=0; interval<intervals.Length; interval++) {
								Generating = 0f;
								Writing = 0f;

								List<MotionState> states = new List<MotionState>();
								float start = editor.Data.GetFrame(intervals[interval].Start).Timestamp;
								float end = editor.Data.GetFrame(intervals[interval].End).Timestamp;

								for(float t=start; t<=end; t+=1f/Framerate) {
									Generating = (t-start) / (end-start-1f/Framerate);
									editor.LoadFrame(t);
									states.Add(editor.GetState());
									//Spin
									items += 1;
									if(items == BatchSize) {
										items = 0;
										yield return new WaitForSeconds(0f);
									}
								}

								for(int state=1; state<states.Count-1; state++) {
									Writing = (float)(state) / (float)(states.Count-2);
									MotionState previous = states[state-1];
									MotionState next = states[state+1];
									MotionState current = states[state];
									editor.LoadFrame(current);

									//Input
									string inputLine = string.Empty;
									for(int k=0; k<12; k++) {
										Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
										float speed = current.Trajectory.Points[k].GetSpeed();
										float[] style = FilterStyle(current.Trajectory.Points[k].Styles);
										inputLine += FormatValue(position.x);
										inputLine += FormatValue(position.z);
										inputLine += FormatValue(direction.x);
										inputLine += FormatValue(direction.z);
										inputLine += FormatValue(velocity.x);
										inputLine += FormatValue(velocity.z);
										inputLine += FormatValue(speed);
										inputLine += FormatArray(style);
									}
									for(int k=0; k<previous.BoneTransformations.Length; k++) {
										Vector3 position = previous.BoneTransformations[k].GetPosition().GetRelativePositionTo(previous.Root);
										Vector3 forward = previous.BoneTransformations[k].GetForward().GetRelativeDirectionTo(previous.Root);
										Vector3 up = previous.BoneTransformations[k].GetUp().GetRelativeDirectionTo(previous.Root);
										Vector3 velocity = previous.BoneVelocities[k].GetRelativeDirectionTo(previous.Root);
										inputLine += FormatVector3(position);
										inputLine += FormatVector3(forward);
										inputLine += FormatVector3(up);
										inputLine += FormatVector3(velocity);
									}
									inputLine = inputLine.Remove(inputLine.Length-1);
									inputLine = inputLine.Replace(",",".");
									input.WriteLine(inputLine);

									//Output
									string outputLine = string.Empty;
									for(int k=6; k<12; k++) {
										Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(next.Root);
										Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(next.Root);
										Vector3 velocity = next.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(next.Root);
										outputLine += FormatValue(position.x);
										outputLine += FormatValue(position.z);
										outputLine += FormatValue(direction.x);
										outputLine += FormatValue(direction.z);
										outputLine += FormatValue(velocity.x);
										outputLine += FormatValue(velocity.z);
									}
									for(int k=0; k<current.BoneTransformations.Length; k++) {
										Vector3 position = current.BoneTransformations[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 forward = current.BoneTransformations[k].GetForward().GetRelativeDirectionTo(current.Root);
										Vector3 up = current.BoneTransformations[k].GetUp().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = current.BoneVelocities[k].GetRelativeDirectionTo(current.Root);
										outputLine += FormatVector3(position);
										outputLine += FormatVector3(forward);
										outputLine += FormatVector3(up);
										outputLine += FormatVector3(velocity);
									}
									outputLine += FormatVector3(next.RootMotion);
									outputLine = outputLine.Remove(outputLine.Length-1);
									outputLine = outputLine.Replace(",",".");
									output.WriteLine(outputLine);

									//Spin
									items += 1;
									if(items == BatchSize) {
										items = 0;
										yield return new WaitForSeconds(0f);
									}
								}
							}
						}
					}
                }
       		}
		}

        yield return new WaitForSeconds(0f);
        
		input.Close();
		output.Close();

        Exporting = false;
	}
	*/

	private float GetPhaseUpdate(float previous, float next) {
		return Mathf.Repeat(((next-previous) + 1f), 1f);
	}

	private float[] FilterStyle(float[] style) {
		float[] filter = new float[StyleFilters.Length];
		for(int i=0; i<StyleFilters.Length; i++) {
			filter[i] = 0f;
			for(int j=0; j<StyleFilters[i].Indices.Length; j++) {
				filter[i] += style[StyleFilters[i].Indices[j]];
			}
		}
		return filter;
	}

	/*
	private float[] NoiseStyle(float[] style, float sigma) {
		float[] noised = new float[style.Length];
		for(int i=0; i<style.Length; i++) {
			noised[i] = style[i] + Utility.GaussianValue(0f, sigma);
		}
		Utility.SoftMax(ref noised);
		return noised;
	}
	*/

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
#endif