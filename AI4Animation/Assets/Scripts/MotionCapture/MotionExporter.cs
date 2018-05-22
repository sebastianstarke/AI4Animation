#if UNITY_EDITOR
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
	public int BatchSize = 10;
	public bool[] Export = new bool[0];
	public MotionData[] Animations = new MotionData[0];
	public StyleFilter[] StyleFilters = new StyleFilter[0];

    private bool Exporting = false;

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

	[MenuItem ("Addons/Motion Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporter));
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

	private IEnumerator ExportLabels() {
        Exporting = true;

		string name = "Labels";
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
		file.WriteLine(index + " " + "Sequence"); index += 1;
		file.WriteLine(index + " " + "Frame"); index += 1;
		file.WriteLine(index + " " + "Timestamp"); index += 1;
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
		for(int i=1; i<=12; i++) {
			file.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectorySpeed"+i); index += 1;
			for(int j=1; j<=StyleFilters.Length; j++) {
				file.WriteLine(index + " " + StyleFilters[j-1].Name + i); index += 1;
			}
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

		string name = "Data";
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

		int sequence = 0;
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
					//editor.VisualiseHeightMap(true);
					//editor.VisualiseMotion(true);
					for(int m=1; m<=2; m++) {
						for(int s=0; s<editor.Data.Sequences.Length; s++) {

							MotionData.Sequence.Interval[] intervals = editor.Data.Sequences[s].GetIntervals();
							for(int interval=0; interval<intervals.Length; interval++) {
								sequence += 1;
								float start = editor.Data.GetFrame(intervals[interval].Start).Timestamp;
								float end = editor.Data.GetFrame(intervals[interval].End).Timestamp;
								for(float t=start; t<=end; t+=1f/Framerate) {
									string line = string.Empty;

									if(m==1) {
										editor.SetMirror(false);
									} else {
										editor.SetMirror(true);
									}
									editor.LoadFrame(t);
									MotionState state = editor.GetState();

									line += sequence + Separator;
									line += state.Index + Separator;
									line += state.Timestamp + Separator;

									//Bone data
									for(int k=0; k<state.BoneTransformations.Length; k++) {
										//Position
										line += FormatVector3(state.BoneTransformations[k].GetPosition().GetRelativePositionTo(state.Root));

										//Rotation
										line += FormatVector3(state.BoneTransformations[k].GetForward().GetRelativeDirectionTo(state.Root));
										line += FormatVector3(state.BoneTransformations[k].GetUp().GetRelativeDirectionTo(state.Root));

										//Bone Velocity
										line += FormatVector3(state.BoneVelocities[k].GetRelativeDirectionTo(state.Root));
									}
									
									//Trajectory data
									for(int k=0; k<12; k++) {
										Vector3 position = state.Trajectory.Points[k].GetPosition().GetRelativePositionTo(state.Root);
										Vector3 direction = state.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(state.Root);
										Vector3 velocity = state.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(state.Root);
										float speed = state.Trajectory.Points[k].GetSpeed();
										line += FormatValue(position.x);
										line += FormatValue(position.z);
										line += FormatValue(direction.x);
										line += FormatValue(direction.z);
										line += FormatValue(velocity.x);
										line += FormatValue(velocity.z);
										line += FormatValue(speed);
										line += FormatArray(FilterStyle(state.Trajectory.Points[k].Styles));
									}

									//Height map
									//for(int k=0; k<state.HeightMap.Points.Length; k++) {
									//	float distance = Vector3.Distance(state.HeightMap.Points[k], state.HeightMap.Pivot.GetPosition());
									//	line += FormatValue(distance);
									//}

									//Depth map
									/*
									for(int k=0; k<state.DepthMap.Points.Length; k++) {
										float distance = Vector3.Distance(state.DepthMap.Points[k], state.DepthMap.Pivot.GetPosition());
										line += FormatValue(distance);
									}
									*/

									//Root motion
									line += FormatVector3(state.RootMotion);

									/*
									//Past postures
									for(int k=0; k<6; k++) {
										for(int p=0; p<state.PastBoneTransformations[k].Length; p++) {
											//Position
											line += FormatVector3(state.PastBoneTransformations[k][p].GetPosition().GetRelativePositionTo(state.Root));

											//Rotation
											line += FormatVector3(state.PastBoneTransformations[k][p].GetForward().GetRelativeDirectionTo(state.Root));
											line += FormatVector3(state.PastBoneTransformations[k][p].GetUp().GetRelativeDirectionTo(state.Root));

											//Bone Velocity
											line += FormatVector3(state.PastBoneVelocities[k][p].GetRelativeDirectionTo(state.Root));
										}
									}
									*/

									//Agilities
									//line += FormatArray(state.Agilities);

									//Phase
									/*
									PhaseEditor phase = FindObjectOfType<PhaseEditor>();
									float previous = 0f;
									float current = 0f;
									if(m==1) {
										previous = phase.RegularPhase[Mathf.Max(0, editor.GetState().Index-2)];
										current = phase.RegularPhase[Mathf.Max(0, editor.GetState().Index-1)];
									} else {
										previous = phase.InversePhase[Mathf.Max(0, editor.GetState().Index-2)];
										current = phase.InversePhase[Mathf.Max(0, editor.GetState().Index-1)];
									}
									line += FormatValue(current);
									line += FormatValue(GetPhaseUpdate(previous, current));
									*/

									//Finish
									line = line.Remove(line.Length-1);
									line = line.Replace(",",".");
									file.WriteLine(line);

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
        
		file.Close();

        Exporting = false;
	}

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