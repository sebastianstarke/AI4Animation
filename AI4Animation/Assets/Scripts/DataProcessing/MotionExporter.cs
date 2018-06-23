#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class MotionExporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public int Framerate = 60;
	public int BatchSize = 10;

	public bool Mirror = true;

	public MotionEditor[] Editors = new MotionEditor[0];
	public bool[] Export = new bool[0];
	public StyleFilter[] StyleFilters = new StyleFilter[0];

    private bool Exporting = false;
	private float Generating = 0f;
	private float Writing = 0f;

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Data Processing/Motion Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporter));
		Scroll = Vector3.zero;
		((MotionExporter)Window).Load();
	}
	
	public void OnInspectorUpdate() {
		Repaint();
	}

	public void Load() {
		Editors = GameObject.FindObjectsOfType<MotionEditor>();
		Export = new bool[Editors.Length];
		for(int i=0; i<Export.Length; i++) {
			Export[i] = true;
		}
		StyleFilters = new StyleFilter[0];
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
					if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
						Load();
					}
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

					if(Utility.GUIButton("Export JSON", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(ExportJSON());
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
				Mirror = EditorGUILayout.Toggle("Mirror", Mirror);

				using(new EditorGUILayout.VerticalScope ("Box")) {
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

					for(int i=0; i<Editors.Length; i++) {

						if(Exporting) {
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
							EditorGUILayout.ObjectField(Editors[i], typeof(MotionEditor), true);
							EditorGUILayout.EndHorizontal();
						}
					}
					
				}
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private StreamWriter CreateFile(string name) {
		string filename = string.Empty;
		string folder = Application.dataPath + "/../../Export/";
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

	private IEnumerator ExportJSON() {
		Exporting = true;

		for(int e=0; e<Editors.Length; e++) {
			if(Export[e]) {
				MotionEditor editor = Editors[e];

				for(int i=0; i<editor.GetFiles().Length; i++) {
					if(editor.Files[i].Data.Export) {
						StreamWriter file = CreateFile(editor.Files[i].Data.Name);
						file.Write(JsonUtility.ToJson(editor.Files[i].Data));
						file.Close();
					}
				}
			}
		}

		Exporting = false;
		yield return new WaitForSeconds(0f);
	}

	private IEnumerator ExportInputLabels() {
		MotionEditor editor = Editors[0];

		Exporting = true;
		
		StreamWriter file = CreateFile("InputLabels");

		StyleModule styleModule = editor.GetFile().Data.GetModule(Module.TYPE.Style) == null ? null : (StyleModule)editor.GetFile().Data.GetModule(Module.TYPE.Style);
		PhaseModule phaseModule = editor.GetFile().Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)editor.GetFile().Data.GetModule(Module.TYPE.Phase);
		//ContactModule contactModule = editor.GetFile().Data.GetModule(Module.TYPE.Contact) == null ? null : (ContactModule)editor.GetFile().Data.GetModule(Module.TYPE.Contact);
	
		int index = 0;
		for(int i=1; i<=12; i++) {
			file.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityX"+i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityZ"+i); index += 1;
			file.WriteLine(index + " " + "TrajectorySpeed"+i); index += 1;
			//for(int j=1; j<=StyleFilters.Length; j++) {
			//	file.WriteLine(index + " " + StyleFilters[j-1].Name + i); index += 1;
			//}
			if(styleModule != null) {
				//for(int j=1; j<=styleModule.Functions.Length; j++) {
				for(int j=1; j<=4; j++) {
					file.WriteLine(index + " " + styleModule.Functions[j-1].Name + i); index += 1;
				}
			}
		}
		for(int i=0; i<editor.GetFile().Data.Source.Bones.Length; i++) {
			if(editor.GetFile().Data.Source.Bones[i].Active) {
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "PositionX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "PositionY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "PositionZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "ForwardX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "ForwardY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "ForwardZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "UpX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "UpY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "UpZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "VelocityX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "VelocityY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "VelocityZ"+(i+1)); index += 1;
			}
		}

		if(phaseModule != null) {
			file.WriteLine(index + " " + "PhaseX"); index += 1;
			file.WriteLine(index + " " + "PhaseY"); index += 1;
			file.WriteLine(index + " " + "PhaseUpdateX"); index += 1;
			file.WriteLine(index + " " + "PhaseUpdateY"); index += 1;
			for(int j=1; j<=4; j++) {
				file.WriteLine(index + " " + styleModule.Functions[j-1].Name + "X"); index += 1;
				file.WriteLine(index + " " + styleModule.Functions[j-1].Name + "Y"); index += 1;
			}
		}

		/*
		if(contactModule != null) {
			for(int i=0; i<contactModule.Functions.Length; i++) {
				file.WriteLine(index + " " + "Contact" + editor.GetFile().Data.Source.Bones[contactModule.Functions[i].Sensor].Name); index += 1;
			}
		}
		*/

		yield return new WaitForSeconds(0f);

		file.Close();

		Exporting = false;
	}

	private IEnumerator ExportOutputLabels() {
		MotionEditor editor = Editors[0];

		if(editor == null) {
			Debug.Log("No editor found.");
		} else {
			Exporting = true;

			StreamWriter file = CreateFile("OutputLabels");

			//StyleModule styleModule = editor.GetFile().Data.GetModule(Module.TYPE.Style) == null ? null : (StyleModule)editor.GetFile().Data.GetModule(Module.TYPE.Style);
			PhaseModule phaseModule = editor.GetFile().Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)editor.GetFile().Data.GetModule(Module.TYPE.Phase);
			//ContactModule contactModule = editor.GetFile().Data.GetModule(Module.TYPE.Contact) == null ? null : (ContactModule)editor.GetFile().Data.GetModule(Module.TYPE.Contact);

			int index = 0;
			for(int i=7; i<=12; i++) {
				file.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryVelocityX"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryVelocityZ"+i); index += 1;
			}
			for(int i=0; i<editor.GetFile().Data.Source.Bones.Length; i++) {
				if(editor.GetFile().Data.Source.Bones[i].Active) {
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "PositionX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "PositionY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "PositionZ"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "ForwardX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "ForwardY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "ForwardZ"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "UpX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "UpY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "UpZ"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "VelocityX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "VelocityY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetFile().Data.Source.Bones[i].Name + "VelocityZ"+(i+1)); index += 1;
				}
			}
			file.WriteLine(index + " " + "RootMotionX"); index += 1;
			file.WriteLine(index + " " + "RootMotionY"); index += 1;
			file.WriteLine(index + " " + "RootMotionZ"); index += 1;

			if(phaseModule != null) {
				file.WriteLine(index + " " + "PhaseUpdate"); index += 1;
			}

			/*
			if(contactModule != null) {
				for(int i=0; i<contactModule.Functions.Length; i++) {
					file.WriteLine(index + " " + "Contact" + editor.GetFile().Data.Source.Bones[contactModule.Functions[i].Sensor].Name); index += 1;
				}
			}
			*/

			yield return new WaitForSeconds(0f);

			file.Close();

			Exporting = false;
		}
	}

	private IEnumerator ExportData() {
        Exporting = true;

		StreamWriter input = CreateFile("Input");
		StreamWriter output = CreateFile("Output");

		for(int e=0; e<Editors.Length; e++) {
			if(Export[e]) {
				MotionEditor editor = Editors[e];
				int items = 0;
				for(int i=0; i<editor.GetFiles().Length; i++) {
					if(editor.Files[i].Data.Export) {
						editor.LoadFile(editor.Files[i]);

						//StyleModule styleModule = editor.GetFile().Data.GetModule(Module.TYPE.Style) == null ? null : (StyleModule)editor.GetFile().Data.GetModule(Module.TYPE.Style);
						PhaseModule phaseModule = editor.GetFile().Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)editor.GetFile().Data.GetModule(Module.TYPE.Phase);
						//ContactModule contactModule = editor.GetFile().Data.GetModule(Module.TYPE.Contact) == null ? null : (ContactModule)editor.GetFile().Data.GetModule(Module.TYPE.Contact);

						for(int m=1; m<=(Mirror ? 2 : 1); m++) {
							if(m==1) {
								editor.SetMirror(false);
							}
							if(m==2) {
								editor.SetMirror(true);
							}
							for(int s=0; s<editor.GetFile().Data.Sequences.Length; s++) {
								MotionData.Sequence.Interval[] intervals = editor.GetFile().Data.Sequences[s].GetIntervals();
								for(int interval=0; interval<intervals.Length; interval++) {
									Generating = 0f;
									Writing = 0f;

									List<State> states = new List<State>();
									float start = editor.GetFile().Data.GetFrame(intervals[interval].Start).Timestamp;
									float end = editor.GetFile().Data.GetFrame(intervals[interval].End).Timestamp;

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
										State previous = states[state-1];
										State next = states[state+1];
										State current = states[state];
										editor.LoadFrame(current);

										//Input
										string inputLine = string.Empty;
										for(int k=0; k<12; k++) {
											Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
											Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
											Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
											float speed = current.Trajectory.Points[k].GetSpeed();
											//float[] style = FilterStyle(current.Trajectory.Points[k].Styles);
											float[] style = current.Trajectory.Points[k].Styles;
											ArrayExtensions.Shrink(ref style);
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
										if(phaseModule != null) {
											float previousPhase = phaseModule.GetPhase(editor.GetFile().Data.GetFrame(previous.Index), editor.ShowMirror);
											float currentPhase = phaseModule.GetPhase(editor.GetFile().Data.GetFrame(current.Index), editor.ShowMirror);
											inputLine += FormatVector2(Utility.GetCirclePhase(currentPhase));
											inputLine += FormatVector2(Utility.GetCirclePhaseUpdate(previousPhase, currentPhase));
											float[] style = current.Trajectory.Points[6].Styles;
											ArrayExtensions.Shrink(ref style);
											style = Utility.StylePhase(style, phaseModule.GetPhase(editor.GetFile().Data.GetFrame(current.Index), editor.ShowMirror));
											inputLine += FormatArray(style);
										}
										/*
										if(contactModule != null) {
											for(int c=0; c<contactModule.Functions.Length; c++) {
												bool contact = contactModule.Functions[c].HasContact(editor.GetFile().Data.GetFrame(current.Index), editor.ShowMirror);
												float value = contact ? 1f : 0f;
												inputLine += FormatValue(value);
											}
										}
										*/
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
										if(phaseModule != null) {
											float currentPhase = phaseModule.GetPhase(editor.GetFile().Data.GetFrame(current.Index), editor.ShowMirror);
											float nextPhase = phaseModule.GetPhase(editor.GetFile().Data.GetFrame(next.Index), editor.ShowMirror);
											outputLine += FormatValue(Utility.GetLinearPhaseUpdate(currentPhase, nextPhase));
										}
										/*
										if(contactModule != null) {
											for(int c=0; c<contactModule.Functions.Length; c++) {
												bool contact = contactModule.Functions[c].HasContact(editor.GetFile().Data.GetFrame(next.Index), editor.ShowMirror);
												float value = contact ? 1f : 0f;
												outputLine += FormatValue(value);
											}
										}
										*/
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
		}
		input.Close();
		output.Close();
        Exporting = false;
	}

	/*
	private float[] FilterStyle(float[] style) {
		if(StyleFilters.Length == 0) {
			return style;
		} else {
			float[] filter = new float[StyleFilters.Length];
			for(int i=0; i<StyleFilters.Length; i++) {
				filter[i] = 0f;
				for(int j=0; j<StyleFilters[i].Indices.Length; j++) {
					filter[i] += style[StyleFilters[i].Indices[j]];
				}
			}
			return filter;
		}
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

	private string FormatVector2(Vector2 vector) {
		return vector.x.ToString(Accuracy) + Separator + vector.y.ToString(Accuracy) + Separator;
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

	[System.Serializable]
	public class StyleFilter {
		public string Name;
		public int[] Indices;

		public StyleFilter(string name, int[] indices) {
			Name = name;
			Indices = indices;
		}
	}

}
#endif