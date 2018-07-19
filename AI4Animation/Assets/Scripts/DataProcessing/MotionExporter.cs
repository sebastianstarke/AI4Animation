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
	public int BatchSize = 60;

	public bool Mirror = true;
	public string[] Styles = new string[0];

	public MotionEditor[] Editors = new MotionEditor[0];
	public bool[] Export = new bool[0];

    private bool Exporting = false;
	private float Generating = 0f;
	private float Writing = 0f;

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Data Processing/Motion Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporter));
		Scroll = Vector3.zero;
	}
	
	public void OnInspectorUpdate() {
		Repaint();
	}

	void OnGUI() {
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Editors = GameObject.FindObjectsOfType<MotionEditor>();
		ArrayExtensions.Resize(ref Export, Editors.Length);

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
				Mirror = EditorGUILayout.Toggle("Mirror", Mirror);
				for(int i=0; i<Styles.Length; i++) {
					Styles[i] = EditorGUILayout.TextField("Style " + (i+1), Styles[i]);
				}
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Add Style", UltiDraw.DarkGrey, UltiDraw.White)) {
					ArrayExtensions.Expand(ref Styles);
				}
				if(Utility.GUIButton("Remove Style", UltiDraw.DarkGrey, UltiDraw.White)) {
					ArrayExtensions.Shrink(ref Styles);
				}
				EditorGUILayout.EndHorizontal();

				using(new EditorGUILayout.VerticalScope ("Box")) {
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

	private IEnumerator ExportInputLabels() {
		MotionEditor editor = Editors[0];

		Exporting = true;
		
		StreamWriter file = CreateFile("InputLabels");
	
		int index = 0;
		for(int i=1; i<=12; i++) {
			file.WriteLine(index + " " + "TrajectoryPositionX" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryPositionZ" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionX" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionZ" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityX" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityZ" + i); index += 1;
			for(int j=0; j<Styles.Length; j++) {
				file.WriteLine(index + " " + Styles[j] + "State" + i); index += 1;
			}
			for(int j=0; j<Styles.Length; j++) {
				file.WriteLine(index + " " + Styles[j] + "Signal" + i); index += 1;
			}
		}
		for(int i=0; i<editor.GetCurrentFile().Data.Source.Bones.Length; i++) {
			if(editor.GetCurrentFile().Data.Source.Bones[i].Active) {
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionX"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionY"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionZ"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardX"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardY"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardZ"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpX"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpY"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpZ"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityX"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityY"); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityZ"); index += 1;
			}
		}

		//for(int i=1; i<=7; i++) {
		//	for(int j=0; j<Styles.Length; j++) {
		//		file.WriteLine(index + " " + Styles[j] + "Value" + i); index += 1;
		//	}
		//}

		for(int i=1; i<=7; i++) {
			for(int j=0; j<Styles.Length; j++) {
				file.WriteLine(index + " " + Styles[j] + "Phase" + i); index += 1;
				file.WriteLine(index + " " + Styles[j] + "Phase" + i); index += 1;
			}
		}

		/*
		for(int k=1; k<=12; k++) {
			for(int i=1; i<=Styles.Length; i++) {
				file.WriteLine(index + " " + Styles[i-1] + "ValueX" + k); index += 1;
				file.WriteLine(index + " " + Styles[i-1] + "ValueY" + k); index += 1;
			}
		}
		*/

		/*
		for(int i=1; i<=Styles.Length; i++) {
			file.WriteLine(index + " " + Styles[i-1] + "ValueX"); index += 1;
			file.WriteLine(index + " " + Styles[i-1] + "ValueY"); index += 1;
		}
	
		for(int i=1; i<=Styles.Length; i++) {
			file.WriteLine(index + " " + Styles[i-1] + "UpdateX"); index += 1;
			file.WriteLine(index + " " + Styles[i-1] + "UpdateY"); index += 1;
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

			int index = 0;
			for(int i=7; i<=12; i++) {
				file.WriteLine(index + " " + "TrajectoryPositionX" + i); index += 1;
				file.WriteLine(index + " " + "TrajectoryPositionZ" + i); index += 1;
				file.WriteLine(index + " " + "TrajectoryDirectionX" + i); index += 1;
				file.WriteLine(index + " " + "TrajectoryDirectionZ" + i); index += 1;
				file.WriteLine(index + " " + "TrajectoryVelocityX" + i); index += 1;
				file.WriteLine(index + " " + "TrajectoryVelocityZ" + i); index += 1;
				for(int j=0; j<Styles.Length; j++) {
					file.WriteLine(index + " " + Styles[j] + "State" + i); index += 1;
				}
			}
			for(int i=0; i<editor.GetCurrentFile().Data.Source.Bones.Length; i++) {
				if(editor.GetCurrentFile().Data.Source.Bones[i].Active) {
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionX"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionY"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionZ"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardX"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardY"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardZ"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpX"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpY"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpZ"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityX"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityY"); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityZ"); index += 1;
				}
			}

			//for(int i=0; i<Styles.Length; i++) {
			//	file.WriteLine(index + " " + Styles[i]); index += 1;
			//}
			file.WriteLine(index + " " + "PhaseUpdate"); index += 1;

			//for(int i=7; i<=12; i++) {
			//	file.WriteLine(index + " " + "PhaseUpdate" + i); index += 1;
			//}

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

						for(int m=1; m<=(Mirror ? 2 : 1); m++) {
							if(m==1) {
								editor.SetMirror(false);
							}
							if(m==2) {
								editor.SetMirror(true);
							}
							for(int s=0; s<editor.GetCurrentFile().Data.Sequences.Length; s++) {
								Generating = 0f;
								Writing = 0f;

								List<State> frames = new List<State>();
								float start = editor.GetCurrentFile().Data.GetFrame(editor.GetCurrentFile().Data.Sequences[s].Start).Timestamp;
								float end = editor.GetCurrentFile().Data.GetFrame(editor.GetCurrentFile().Data.Sequences[s].End).Timestamp;

								for(float t=start; t<=end; t+=1f/Framerate) {
									Generating = (t-start) / (end-start-1f/Framerate);
									editor.LoadFrame(t);
									frames.Add(new State(this, editor));
									//Spin
									items += 1;
									if(items == BatchSize) {
										items = 0;
										yield return new WaitForSeconds(0f);
									}
								}

								for(int frame=0; frame<frames.Count-1; frame++) {
									Writing = (float)(frame) / (float)(frames.Count-2);
									//State previous = frames[frame-1];
									State current = frames[frame];
									State next = frames[frame+1];

									//Input
									string inputLine = string.Empty;
									for(int k=0; k<12; k++) {
										Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
										float[] state = FilterStyle(current.Trajectory.Points[k].Styles, current.Trajectory.Styles, Styles);
										float[] signal = FilterControl(current.Trajectory.Points[k].Signals, current.Trajectory.Styles, Styles);
										inputLine += Format(position.x);
										inputLine += Format(position.z);
										inputLine += Format(direction.x);
										inputLine += Format(direction.z);
										inputLine += Format(velocity.x);
										inputLine += Format(velocity.z);
										inputLine += Format(state);
										inputLine += Format(signal);
									}
									for(int k=0; k<current.Posture.Length; k++) {
										Vector3 position = current.Posture[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 forward = current.Posture[k].GetForward().GetRelativeDirectionTo(current.Root);
										Vector3 up = current.Posture[k].GetUp().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = current.Velocities[k].GetRelativeDirectionTo(current.Root);
										inputLine += Format(position);
										inputLine += Format(forward);
										inputLine += Format(up);
										inputLine += Format(velocity);
									}
									//inputLine += Format(Utility.StylePhase(FilterStyle(current.Trajectory.Points[6].Styles, current.Trajectory.Styles, Styles), current.Trajectory.Points[6].Phase));
									
									//for(int k=0; k<7; k++) {
									///	inputLine += Format(FilterStyle(current.Trajectory.Points[k].Styles, current.Trajectory.Styles, Styles));
									//}
									for(int k=0; k<7; k++) {
										inputLine += Format(Utility.StylePhase(FilterStyle(current.Trajectory.Points[k].Styles, current.Trajectory.Styles, Styles), current.Trajectory.Points[k].Phase));
									}
									
									/*
									float[] previousStyle = FilterStyle(previous.Trajectory.Points[6].Styles, previous.Trajectory.Styles, Styles);
									float[] currentStyle = FilterStyle(current.Trajectory.Points[6].Styles, current.Trajectory.Styles, Styles);
									inputLine += Format(Utility.StylePhase(currentStyle, current.Trajectory.Points[6].Phase));
									inputLine += Format(Utility.StyleUpdatePhase(previousStyle, currentStyle, previous.Trajectory.Points[6].Phase, current.Trajectory.Points[6].Phase));
									*/

									inputLine = inputLine.Remove(inputLine.Length-1);
									inputLine = inputLine.Replace(",",".");
									input.WriteLine(inputLine);

									//Output
									string outputLine = string.Empty;
									for(int k=6; k<12; k++) {
										Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = next.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
										float[] state = FilterStyle(next.Trajectory.Points[k].Styles, next.Trajectory.Styles, Styles);
										outputLine += Format(position.x);
										outputLine += Format(position.z);
										outputLine += Format(direction.x);
										outputLine += Format(direction.z);
										outputLine += Format(velocity.x);
										outputLine += Format(velocity.z);
										outputLine += Format(state);
									}
									for(int k=0; k<next.Posture.Length; k++) {
										Vector3 position = next.Posture[k].GetPosition().GetRelativePositionTo(current.Root);
										Vector3 forward = next.Posture[k].GetForward().GetRelativeDirectionTo(current.Root);
										Vector3 up = next.Posture[k].GetUp().GetRelativeDirectionTo(current.Root);
										Vector3 velocity = next.Velocities[k].GetRelativeDirectionTo(current.Root);
										outputLine += Format(position);
										outputLine += Format(forward);
										outputLine += Format(up);
										outputLine += Format(velocity);
									}

									//outputLine += Format(FilterStyle(next.Trajectory.Points[6].Styles, next.Trajectory.Styles, Styles));
									outputLine += Format(Utility.GetLinearPhaseUpdate(current.Trajectory.Points[6].Phase, next.Trajectory.Points[6].Phase));

									//for(int k=6; k<12; k++) {
									//	outputLine += Format(Utility.GetLinearPhaseUpdate(current.Trajectory.Points[k].Phase, next.Trajectory.Points[k].Phase));
									//}

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

		input.Close();
		output.Close();

        Exporting = false;
		yield return new WaitForSeconds(0f);
	}

	private float[] FilterStyle(float[] values, string[] from, string[] to) {
		float[] filtered = new float[to.Length];
		for(int i=0; i<to.Length; i++) {
			for(int j=0; j<from.Length; j++) {
				if(to[i] == from[j]) {
					filtered[i] = values[j];
				}
			}
		}
		return filtered;
	}

	private float[] FilterControl(float[] values, string[] from, string[] to) {
		float[] filtered = new float[to.Length];
		for(int i=0; i<to.Length; i++) {
			for(int j=0; j<from.Length; j++) {
				if(to[i] == from[j]) {
					filtered[i] = values[j];
				}
			}
		}
		return filtered;
	}

	private string Format(string value) {
		return value + Separator;
	}

	private string Format(float value) {
		return value.ToString(Accuracy) + Separator;
	}

	private string Format(float[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			format += array[i].ToString(Accuracy) + Separator;
		}
		return format;
	}

	private string Format(bool[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			float value = array[i] ? 1f : 0f;
			format += value.ToString(Accuracy) + Separator;
		}
		return format;
	}

	private string Format(Vector2 vector) {
		return vector.x.ToString(Accuracy) + Separator + vector.y.ToString(Accuracy) + Separator;
	}

	private string Format(Vector3 vector) {
		return vector.x.ToString(Accuracy) + Separator + vector.y.ToString(Accuracy) + Separator + vector.z.ToString(Accuracy) + Separator;
	}

	private string Format(Quaternion quaternion) {
		return quaternion.x + Separator + quaternion.y + Separator + quaternion.z + Separator + quaternion.w + Separator;
	}

	public class State {
		public float Timestamp;
		public Matrix4x4 Root;
		public Matrix4x4[] Posture;
		public Vector3[] Velocities;
		public Trajectory Trajectory;
		public float[] Control;

		public State(MotionExporter exporter, MotionEditor editor) {
			Timestamp = editor.GetCurrentFrame().Timestamp;
			Root = editor.GetCurrentFrame().GetRootTransformation(editor.Mirror);
			Posture = editor.GetCurrentFrame().GetBoneTransformations(editor.Mirror);
			Velocities = editor.GetCurrentFrame().GetBoneVelocities(editor.Mirror);
			Trajectory = ((TrajectoryModule)editor.GetCurrentFile().Data.GetModule(Module.TYPE.Trajectory)).GetTrajectory(editor.GetCurrentFrame(), editor.Mirror);
			Control = ((StyleModule)editor.GetCurrentFile().Data.GetModule(Module.TYPE.Style)).GetControl(editor.GetCurrentFrame());
		}
	}

}
#endif