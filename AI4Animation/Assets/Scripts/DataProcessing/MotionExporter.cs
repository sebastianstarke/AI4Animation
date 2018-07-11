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

		PhaseModule phaseModule = editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase);
	
		int index = 0;
		for(int i=1; i<=12; i++) {
			file.WriteLine(index + " " + "TrajectoryPositionX" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryPositionZ" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionX" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryDirectionZ" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityX" + i); index += 1;
			file.WriteLine(index + " " + "TrajectoryVelocityZ" + i); index += 1;
			for(int j=0; j<Styles.Length; j++) {
				file.WriteLine(index + " " + Styles[j] + i); index += 1;
			}
		}
		for(int i=0; i<editor.GetCurrentFile().Data.Source.Bones.Length; i++) {
			if(editor.GetCurrentFile().Data.Source.Bones[i].Active) {
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpZ"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityX"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityY"+(i+1)); index += 1;
				file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityZ"+(i+1)); index += 1;
			}
		}

		if(phaseModule != null) {
			for(int i=1; i<=Styles.Length; i++) {
				file.WriteLine(index + " " + Styles[i-1] + "X"); index += 1;
				file.WriteLine(index + " " + Styles[i-1] + "Y"); index += 1;
			}
		}

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

			PhaseModule phaseModule = editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase);

			int index = 0;
			for(int i=7; i<=12; i++) {
				file.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryVelocityX"+i); index += 1;
				file.WriteLine(index + " " + "TrajectoryVelocityZ"+i); index += 1;
				for(int j=0; j<Styles.Length; j++) {
					file.WriteLine(index + " " + Styles[j] + i); index += 1;
				}
			}
			for(int i=0; i<editor.GetCurrentFile().Data.Source.Bones.Length; i++) {
				if(editor.GetCurrentFile().Data.Source.Bones[i].Active) {
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "PositionZ"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "ForwardZ"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "UpZ"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityX"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityY"+(i+1)); index += 1;
					file.WriteLine(index + " " + editor.GetActor().Bones[i].GetName() + "VelocityZ"+(i+1)); index += 1;
				}
			}

			if(phaseModule != null) {
				file.WriteLine(index + " " + "PhaseUpdate"); index += 1;
			}

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

						PhaseModule phaseModule = editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase);

						for(int m=1; m<=(Mirror ? 2 : 1); m++) {
							if(m==1) {
								editor.SetMirror(false);
							}
							if(m==2) {
								editor.SetMirror(true);
							}
							for(int s=0; s<editor.GetCurrentFile().Data.Sequences.Length; s++) {
								MotionData.Sequence.Interval[] intervals = editor.GetCurrentFile().Data.Sequences[s].GetIntervals();
								for(int interval=0; interval<intervals.Length; interval++) {
									Generating = 0f;
									Writing = 0f;

									List<State> states = new List<State>();
									float start = editor.GetCurrentFile().Data.GetFrame(intervals[interval].Start).Timestamp;
									float end = editor.GetCurrentFile().Data.GetFrame(intervals[interval].End).Timestamp;

									for(float t=start; t<=end; t+=1f/Framerate) {
										Generating = (t-start) / (end-start-1f/Framerate);
										editor.LoadFrame(t);
										states.Add(new State(editor));
										//Spin
										items += 1;
										if(items == BatchSize) {
											items = 0;
											yield return new WaitForSeconds(0f);
										}
									}

									for(int state=0; state<states.Count-1; state++) {
										Writing = (float)(state) / (float)(states.Count-2);
										State current = states[state];
										State next = states[state+1];
										editor.LoadFrame(current.Index);

										//Input
										string inputLine = string.Empty;
										for(int k=0; k<12; k++) {
											Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
											Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
											Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
											float[] style = current.Trajectory.Points[k].Styles;
											inputLine += FormatValue(position.x);
											inputLine += FormatValue(position.z);
											inputLine += FormatValue(direction.x);
											inputLine += FormatValue(direction.z);
											inputLine += FormatValue(velocity.x);
											inputLine += FormatValue(velocity.z);
											inputLine += FormatArray(style);
										}
										for(int k=0; k<current.WorldPosture.Length; k++) {
											Vector3 position = current.WorldPosture[k].GetPosition().GetRelativePositionTo(current.Root);
											Vector3 forward = current.WorldPosture[k].GetForward().GetRelativeDirectionTo(current.Root);
											Vector3 up = current.WorldPosture[k].GetUp().GetRelativeDirectionTo(current.Root);
											Vector3 velocity = current.Velocities[k].GetRelativeDirectionTo(current.Root);
											inputLine += FormatVector3(position);
											inputLine += FormatVector3(forward);
											inputLine += FormatVector3(up);
											inputLine += FormatVector3(velocity);
										}
										if(phaseModule != null) {
											float currentPhase = phaseModule.GetPhase(editor.GetCurrentFile().Data.GetFrame(current.Index), editor.Mirror);
											inputLine += FormatArray(Utility.StylePhase(current.Trajectory.Points[6].Styles, currentPhase));
										}

										inputLine = inputLine.Remove(inputLine.Length-1);
										inputLine = inputLine.Replace(",",".");
										input.WriteLine(inputLine);

										//Output
										string outputLine = string.Empty;
										for(int k=6; k<12; k++) {
											Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
											Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
											Vector3 velocity = next.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
											float[] style = next.Trajectory.Points[k].Styles;
											outputLine += FormatValue(position.x);
											outputLine += FormatValue(position.z);
											outputLine += FormatValue(direction.x);
											outputLine += FormatValue(direction.z);
											outputLine += FormatValue(velocity.x);
											outputLine += FormatValue(velocity.z);
											outputLine += FormatArray(style);
										}
										for(int k=0; k<next.WorldPosture.Length; k++) {
											Vector3 position = next.WorldPosture[k].GetPosition().GetRelativePositionTo(current.Root);
											Vector3 forward = next.WorldPosture[k].GetForward().GetRelativeDirectionTo(current.Root);
											Vector3 up = next.WorldPosture[k].GetUp().GetRelativeDirectionTo(current.Root);
											Vector3 velocity = next.Velocities[k].GetRelativeDirectionTo(current.Root);
											outputLine += FormatVector3(position);
											outputLine += FormatVector3(forward);
											outputLine += FormatVector3(up);
											outputLine += FormatVector3(velocity);
										}
										if(phaseModule != null) {
											float currentPhase = phaseModule.GetPhase(editor.GetCurrentFile().Data.GetFrame(current.Index), editor.Mirror);
											float nextPhase = phaseModule.GetPhase(editor.GetCurrentFile().Data.GetFrame(next.Index), editor.Mirror);
											outputLine += FormatValue(Utility.GetLinearPhaseUpdate(currentPhase, nextPhase));
										}

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
		}

		input.Close();
		output.Close();

        Exporting = false;
		yield return new WaitForSeconds(0f);
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

	public class State {
		public int Index;
		public float Timestamp;
		public float Phase;
		public Matrix4x4 Root;
		public Matrix4x4[] LocalPosture;
		public Matrix4x4[] WorldPosture;
		public Vector3[] Velocities;
		public Trajectory Trajectory;

		public State(MotionEditor editor) {
			Index = editor.GetCurrentFrame().Index;
			Timestamp = editor.GetCurrentFrame().Timestamp;
			Phase = ((PhaseModule)editor.GetCurrentFile().Data.GetModule(Module.TYPE.Phase)).GetPhase(editor.GetCurrentFrame(), editor.Mirror);
			Root = editor.GetActor().GetRoot().GetWorldMatrix();
			LocalPosture = editor.GetActor().GetLocalPosture();
			WorldPosture = editor.GetActor().GetWorldPosture();
			Velocities = editor.GetCurrentFrame().GetBoneVelocities(editor.Mirror);
			Trajectory = editor.GetCurrentFrame().GetTrajectory(editor.Mirror);
		}
	}

}
#endif