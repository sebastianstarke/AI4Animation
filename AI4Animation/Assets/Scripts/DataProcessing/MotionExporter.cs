#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;
using DeepLearning;

public class MotionExporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public int Framerate = 60;
	public int BatchSize = 60;

	public bool Mirror = true;
	public string[] Styles = new string[0];

	public MotionEditor Editor = null;
	public bool[] Export = new bool[0];

    private bool Exporting = false;
	private float Generating = 0f;
	private float Processing = 0f;
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

		Editor = GameObject.FindObjectOfType<MotionEditor>();

		if(Editor == null) {
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
					EditorGUILayout.LabelField("No Motion Editor found in scene.");
				}
			}
		} else {
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

						EditorGUILayout.LabelField("Processing");
						EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Processing * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Transparent(0.75f));

						EditorGUILayout.LabelField("Writing");
						EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Writing * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Transparent(0.75f));

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
		if(Editor == null) {
			Debug.Log("No editor found.");
		} else {
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
			for(int i=0; i<Editor.GetCurrentFile().Data.Source.Bones.Length; i++) {
				if(Editor.GetCurrentFile().Data.Source.Bones[i].Active) {
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "PositionX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "PositionY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "PositionZ"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "ForwardX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "ForwardY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "ForwardZ"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "UpX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "UpY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "UpZ"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "VelocityX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "VelocityY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "VelocityZ"); index += 1;
				}
			}
			for(int i=1; i<=7; i++) {
				for(int j=0; j<Styles.Length; j++) {
					file.WriteLine(index + " " + Styles[j] + "Phase" + i); index += 1;
					file.WriteLine(index + " " + Styles[j] + "Phase" + i); index += 1;
				}
			}

			yield return new WaitForSeconds(0f);

			file.Close();

			Exporting = false;
		}
	}

	private IEnumerator ExportOutputLabels() {
		if(Editor == null) {
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
			for(int i=0; i<Editor.GetCurrentFile().Data.Source.Bones.Length; i++) {
				if(Editor.GetCurrentFile().Data.Source.Bones[i].Active) {
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "PositionX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "PositionY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "PositionZ"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "ForwardX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "ForwardY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "ForwardZ"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "UpX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "UpY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "UpZ"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "VelocityX"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "VelocityY"); index += 1;
					file.WriteLine(index + " " + Editor.GetActor().Bones[i].GetName() + "VelocityZ"); index += 1;
				}
			}
			file.WriteLine(index + " " + "PhaseUpdate"); index += 1;

			yield return new WaitForSeconds(0f);

			file.Close();

			Exporting = false;
		}
	}

	private IEnumerator ExportData() {
		if(Editor == null) {
			Debug.Log("No editor found.");
		} else {
			Exporting = true;

			Generating = 0f;
			Processing = 0f;
			Writing = 0f;
			
			List<List<State>> capture = new List<List<State>>();

			int items = 0;
			int files = Editor.Files.Length;

			//Generating
			int filesToGenerate = 0;
			for(int j=0; j<files; j++) {
				if(Editor.Files[j].Data.Export) {
					filesToGenerate += 1;
				}
			}
			for(int i=0; i<files; i++) {
				if(Editor.Files[i].Data.Export) {
					Editor.LoadFile(Editor.Files[i]);
					for(int m=1; m<=(Mirror ? 2 : 1); m++) {
						if(m==1) {
							Editor.SetMirror(false);
						}
						if(m==2) {
							Editor.SetMirror(true);
						}
						for(int s=0; s<Editor.GetCurrentFile().Data.Sequences.Length; s++) {
							List<State> states = new List<State>();
							float start = Editor.GetCurrentFile().Data.GetFrame(Editor.GetCurrentFile().Data.Sequences[s].Start).Timestamp;
							float end = Editor.GetCurrentFile().Data.GetFrame(Editor.GetCurrentFile().Data.Sequences[s].End).Timestamp;

							for(float t=start; t<=end; t+=1f/Framerate) {
								Editor.LoadFrame(t);
								states.Add(new State(Editor));

								items += 1;
								if(items == BatchSize) {
									items = 0;
									yield return new WaitForSeconds(0f);
								}
							}
							capture.Add(states);
						}
					}
					Generating += 1f / (float)filesToGenerate;
				}
			}

			//Processing
			Data data = new Data();
			for(int i=0; i<capture.Count; i++) {
				for(int j=0; j<capture[i].Count-1; j++) {
					State current = capture[i][j];
					State next = capture[i][j+1];
					Sample sample = new Sample();

					//Input
					for(int k=0; k<12; k++) {
						Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
						float[] state = Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles);
						float[] signal = Filter(ref current.Trajectory.Points[k].Signals, ref current.Trajectory.Styles, ref Styles);
						sample.Input.Add(position.x);
						sample.Input.Add(position.z);
						sample.Input.Add(direction.x);
						sample.Input.Add(direction.z);
						sample.Input.Add(velocity.x);
						sample.Input.Add(velocity.z);
						sample.Input.AddRange(state);
						sample.Input.AddRange(signal);
					}
					for(int k=0; k<current.Posture.Length; k++) {
						Vector3 position = current.Posture[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 forward = current.Posture[k].GetForward().GetRelativeDirectionTo(current.Root);
						Vector3 up = current.Posture[k].GetUp().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = current.Velocities[k].GetRelativeDirectionTo(current.Root);
						sample.Input.Add(position.x);
						sample.Input.Add(position.y);
						sample.Input.Add(position.z);
						sample.Input.Add(forward.x);
						sample.Input.Add(forward.y);
						sample.Input.Add(forward.z);
						sample.Input.Add(up.x);
						sample.Input.Add(up.y);
						sample.Input.Add(up.z);
						sample.Input.Add(velocity.x);
						sample.Input.Add(velocity.y);
						sample.Input.Add(velocity.z);
					}
					for(int k=0; k<7; k++) {
						sample.Input.AddRange(Utility.StylePhase(Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles), current.Trajectory.Points[k].Phase));
					}

					//Output
					for(int k=6; k<12; k++) {
						Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = next.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
						float[] state = Filter(ref next.Trajectory.Points[k].Styles, ref next.Trajectory.Styles, ref Styles);
						sample.Output.Add(position.x);
						sample.Output.Add(position.z);
						sample.Output.Add(direction.x);
						sample.Output.Add(direction.z);
						sample.Output.Add(velocity.x);
						sample.Output.Add(velocity.z);
						sample.Output.AddRange(state);
					}
					for(int k=0; k<next.Posture.Length; k++) {
						Vector3 position = next.Posture[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 forward = next.Posture[k].GetForward().GetRelativeDirectionTo(current.Root);
						Vector3 up = next.Posture[k].GetUp().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = next.Velocities[k].GetRelativeDirectionTo(current.Root);
						sample.Output.Add(position.x);
						sample.Output.Add(position.y);
						sample.Output.Add(position.z);
						sample.Output.Add(forward.x);
						sample.Output.Add(forward.y);
						sample.Output.Add(forward.z);
						sample.Output.Add(up.x);
						sample.Output.Add(up.y);
						sample.Output.Add(up.z);
						sample.Output.Add(velocity.x);
						sample.Output.Add(velocity.y);
						sample.Output.Add(velocity.z);
					}
					sample.Output.Add(Utility.GetLinearPhaseUpdate(current.Trajectory.Points[6].Phase, next.Trajectory.Points[6].Phase));

					data.Samples.Add(sample);

					items += 1;
					if(items == BatchSize) {
						Processing = (float)(i+1) / (float)capture.Count;
						items = 0;
						yield return new WaitForSeconds(0f);
					}
				}
			}
			for(int i=0; i<data.Samples[0].Input.Count; i++) {
				data.XMean.Add(0f);
				data.XStd.Add(1f);
			}
			for(int i=0; i<data.Samples[0].Output.Count; i++) {
				data.YMean.Add(0f);
				data.YStd.Add(1f);
			}
			Processing = 1f;

			//Writing
			StreamWriter input = CreateFile("Input");
			for(int i=0; i<data.Samples.Count; i++) {
				WriteLine(input, data.Samples[i].Input);
				items += 1;
				if(items == BatchSize) {
					Writing = 0f + 0.45f * (float)(i+1) / (float)data.Samples.Count;
					items = 0;
					yield return new WaitForSeconds(0f);
				}
			}
			input.Close();
			StreamWriter output = CreateFile("Output");
			for(int i=0; i<data.Samples.Count; i++) {
				WriteLine(input, data.Samples[i].Output);
				items += 1;
				if(items == BatchSize) {
					Writing = 0.45f + 0.45f * (float)(i+1) / (float)data.Samples.Count;
					items = 0;
					yield return new WaitForSeconds(0f);
				}
			}
			output.Close();
			StreamWriter normInput = CreateFile("InputNorm");
			WriteLine(normInput, data.XMean);
			WriteLine(normInput, data.XStd);
			normInput.Close();
			StreamWriter normOutput = CreateFile("OutputNorm");
			WriteLine(normOutput, data.YMean);
			WriteLine(normOutput, data.YStd);
			normOutput.Close();
			Writing = 1f;

			Exporting = false;
			yield return new WaitForSeconds(0f);
		}
	}

	private float[] Filter(ref float[] values, ref string[] from, ref string[] to) {
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

	private void WriteLine(StreamWriter stream, List<float> values) {
		string line = string.Empty;
		for(int i=0; i<values.Count; i++) {
			line += values[i].ToString(Accuracy) + Separator;
		}
		line = line.Remove(line.Length-1);
		line = line.Replace(",",".");
		stream.WriteLine(line);
	}

	public class Data {
		public List<Sample> Samples = new List<Sample>();
		public List<float> XMean = new List<float>();
		public List<float> XStd = new List<float>();
		public List<float> YMean = new List<float>();
		public List<float> YStd = new List<float>();
	}

	public class Sample {
		public List<float> Input = new List<float>();
		public List<float> Output = new List<float>();
	}

	public class State {
		public float Timestamp;
		public Matrix4x4 Root;
		public Matrix4x4[] Posture;
		public Vector3[] Velocities;
		public Trajectory Trajectory;

		public State(MotionEditor editor) {
			MotionEditor.File file = editor.GetCurrentFile();
			Frame frame = editor.GetCurrentFrame();

			Timestamp = frame.Timestamp;
			Root = frame.GetRootTransformation(editor.Mirror);
			Posture = frame.GetBoneTransformations(editor.Mirror);
			Velocities = frame.GetBoneVelocities(editor.Mirror);
			Trajectory = ((TrajectoryModule)file.Data.GetModule(Module.TYPE.Trajectory)).GetTrajectory(frame, editor.Mirror);
		}
	}

}
#endif