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
			
			List<State[]> capture = new List<State[]>();

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
							capture.Add(states.ToArray());
						}
					}
					Generating += 1f / (float)filesToGenerate;
				}
			}

			//Processing
			Data X = new Data();
			Data Y = new Data();
			for(int i=0; i<capture.Count; i++) {
				for(int j=0; j<capture[i].Length-1; j++) {
					State current = capture[i][j];
					State next = capture[i][j+1];

					//Input
					X.Allocate();
					for(int k=0; k<12; k++) {
						Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
						float[] state = Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles);
						float[] signal = Filter(ref current.Trajectory.Points[k].Signals, ref current.Trajectory.Styles, ref Styles);
						X.Feed(position.x, Data.ID.Standard);
						X.Feed(position.z, Data.ID.Standard);
						X.Feed(direction.x, Data.ID.Standard);
						X.Feed(direction.z, Data.ID.Standard);
						X.Feed(velocity.x, Data.ID.Standard);
						X.Feed(velocity.z, Data.ID.Standard);
						X.Feed(state, Data.ID.OnOff);
						X.Feed(signal, Data.ID.OnOff);
					}
					for(int k=0; k<current.Posture.Length; k++) {
						Vector3 position = current.Posture[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 forward = current.Posture[k].GetForward().GetRelativeDirectionTo(current.Root);
						Vector3 up = current.Posture[k].GetUp().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = current.Velocities[k].GetRelativeDirectionTo(current.Root);
						X.Feed(position.x, Data.ID.Standard);
						X.Feed(position.y, Data.ID.Standard);
						X.Feed(position.z, Data.ID.Standard);
						X.Feed(forward.x, Data.ID.Standard);
						X.Feed(forward.y, Data.ID.Standard);
						X.Feed(forward.z, Data.ID.Standard);
						X.Feed(up.x, Data.ID.Standard);
						X.Feed(up.y, Data.ID.Standard);
						X.Feed(up.z, Data.ID.Standard);
						X.Feed(velocity.x, Data.ID.Standard);
						X.Feed(velocity.y, Data.ID.Standard);
						X.Feed(velocity.z, Data.ID.Standard);
					}
					for(int k=0; k<7; k++) {
						X.Feed(Utility.StylePhase(Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles), current.Trajectory.Points[k].Phase), Data.ID.Ignore);
					}
					X.Store();
					//

					//Output
					Y.Allocate();
					for(int k=6; k<12; k++) {
						Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = next.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
						float[] state = Filter(ref next.Trajectory.Points[k].Styles, ref next.Trajectory.Styles, ref Styles);
						Y.Feed(position.x, Data.ID.Standard);
						Y.Feed(position.z, Data.ID.Standard);
						Y.Feed(direction.x, Data.ID.Standard);
						Y.Feed(direction.z, Data.ID.Standard);
						Y.Feed(velocity.x, Data.ID.Standard);
						Y.Feed(velocity.z, Data.ID.Standard);
						Y.Feed(state, Data.ID.OnOff);
					}
					for(int k=0; k<next.Posture.Length; k++) {
						Vector3 position = next.Posture[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 forward = next.Posture[k].GetForward().GetRelativeDirectionTo(current.Root);
						Vector3 up = next.Posture[k].GetUp().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = next.Velocities[k].GetRelativeDirectionTo(current.Root);
						Y.Feed(position.x, Data.ID.Standard);
						Y.Feed(position.y, Data.ID.Standard);
						Y.Feed(position.z, Data.ID.Standard);
						Y.Feed(forward.x, Data.ID.Standard);
						Y.Feed(forward.y, Data.ID.Standard);
						Y.Feed(forward.z, Data.ID.Standard);
						Y.Feed(up.x, Data.ID.Standard);
						Y.Feed(up.y, Data.ID.Standard);
						Y.Feed(up.z, Data.ID.Standard);
						Y.Feed(velocity.x, Data.ID.Standard);
						Y.Feed(velocity.y, Data.ID.Standard);
						Y.Feed(velocity.z, Data.ID.Standard);
					}
					Y.Feed(Utility.GetLinearPhaseUpdate(current.Trajectory.Points[6].Phase, next.Trajectory.Points[6].Phase), Data.ID.Standard);
					Y.Store();
					//

					items += 1;
					if(items == BatchSize) {
						Processing = (float)(i+1) / (float)capture.Count;
						items = 0;
						yield return new WaitForSeconds(0f);
					}
				}
			}
			X.Finalise();
			Y.Finalise();

			Processing = 1f;

			//Writing
			StreamWriter input = CreateFile("Input");
			for(int i=0; i<X.Samples.Count; i++) {
				WriteLine(input, X.Samples[i]);
				items += 1;
				if(items == BatchSize) {
					Writing = 0f + 0.45f * (float)(i+1) / (float)X.Samples.Count;
					items = 0;
					yield return new WaitForSeconds(0f);
				}
			}
			input.Close();
			StreamWriter output = CreateFile("Output");
			for(int i=0; i<Y.Samples.Count; i++) {
				WriteLine(output, Y.Samples[i]);
				items += 1;
				if(items == BatchSize) {
					Writing = 0.45f + 0.45f * (float)(i+1) / (float)Y.Samples.Count;
					items = 0;
					yield return new WaitForSeconds(0f);
				}
			}
			output.Close();
			StreamWriter normInput = CreateFile("InputNorm");
			WriteLine(normInput, X.Mean);
			WriteLine(normInput, X.Std);
			normInput.Close();
			StreamWriter normOutput = CreateFile("OutputNorm");
			WriteLine(normOutput, Y.Mean);
			WriteLine(normOutput, Y.Std);
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

	private void WriteLine(StreamWriter stream, float[] values) {
		string line = string.Empty;
		for(int i=0; i<values.Length; i++) {
			line += values[i].ToString(Accuracy) + Separator;
		}
		line = line.Remove(line.Length-1);
		line = line.Replace(",",".");
		stream.WriteLine(line);
	}

	public class Data {
		public enum ID {Standard, OnOff, Ignore}
		public List<float[]> Samples = new List<float[]>();
		public ID[] Types;
		public float[] Mean;
		public float[] Std;

		private List<float> _sample = new List<float>();
		private List<ID> _types = new List<ID>();

		public void Allocate() {
			_sample.Clear();
			_types.Clear();
		}

		public void Feed(float value, ID type) {
			_sample.Add(value);
			_types.Add(type);
		}

		public void Feed(float[] values, ID type) {
			_sample.AddRange(values);
			for(int i=0; i<values.Length; i++) {
				_types.Add(type);
			}
		}

		public void Store() {
			Samples.Add(_sample.ToArray());
			Types = _types.ToArray();
		}

		public void Finalise() {
			Mean = GetMean();
			Std = GetStd();
		}

		public float[] GetMean() {
			float[] mean = new float[Types.Length];
			for(int i=0; i<mean.Length; i++) {
				switch(Types[i]) {
					case ID.Standard:
						mean[i] = ComputeMean(i);
					break;
					case ID.OnOff:
						mean[i] = 0.5f;
					break;
					case ID.Ignore:
						mean[i] = 0f;
					break;
				}
			}
			return mean;
		}

		public float[] GetStd() {
			float[] std = new float[Types.Length];
			for(int i=0; i<std.Length; i++) {
				switch(Types[i]) {
					case ID.Standard:
						std[i] = ComputeStd(i);
					break;
					case ID.OnOff:
						std[i] = 0.5f;
					break;
					case ID.Ignore:
						std[i] = 1f;
					break;
				}
			}
			return std;
		}

		public float ComputeMean(int dim) {
			float mean = 0f;
			for(int i=0; i<Samples.Count; i++) {
				mean += Samples[i][dim];
			}
			return mean / Samples.Count;
		}

		public float ComputeStd(int dim) {
			float mean = ComputeMean(dim);
			float sum = 0f;
			for(int i=0; i<Samples.Count; i++) {
				sum += Mathf.Pow(Samples[i][dim] - mean, 2f);
			}
			return Mathf.Sqrt(sum / Samples.Count);
		}
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