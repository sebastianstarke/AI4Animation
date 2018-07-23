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

	public bool WriteData = true;
	public bool WriteNorm = true;

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
				//for(int j=0; j<Styles.Length; j++) {
				//	file.WriteLine(index + " " + Styles[j] + "State" + i); index += 1;
				//}
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
			
			//for(int i=1; i<=7; i++) {
			//	for(int j=0; j<Styles.Length; j++) {
			//		file.WriteLine(index + " " + Styles[j] + "State" + i); index += 1;
			//	}
			//}

			for(int i=1; i<=7; i++) {
				for(int j=0; j<Styles.Length; j++) {
					file.WriteLine(index + " " + Styles[j] + "Phase" + i); index += 1;
					file.WriteLine(index + " " + Styles[j] + "Phase" + i); index += 1;
				}
			}

			//for(int j=0; j<Styles.Length; j++) {
			//	file.WriteLine(index + " " + Styles[j] + "Phase"); index += 1;
			//	file.WriteLine(index + " " + Styles[j] + "Phase"); index += 1;
			//}

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
				//for(int j=0; j<Styles.Length; j++) {
				//	file.WriteLine(index + " " + Styles[j] + "State" + i); index += 1;
				//}
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
			for(int j=0; j<Styles.Length; j++) {
				file.WriteLine(index + " " + Styles[j] + "State"); index += 1;
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
			Data X = new Data(WriteData ? CreateFile("Input") : null, WriteNorm ? CreateFile("InputNorm") : null);
			Data Y = new Data(WriteData ? CreateFile("Output") : null, WriteNorm ? CreateFile("OutputNorm") : null);
			for(int i=0; i<capture.Count; i++) {
				for(int j=0; j<capture[i].Length-1; j++) {
					State current = capture[i][j];
					State next = capture[i][j+1];

					//Input
					for(int k=0; k<12; k++) {
						Vector3 position = current.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 direction = current.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = current.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
						//float[] state = Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles);
						float[] signal = Filter(ref current.Trajectory.Points[k].Signals, ref current.Trajectory.Styles, ref Styles);
						X.Feed(position.x, Data.ID.Standard);
						X.Feed(position.z, Data.ID.Standard);
						X.Feed(direction.x, Data.ID.Standard);
						X.Feed(direction.z, Data.ID.Standard);
						X.Feed(velocity.x, Data.ID.Standard);
						X.Feed(velocity.z, Data.ID.Standard);
						//X.Feed(state, Data.ID.OnOff);
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
					//for(int k=0; k<7; k++) {
					//	X.Feed(Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles), Data.ID.OnOff);
					//}
					for(int k=0; k<7; k++) {
						X.Feed(Utility.StylePhase(Filter(ref current.Trajectory.Points[k].Styles, ref current.Trajectory.Styles, ref Styles), current.Trajectory.Points[k].Phase), Data.ID.Ignore);
					}
					//X.Feed(Utility.StylePhase(Filter(ref current.Trajectory.Points[6].Styles, ref current.Trajectory.Styles, ref Styles), current.Trajectory.Points[6].Phase), Data.ID.Ignore);
					X.Store();
					//

					//Output
					for(int k=6; k<12; k++) {
						Vector3 position = next.Trajectory.Points[k].GetPosition().GetRelativePositionTo(current.Root);
						Vector3 direction = next.Trajectory.Points[k].GetDirection().GetRelativeDirectionTo(current.Root);
						Vector3 velocity = next.Trajectory.Points[k].GetVelocity().GetRelativeDirectionTo(current.Root);
						//float[] state = Filter(ref next.Trajectory.Points[k].Styles, ref next.Trajectory.Styles, ref Styles);
						Y.Feed(position.x, Data.ID.Standard);
						Y.Feed(position.z, Data.ID.Standard);
						Y.Feed(direction.x, Data.ID.Standard);
						Y.Feed(direction.z, Data.ID.Standard);
						Y.Feed(velocity.x, Data.ID.Standard);
						Y.Feed(velocity.z, Data.ID.Standard);
						//Y.Feed(state, Data.ID.OnOff);
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
					Y.Feed(Filter(ref next.Trajectory.Points[6].Styles, ref next.Trajectory.Styles, ref Styles), Data.ID.OnOff);
					Y.Feed(Utility.GetLinearPhaseUpdate(current.Trajectory.Points[6].Phase, next.Trajectory.Points[6].Phase), Data.ID.Standard);
					Y.Store();
					//

					items += 1;
					if(items == BatchSize) {
						Processing = (float)(i+1) / (float)capture.Count + 1f / (float)capture.Count * j / capture[i].Length;
						items = 0;
						yield return new WaitForSeconds(0f);
					}
				}
			}
			X.Finish();
			Y.Finish();
			Processing = 1f;
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

	public class Data {
		public StreamWriter File, Norm;
		public enum ID {Standard, OnOff, Ignore}

		public RunningStatistics[] Statistics = null;

		private List<float> Values = new List<float>();
		private List<ID> Types = new List<ID>();
		private int Dim = 0;

		public Data(StreamWriter file, StreamWriter norm) {
			File = file;
			Norm = norm;
		}

		public void Feed(float value, ID type) {
			Dim += 1;
			if(Values.Count < Dim) {
				Values.Add(value);
			} else {
				Values[Dim-1] = value;
			}
			if(Types.Count < Dim) {
				Types.Add(type);
			} else {
				Types[Dim-1] = type;
			}
		}

		public void Feed(float[] values, ID type) {
			for(int i=0; i<values.Length; i++) {
				Feed(values[i], type);
			}
		}

		public void Store() {
			if(Norm != null) {
				if(Statistics == null) {
					Statistics = new RunningStatistics[Values.Count];
					for(int i=0; i<Statistics.Length; i++) {
						Statistics[i] = new RunningStatistics();
					}
				}
				for(int i=0; i<Values.Count; i++) {
					switch(Types[i]) {
						case ID.Standard:		//Ground Truth
						Statistics[i].Add(Values[i]);
						break;
						case ID.OnOff:			//Mean 0.5 Std 0.5
						Statistics[i].Add(1f);
						Statistics[i].Add(0f);
						break;
						case ID.Ignore:			//Mean 0.0 Std 0.0
						Statistics[i].Add(0f);
						break;
					}
				}
			}

			if(File != null) {
				string line = string.Empty;
				for(int i=0; i<Values.Count; i++) {
					line += Values[i].ToString(Accuracy) + Separator;
				}
				line = line.Remove(line.Length-1);
				line = line.Replace(",",".");
				File.WriteLine(line);
			}

			Dim = 0;
		}

		public void Finish() {
			Values.Clear();
			Types.Clear();
			Dim = 0;

			if(File != null) {
				File.Close();
			}

			if(Norm != null) {
				string mean = string.Empty;
				for(int i=0; i<Statistics.Length; i++) {
					mean += Statistics[i].Mean().ToString(Accuracy) + Separator;
				}
				mean = mean.Remove(mean.Length-1);
				mean = mean.Replace(",",".");
				Norm.WriteLine(mean);

				string std = string.Empty;
				for(int i=0; i<Statistics.Length; i++) {
					std += Statistics[i].Std().ToString(Accuracy) + Separator;
				}
				std = std.Remove(std.Length-1);
				std = std.Replace(",",".");
				Norm.WriteLine(std);

				Norm.Close();
			}
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