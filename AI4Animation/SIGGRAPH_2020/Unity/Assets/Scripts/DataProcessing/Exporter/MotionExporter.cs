#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Threading;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;

public class MotionExporter : EditorWindow {

	public enum PIPELINE {Basketball, Quadruped};

	[Serializable]
	public class Asset {
		public string GUID = string.Empty;
		public bool Selected = true;
		public bool Exported = false;
	}

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public PIPELINE Pipeline = PIPELINE.Quadruped; 

	public int FrameShifts = 0;
	public int FrameBuffer = 10;
	public bool WriteMirror = true;

	private string Filter = string.Empty;
	private Asset[] Assets = new Asset[0];
	[NonSerialized] private Asset[] Instances = null;

	private static bool Aborting = false;
	private static bool Exporting = false;

	private int Page = 0;
	private int Items = 25;

	private float Progress = 0f;
	private float Performance = 0f;

	private static string Separator = " ";
	private static string Accuracy = "F5";

	private MotionEditor Editor = null;

	[MenuItem ("AI4Animation/Exporter/Motion Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporter));
		Scroll = Vector3.zero;
	}
	
	public void OnInspectorUpdate() {
		Repaint();
	}
	
	public void Refresh() {
		if(Editor == null) {
			Editor = GameObject.FindObjectOfType<MotionEditor>();
		}
		if(Editor != null && Assets.Length != Editor.Assets.Length) {
			Assets = new Asset[Editor.Assets.Length];
			for(int i=0; i<Editor.Assets.Length; i++) {
				Assets[i] = new Asset();
				Assets[i].GUID = Editor.Assets[i];
				Assets[i].Selected = true;
				Assets[i].Exported = false;
			}
			Aborting = false;
			Exporting = false;
			ApplyFilter(string.Empty);
		}
		if(Instances == null) {
			ApplyFilter(string.Empty);
		}
	}

	public void ApplyFilter(string filter) {
		Filter = filter;
		if(Filter == string.Empty) {
			Instances = Assets;
		} else {
			List<Asset> instances = new List<Asset>();
			for(int i=0; i<Assets.Length; i++) {
				if(Utility.GetAssetName(Assets[i].GUID).ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Assets[i]);
				}
			}
			Instances = instances.ToArray();
		}
		LoadPage(1);
	}

	public void LoadPage(int page) {
		Page = Mathf.Clamp(page, 1, GetPages());
	}

	public int GetPages() {
		return Mathf.CeilToInt(Instances.Length/Items)+1;
	}

	public int GetStart() {
		return (Page-1)*Items;
	}

	public int GetEnd() {
		return Mathf.Min(Page*Items, Instances.Length);
	}

	private string GetExportPath() {
		string path = Application.dataPath;
		path = path.Substring(0, path.LastIndexOf("/"));
		path = path.Substring(0, path.LastIndexOf("/"));
		path += "/DeepLearning";
		return path;
	}

	void OnGUI() {
		Refresh();

		if(Editor == null) {
			EditorGUILayout.LabelField("No editor available in scene.");
			return;
		}

		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Motion Exporter");
				}

				Utility.SetGUIColor(UltiDraw.White);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.FloatField("Export Framerate", Editor.TargetFramerate);
					EditorGUILayout.TextField("Export Path", GetExportPath());
					EditorGUI.EndDisabledGroup();
				}

				Pipeline = (PIPELINE)EditorGUILayout.EnumPopup("Pipeline", Pipeline);
				FrameShifts = EditorGUILayout.IntField("Frame Shifts", FrameShifts);
				FrameBuffer = Mathf.Max(1, EditorGUILayout.IntField("Frame Buffer", FrameBuffer));
				WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);

				if(!Exporting) {
					if(Utility.GUIButton("Export Data", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(ExportData());
					}
				} else {
					EditorGUILayout.LabelField("Asset: " + Editor.GetAsset().GetName());
					EditorGUILayout.LabelField("Index: " + (Editor.GetAssetIndex()+1) + " / " + Assets.Length);
					EditorGUILayout.LabelField("Mirror: " + Editor.Mirror);
					EditorGUILayout.LabelField("Frames Per Second: " + Performance.ToString("F3"));
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, (float)(Editor.GetAssetIndex()+1) / (float)Assets.Length * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Opacity(0.75f));
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Progress * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Opacity(0.75f));

					EditorGUI.BeginDisabledGroup(Aborting);
					if(Utility.GUIButton(Aborting ? "Aborting" : "Stop", Aborting ? UltiDraw.Gold : UltiDraw.DarkRed, UltiDraw.White)) {
						Aborting = true;
					}
					EditorGUI.EndDisabledGroup();
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();

						EditorGUILayout.LabelField("Page", GUILayout.Width(40f));
						EditorGUI.BeginChangeCheck();
						int page = EditorGUILayout.IntField(Page, GUILayout.Width(40f));
						if(EditorGUI.EndChangeCheck()) {
							LoadPage(page);
						}
						EditorGUILayout.LabelField("/" + GetPages());
						
						EditorGUILayout.LabelField("Filter", GUILayout.Width(40f));
						EditorGUI.BeginChangeCheck();
						string filter = EditorGUILayout.TextField(Filter, GUILayout.Width(200f));
						if(EditorGUI.EndChangeCheck()) {
							ApplyFilter(filter);
						}

						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							foreach(Asset a in Assets) {
								a.Selected = true;
							}
						}
						if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							foreach(Asset a in Assets) {
								a.Selected = false;
							}
						}
						if(Utility.GUIButton("Current", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							string guid = Utility.GetAssetGUID(Editor.GetAsset());
							foreach(Asset a in Assets) {
								a.Selected = a.GUID == guid;
							}
						}
						EditorGUILayout.EndHorizontal();

						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							LoadPage(Mathf.Max(Page-1, 1));
						}
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							LoadPage(Mathf.Min(Page+1, GetPages()));
						}
						EditorGUILayout.EndHorizontal();
					}
					
					int start = GetStart();
					int end = GetEnd();
					for(int i=start; i<end; i++) {
						if(Instances[i].Exported) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else if(Instances[i].Selected) {
							Utility.SetGUIColor(UltiDraw.Gold);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Instances[i].Selected = EditorGUILayout.Toggle(Instances[i].Selected, GUILayout.Width(20f));
							EditorGUILayout.LabelField(Utility.GetAssetName(Instances[i].GUID));
							EditorGUILayout.EndHorizontal();
						}
					}
				}
			}
		}

		EditorGUILayout.EndScrollView();
	}

	public class Data {
		public StreamWriter File, Norm, Labels;

		public RunningStatistics[] Statistics = null;

		private Queue<float[]> Buffer = new Queue<float[]>();
		private Task Writer = null;

		private float[] Values = new float[0];
		private string[] Names = new string[0];
		private float[] Weights = new float[0];
		private int Dim = 0;

		private bool Finished = false;
		private bool Setup = false;

		public Data(StreamWriter file, StreamWriter norm, StreamWriter labels) {
			File = file;
			Norm = norm;
			Labels = labels;
			Writer = Task.Factory.StartNew(() => WriteData());
		}

		public void Feed(float value, string name, float weight=1f) {
			if(!Setup) {
				ArrayExtensions.Append(ref Values, value);
				ArrayExtensions.Append(ref Names, name);
				ArrayExtensions.Append(ref Weights, weight);
			} else {
				Dim += 1;
				Values[Dim-1] = value;
			}
		}

		public void Feed(float[] values, string name, float weight=1f) {
			for(int i=0; i<values.Length; i++) {
				Feed(values[i], name + (i+1), weight);
			}
		}

		public void Feed(bool[] values, string name, float weight=1f) {
			for(int i=0; i<values.Length; i++) {
				Feed(values[i] ? 1f : 0f, name + (i+1), weight);
			}
		}

		public void Feed(float[,] values, string name, float weight=1f) {
			for(int i=0; i<values.GetLength(0); i++) {
				for(int j=0; j<values.GetLength(1); j++) {
					Feed(values[i,j], name+(i*values.GetLength(1)+j+1), weight);
				}
			}
		}

		public void Feed(bool[,] values, string name, float weight=1f) {
			for(int i=0; i<values.GetLength(0); i++) {
				for(int j=0; j<values.GetLength(1); j++) {
					Feed(values[i,j] ? 1f : 0f, name+(i*values.GetLength(1)+j+1), weight);
				}
			}
		}

		public void Feed(Vector2 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.y, name+"Y", weight);
		}

		public void Feed(Vector3 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.y, name+"Y", weight);
			Feed(value.z, name+"Z", weight);
		}

		public void FeedXY(Vector3 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.y, name+"Y", weight);
		}

		public void FeedXZ(Vector3 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.z, name+"Z", weight);
		}

		public void FeedYZ(Vector3 value, string name, float weight=1f) {
			Feed(value.y, name+"Y", weight);
			Feed(value.z, name+"Z", weight);
		}

		private void WriteData() {
			while(Exporting && (!Finished || Buffer.Count > 0)) {
				if(Buffer.Count > 0) {
					float[] item;
					lock(Buffer) {
						item = Buffer.Dequeue();	
					}
					//Update Mean and Std
					for(int i=0; i<item.Length; i++) {
						Statistics[i].Add(item[i]);
					}
					//Write to File
					File.WriteLine(String.Join(Separator, Array.ConvertAll(item, x => x.ToString(Accuracy))));
				} else {
					Thread.Sleep(1);
				}
			}
		}

		public void Store() {
			if(!Setup) {
				//Setup Mean and Std
				Statistics = new RunningStatistics[Values.Length];
				for(int i=0; i<Statistics.Length; i++) {
					Statistics[i] = new RunningStatistics();
				}

				//Write Labels
				for(int i=0; i<Names.Length; i++) {
					Labels.WriteLine("[" + i + "]" + " " + Names[i]);
				}
				Labels.Close();

				Setup = true;
			}

			//Enqueue Sample
			float[] item = (float[])Values.Clone();
			lock(Buffer) {
				Buffer.Enqueue(item);
			}

			//Reset Running Index
			Dim = 0;
		}

		public void Finish() {
			Finished = true;

			Task.WaitAll(Writer);

			File.Close();

			if(Setup) {
				//Write Mean
				float[] mean = new float[Statistics.Length];
				for(int i=0; i<mean.Length; i++) {
					mean[i] = Statistics[i].Mean();
				}
				Norm.WriteLine(String.Join(Separator, Array.ConvertAll(mean, x => x.ToString(Accuracy))));

				//Write Std
				float[] std = new float[Statistics.Length];
				for(int i=0; i<std.Length; i++) {
					std[i] = Statistics[i].Std();
				}
				std.Replace(0f, 1f);
				Norm.WriteLine(String.Join(Separator, Array.ConvertAll(std, x => x.ToString(Accuracy))));
			}

			Norm.Close();
		}
	}

	private IEnumerator ExportData() {
		if(Editor == null) {
			Debug.Log("No editor found.");
		} else if(!System.IO.Directory.Exists(GetExportPath())) {
			Debug.Log("No export folder found at " + GetExportPath() + ".");
		} else {
			Aborting = false;
			Exporting = true;
			Progress = 0f;

			int sequence = 0;
			int items = 0;
			int samples = 0;
			DateTime timestamp = Utility.GetTimestamp();

			StreamWriter S = CreateFile("Sequences");
			Data X = new Data(CreateFile("Input"), CreateFile("InputNorm"), CreateFile("InputLabels"));
			Data Y = new Data(CreateFile("Output"), CreateFile("OutputNorm"), CreateFile("OutputLabels"));
			StreamWriter CreateFile(string name) {
				return File.CreateText(GetExportPath() + "/" + name + ".txt");
			}

			for(int i=0; i<Assets.Length; i++) {
				Assets[i].Exported = false;
			}
			for(int i=0; i<Assets.Length; i++) {
				if(Aborting) {
					break;
				}
				if(Assets[i].Selected) {
					MotionData data = Editor.LoadData(Assets[i].GUID);
					while(!data.GetScene().isLoaded) {
						Debug.Log("Waiting for scene being loaded...");
						yield return new WaitForSeconds(0f);
					}
					if(!data.Export) {
						Debug.Log("Skipping Asset: " + data.GetName());
						yield return new WaitForSeconds(0f);
						continue;
					}
					for(int m=1; m<=2; m++) {
						if(m==1) {
							Editor.SetMirror(false);
						}
						if(m==2) {
							Editor.SetMirror(true);
						}
						if(!Editor.Mirror || WriteMirror && Editor.Mirror) {
							// Debug.Log("Exporting asset " + data.GetName() + " " + (Editor.Mirror ? "[Mirror]" : "[Default]"));
							for(int shift=0; shift<=FrameShifts; shift++) {
								foreach(Sequence seq in data.Sequences) {
									sequence += 1;
									float start = Editor.CeilToTargetTime(data.GetFrame(seq.Start).Timestamp);
									float end = Editor.FloorToTargetTime(data.GetFrame(seq.End).Timestamp);
									int index = 0;
									while(start + (index+1)/Editor.TargetFramerate + shift/data.Framerate <= end) {
										Editor.SetRandomSeed(Editor.GetCurrentFrame().Index);
										S.WriteLine(sequence.ToString());

										float tCurrent = start + index/Editor.TargetFramerate + shift/data.Framerate;
										float tNext = start + (index+1)/Editor.TargetFramerate + shift/data.Framerate;

										if(Pipeline == PIPELINE.Basketball) {
											BasketballSetup.Export(this, X, Y, tCurrent, tNext);
										}
										if(Pipeline == PIPELINE.Quadruped) {
											QuadrupedSetup.Export(this, X, Y, tCurrent, tNext);
										}

										X.Store();
										Y.Store();

										index += 1;
										Progress = (index/Editor.TargetFramerate) / (end-start);
										items += 1;
										samples += 1;
										if(items >= FrameBuffer) {
											Performance = items / (float)Utility.GetElapsedTime(timestamp);
											timestamp = Utility.GetTimestamp();
											items = 0;
											yield return new WaitForSeconds(0f);
										}
									}
									Progress = 0f;
								}
							}
						}
					}
					Assets[i].Exported = true;
				}
			}
			
			S.Close();
			X.Finish();
			Y.Finish();

			Aborting = false;
			Exporting = false;
			Progress = 0f;
			foreach(Asset a in Assets) {
				a.Exported = false;
			}
			yield return new WaitForSeconds(0f);

			Debug.Log("Exported " + samples + " samples.");
		}
	}

	public class BasketballSetup {
		public static void Export(MotionExporter exporter, Data X, Data Y, float tCurrent, float tNext) {
			Container current = new Container(exporter.Editor, tCurrent);
			Container next = new Container(exporter.Editor, tNext);

			if(current.Frame.Index == next.Frame.Index) {
				Debug.LogError("Same frames for input output pairs selected!");
			}

			string[] styles = new string[]{"Stand", "Move", "Dribble", "Hold", "Shoot"};

			//Input
			//Control
			for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
				X.FeedXZ(next.RootSeries.GetPosition(k).GetRelativePositionTo(current.Root), "TrajectoryPosition"+(k+1));
				X.FeedXZ(next.RootSeries.GetDirection(k).GetRelativeDirectionTo(current.Root), "TrajectoryDirection"+(k+1));
				X.FeedXZ(next.RootSeries.GetVelocity(k).GetRelativeDirectionTo(current.Root), "TrajectoryVelocity"+(k+1));
				X.Feed(next.DribbleSeries.Pivots[k].GetDirectionFromTo(next.RootSeries.Transformations[k], current.RootSeries.Transformations[k]), "DribblePivot"+(k+1));
				X.Feed(next.DribbleSeries.Momentums[k].GetDirectionFromTo(next.RootSeries.Transformations[k], current.RootSeries.Transformations[k]), "DribbleMomentum"+(k+1));
				X.Feed(next.StyleSeries.GetStyles(k, styles), "Style"+(k+1)+"-");
			}

			//Auto-Regressive Posture
			for(int k=0; k<current.ActorPosture.Length; k++) {
				X.Feed(current.ActorPosture[k].GetPosition().GetRelativePositionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Position");
				X.Feed(current.ActorPosture[k].GetForward().GetRelativeDirectionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Forward");
				X.Feed(current.ActorPosture[k].GetUp().GetRelativeDirectionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Up");
				X.Feed(current.ActorVelocities[k].GetRelativeDirectionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Velocity");
			}
			
			//Ball
			for(int k=0; k<=current.TimeSeries.Pivot; k++) {
				X.Feed(current.DribbleSeries.GetControlWeight(k, current.RootSeries.Transformations[k].GetPosition()), "BallWeight"+(k+1));
				X.Feed(current.DribbleSeries.GetWeightedBallPosition(k, current.RootSeries.Transformations[k].GetPosition()).GetRelativePositionTo(current.Root), "WeightedBallPosition"+(k+1));
				X.Feed(current.DribbleSeries.GetWeightedBallVelocity(k, current.RootSeries.Transformations[k].GetPosition()).GetRelativeDirectionTo(current.Root), "WeightedBallVelocity"+(k+1));
			}

			//Contacts
			for(int k=0; k<=current.TimeSeries.Pivot; k++) {
				X.Feed(current.ContactSeries.GetContacts(k), "Contacts"+(k+1)+"-");
			}

			//Rival
			for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
				X.Feed(current.DribbleSeries.GetInteractorWeight(k), "InteractorWeight"+(k+1));
			}
			for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
				X.FeedXZ(current.DribbleSeries.GetInteractorGradient(k), "InteractorRootGradient"+(k+1));
				X.FeedXZ(current.DribbleSeries.GetInteractorDirection(k), "InteractorRootDirection"+(k+1));
				X.FeedXZ(current.DribbleSeries.GetInteractorVelocity(k), "InteractorRootVelocity"+(k+1));
			}
			X.Feed(current.RivalBoneDistances, "InteractorBoneDistances");

			//Gating Variables
			{
				int index = 0;
				for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
					for(int b=0; b<current.PhaseSeries.Bones.Length; b++) {
						Vector2 phase = Utility.PhaseVector(current.PhaseSeries.Phases[k][b], current.PhaseSeries.Amplitudes[k][b]);
						index += 1;
						X.Feed(phase.x, "Gating"+index+"-Key"+(k+1)+"-Bone"+current.PhaseSeries.Bones[b]);
						index += 1;
						X.Feed(phase.y, "Gating"+index+"-Key"+(k+1)+"-Bone"+current.PhaseSeries.Bones[b]);
					}
				}
			}

			//Output
			//Root Update
			Matrix4x4 delta = next.Root.GetRelativeTransformationTo(current.Root);
			Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
			Y.FeedXZ(next.RootSeries.Velocities[next.TimeSeries.Pivot].GetRelativeDirectionTo(next.Root), "RootVelocity");
			Y.Feed(next.DribbleSeries.Pivots[next.TimeSeries.Pivot], "RootDribblePivot");
			Y.Feed(next.DribbleSeries.Momentums[next.TimeSeries.Pivot], "RootDribbleMomentum");
			Y.Feed(next.StyleSeries.GetStyles(next.TimeSeries.Pivot, styles), "RootStyle");

			//Control
			for(int k=next.TimeSeries.Pivot+1; k<next.TimeSeries.Samples.Length; k++) {
				Y.FeedXZ(next.RootSeries.GetPosition(k).GetRelativePositionTo(next.Root), "TrajectoryPosition"+(k+1));
				Y.FeedXZ(next.RootSeries.GetDirection(k).GetRelativeDirectionTo(next.Root), "TrajectoryDirection"+(k+1));
				Y.FeedXZ(next.RootSeries.GetVelocity(k).GetRelativeDirectionTo(next.Root), "TrajectoryVelocity"+(k+1));
				Y.Feed(next.DribbleSeries.Pivots[k], "DribblePivot"+(k+1));
				Y.Feed(next.DribbleSeries.Momentums[k], "DribbleMomentum"+(k+1));
				Y.Feed(next.StyleSeries.GetStyles(k, styles), "Style"+(k+1)+"-");
			}

			//Auto-Regressive Posture
			for(int k=0; k<next.ActorPosture.Length; k++) {
				Y.Feed(next.ActorPosture[k].GetPosition().GetRelativePositionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Position");
				Y.Feed(next.ActorPosture[k].GetForward().GetRelativeDirectionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Forward");
				Y.Feed(next.ActorPosture[k].GetUp().GetRelativeDirectionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Up");
				Y.Feed(next.ActorVelocities[k].GetRelativeDirectionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Velocity");
			}

			//Ball
			Y.Feed(next.DribbleSeries.GetControlWeight(next.TimeSeries.Pivot, next.Root.GetPosition()), "BallWeight");
			Y.Feed(next.DribbleSeries.GetWeightedBallPosition(next.TimeSeries.Pivot, next.Root.GetPosition()).GetRelativePositionTo(next.Root), "BallPosition");
			Y.Feed(next.DribbleSeries.GetWeightedBallVelocity(next.TimeSeries.Pivot, next.Root.GetPosition()).GetRelativeDirectionTo(next.Root), "BallVelocity");
			Quaternion rotationDelta = next.DribbleSeries.BallTransformations[next.TimeSeries.Pivot].GetRotation().GetRelativeRotationTo(current.DribbleSeries.BallTransformations[current.TimeSeries.Pivot]);
			Y.Feed(next.DribbleSeries.GetControlWeight(next.TimeSeries.Pivot, next.Root.GetPosition()) * rotationDelta.GetForward(), "BallForwardDelta");
			Y.Feed(next.DribbleSeries.GetControlWeight(next.TimeSeries.Pivot, next.Root.GetPosition()) * rotationDelta.GetUp(), "BallUpDelta");

			//Contacts
			for(int k=next.TimeSeries.Pivot; k<=next.TimeSeries.Pivot; k++) {
				Y.Feed(next.ContactSeries.GetContacts(k), "Contacts-");
			}

			//Phase Update
			for(int k=next.TimeSeries.Pivot; k<next.TimeSeries.Samples.Length; k++) {
				for(int b=0; b<next.PhaseSeries.Bones.Length; b++) {
					Y.Feed(Utility.PhaseVector(Utility.SignedPhaseUpdate(current.PhaseSeries.Phases[k][b], next.PhaseSeries.Phases[k][b]), next.PhaseSeries.Amplitudes[k][b]), "PhaseUpdate-"+(k+1)+"-"+(b+1));
					Y.Feed(Utility.PhaseVector(next.PhaseSeries.Phases[k][b], next.PhaseSeries.Amplitudes[k][b]), "PhaseState-"+(k+1)+"-"+(b+1));
				}
			}
		}

		private class Container {
			public MotionData Asset;
			public Frame Frame;

			public TimeSeries TimeSeries;
			public RootSeries RootSeries;
			public DribbleSeries DribbleSeries;
			public StyleSeries StyleSeries;
			public ContactSeries ContactSeries;
			public PhaseSeries PhaseSeries;

			//Actor Features
			public Matrix4x4 Root;
			public Matrix4x4[] ActorPosture;
			public Vector3[] ActorVelocities;
			public float[] RivalBoneDistances;

			public Container(MotionEditor editor, float timestamp) {
				editor.LoadFrame(timestamp);

				Asset = editor.GetAsset();
				Frame = editor.GetCurrentFrame();

				TimeSeries = editor.GetTimeSeries();
				RootSeries = (RootSeries)Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				DribbleSeries = (DribbleSeries)Asset.GetModule<DribbleModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				StyleSeries = (StyleSeries)Asset.GetModule<StyleModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				ContactSeries = (ContactSeries)Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				PhaseSeries = (PhaseSeries)Asset.GetModule<PhaseModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);

				Root = editor.GetActor().transform.GetWorldMatrix(true);
				ActorPosture = editor.GetActor().GetBoneTransformations();
				ActorVelocities = editor.GetActor().GetBoneVelocities();
				RivalBoneDistances = DribbleSeries.GetInteractorBoneDistances();
			}
		}
	}

	public class QuadrupedSetup {
		public static void Export(MotionExporter exporter, Data X, Data Y, float tCurrent, float tNext) {
			Container current = new Container(exporter.Editor, tCurrent);
			Container next = new Container(exporter.Editor, tNext);

			if(current.Frame.Index == next.Frame.Index) {
				Debug.LogError("Same frames for input output pairs selected!");
			}

			string[] styles = new string[]{"Idle", "Move", "Speed"};
			string[] contacts = new string[]{"LeftHandSite", "RightHandSite", "LeftFootSite", "RightFootSite"};
			string[] alignment = new string[]{"LeftHandSite", "RightHandSite", "LeftFootSite", "RightFootSite"};

			//Input
			//Control
			for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
				X.FeedXZ(next.RootSeries.GetPosition(k).GetRelativePositionTo(current.Root), "TrajectoryPosition"+(k+1));
				X.FeedXZ(next.RootSeries.GetDirection(k).GetRelativeDirectionTo(current.Root), "TrajectoryDirection"+(k+1));
				X.FeedXZ(next.RootSeries.GetVelocity(k).GetRelativeDirectionTo(current.Root), "TrajectoryVelocity"+(k+1));
				X.Feed(next.StyleSeries.GetStyles(k, styles), "Action"+(k+1)+"-");
			}

			//Auto-Regressive Posture
			for(int k=0; k<current.ActorPosture.Length; k++) {
				X.Feed(current.ActorPosture[k].GetPosition().GetRelativePositionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Position");
				X.Feed(current.ActorPosture[k].GetForward().GetRelativeDirectionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Forward");
				X.Feed(current.ActorPosture[k].GetUp().GetRelativeDirectionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Up");
				X.Feed(current.ActorVelocities[k].GetRelativeDirectionTo(current.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Velocity");
			}

			//Contacts
			for(int k=0; k<=current.TimeSeries.Pivot; k++) {
				X.Feed(current.ContactSeries.GetContacts(k, contacts), "Contacts"+(k+1)+"-");
			}

			//Gating Variables
			{
				int index = 0;
				for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
					for(int b=0; b<current.PhaseSeries.Bones.Length; b++) {
						if(alignment.Contains(current.PhaseSeries.Bones[b])) {
							Vector2 phase = Utility.PhaseVector(current.PhaseSeries.Phases[k][b], current.PhaseSeries.Amplitudes[k][b]);
							index += 1;
							X.Feed(phase.x, "Gating"+index+"-Key"+(k+1)+"-Bone"+current.PhaseSeries.Bones[b]);
							index += 1;
							X.Feed(phase.y, "Gating"+index+"-Key"+(k+1)+"-Bone"+current.PhaseSeries.Bones[b]);
						}
					}
				}
			}

			//Output
			//Root Update
			Matrix4x4 delta = next.Root.GetRelativeTransformationTo(current.Root);
			Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
			Y.FeedXZ(next.RootSeries.Velocities[next.TimeSeries.Pivot].GetRelativeDirectionTo(next.Root), "RootVelocity");
			Y.Feed(next.StyleSeries.GetStyles(next.TimeSeries.Pivot, styles), "RootAction");

			//Control
			for(int k=next.TimeSeries.Pivot+1; k<next.TimeSeries.Samples.Length; k++) {
				Y.FeedXZ(next.RootSeries.GetPosition(k).GetRelativePositionTo(next.Root), "TrajectoryPosition"+(k+1));
				Y.FeedXZ(next.RootSeries.GetDirection(k).GetRelativeDirectionTo(next.Root), "TrajectoryDirection"+(k+1));
				Y.FeedXZ(next.RootSeries.GetVelocity(k).GetRelativeDirectionTo(next.Root), "TrajectoryVelocity"+(k+1));
				Y.Feed(next.StyleSeries.GetStyles(k, styles), "Action"+(k+1)+"-");
			}

			//Auto-Regressive Posture
			for(int k=0; k<next.ActorPosture.Length; k++) {
				Y.Feed(next.ActorPosture[k].GetPosition().GetRelativePositionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Position");
				Y.Feed(next.ActorPosture[k].GetForward().GetRelativeDirectionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Forward");
				Y.Feed(next.ActorPosture[k].GetUp().GetRelativeDirectionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Up");
				Y.Feed(next.ActorVelocities[k].GetRelativeDirectionTo(next.Root), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Velocity");
			}

			//Contacts
			for(int k=next.TimeSeries.Pivot; k<=next.TimeSeries.Pivot; k++) {
				Y.Feed(next.ContactSeries.GetContacts(k, contacts), "Contacts-");
			}

			//Phase Update
			for(int k=next.TimeSeries.Pivot; k<next.TimeSeries.Samples.Length; k++) {
				for(int b=0; b<next.PhaseSeries.Bones.Length; b++) {
					if(alignment.Contains(current.PhaseSeries.Bones[b])) {
						Y.Feed(Utility.PhaseVector(Utility.SignedPhaseUpdate(current.PhaseSeries.Phases[k][b], next.PhaseSeries.Phases[k][b]), next.PhaseSeries.Amplitudes[k][b]), "PhaseUpdate-"+(k+1)+"-"+(b+1));
						Y.Feed(Utility.PhaseVector(next.PhaseSeries.Phases[k][b], next.PhaseSeries.Amplitudes[k][b]), "PhaseState-"+(k+1)+"-"+(b+1));
					}
				}
			}
		}

		private class Container {
			public MotionData Asset;
			public Frame Frame;

			public TimeSeries TimeSeries;
			public RootSeries RootSeries;
			public StyleSeries StyleSeries;
			public ContactSeries ContactSeries;
			public PhaseSeries PhaseSeries;

			//Actor Features
			public Matrix4x4 Root;
			public Matrix4x4[] ActorPosture;
			public Vector3[] ActorVelocities;

			public Container(MotionEditor editor, float timestamp) {
				editor.LoadFrame(timestamp);

				Asset = editor.GetAsset();
				Frame = editor.GetCurrentFrame();

				TimeSeries = editor.GetTimeSeries();
				RootSeries = (RootSeries)Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				StyleSeries = (StyleSeries)Asset.GetModule<StyleModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				ContactSeries = (ContactSeries)Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				PhaseSeries = (PhaseSeries)Asset.GetModule<PhaseModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);

				Root = editor.GetActor().transform.GetWorldMatrix(true);
				ActorPosture = editor.GetActor().GetBoneTransformations();
				ActorVelocities = editor.GetActor().GetBoneVelocities();
			}
		}
	}

}
#endif