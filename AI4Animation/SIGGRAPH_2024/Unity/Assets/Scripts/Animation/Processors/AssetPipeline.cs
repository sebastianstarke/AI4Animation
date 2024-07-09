#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Threading;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Globalization;

namespace AI4Animation {
	public class AssetPipeline : BatchProcessor {

		public AssetPipelineSetup Setup = null;

		private MotionEditor Editor = null;

		[MenuItem ("AI4Animation/Tools/Asset Pipeline")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(AssetPipeline));
			Scroll = Vector3.zero;
		}

		public MotionEditor GetEditor() {
			if(Editor == null) {
				Editor = GameObjectExtensions.Find<MotionEditor>(true);
			}
			return Editor;
		}

		public override string GetID(Item item) {
			return Utility.GetAssetName(item.ID);
		}

		public override void DerivedRefresh() {
			
		}
		
		public override void DerivedInspector() {
			if(GetEditor() == null) {
				EditorGUILayout.LabelField("No editor available in scene.");
				return;
			}

			EditorGUILayout.ObjectField("Editor", Editor, typeof(MotionEditor), true);

			Assign((AssetPipelineSetup)EditorGUILayout.ObjectField("Setup", Setup, typeof(AssetPipelineSetup), true));

			if(Setup != null) {
				Setup.Inspector();
			}

			if(Utility.GUIButton("Refresh", UltiDraw.DarkGrey, UltiDraw.White)) {
				LoadItems(GetEditor().Assets.ToArray());
			}

		}

		public override void DerivedInspector(Item item) {
			if(Setup != null) {
				Setup.Inspector(item);
			}
		}

		public override bool CanProcess() {
			return Setup != null && Setup.CanProcess();
		}

		public override void DerivedStart() {
			Setup.Begin();
		}

		public override IEnumerator DerivedProcess(Item item) {
			EditorCoroutines.EditorCoroutine c = this.StartCoroutine(Setup.Iterate(MotionAsset.Retrieve(item.ID)));
			while(!c.finished) {
				yield return new WaitForSeconds(0f);
			}
		}

		public override void BatchCallback() {
			Setup.Callback();
		}

		public override void DerivedFinish() {
			Setup.Finish();
		}

		public void Assign(AssetPipelineSetup setup) {
			Setup = setup;
			if(Setup != null) {
				setup.Pipeline = this;
			}
		}

		public class Data {
			public enum TYPE {Text, Binary}

			public static string Separator = " ";
			public static int Digits = 6;
			public static string Accuracy = "F6";

			private File Features, Normalization, Labels, Shape;

			private Queue<float[]> Buffer = new Queue<float[]>();
			private Task Process = null;

			private float[] Values = new float[0];
			private string[] Names = new string[0];
			private string[] MeanGroups = new string[0];
			private string[] SigmaGroups = new string[0];
			private Dictionary<string, RunningStatistics> MeanStatistics = new Dictionary<string, RunningStatistics>();
			private Dictionary<string, RunningStatistics> SigmaStatistics = new Dictionary<string, RunningStatistics>();
			private int Samples = 0;
			private int Dim = 0;

			private bool Finished = false;
			private bool Setup = false;

			private string Name;

			public class File {
				public object Writer;
				public TYPE Type;
				private int Lines = 0;
				public File(string name, TYPE type, string directory="") {
					Type = type;
					directory = GetExportPath() + (directory == "" ? (directory) : ("/" + directory));
					if(!Directory.Exists(directory)) {
						Directory.CreateDirectory(directory);
					}
					if(Type == TYPE.Text) {
						Writer = new StreamWriter(System.IO.File.Open(directory + "/" + name + ".txt", FileMode.Create));
					}
					if(Type == TYPE.Binary) {
						Writer = new BinaryWriter(System.IO.File.Open(directory + "/" + name + ".bin", FileMode.Create));
					}
				}
				public void WriteLine(string value) {
					if(Writer == null) {return;};
					if(Type == TYPE.Text) {
						if(Lines > 0) {
							((StreamWriter)Writer).Write(((StreamWriter)Writer).NewLine);
						}
						((StreamWriter)Writer).Write(value);
					}
					if(Type == TYPE.Binary) {
						((BinaryWriter)Writer).Write(value);
					}
					Lines += 1;
				}
				public void WriteLine(float[] values) {
					if(Writer == null) {return;};
					if(Type == TYPE.Text) {
						if(Lines > 0) {
							((StreamWriter)Writer).Write(((StreamWriter)Writer).NewLine);
						}
						((StreamWriter)Writer).Write(String.Join(Separator, Array.ConvertAll(values, x => x.Round(Digits).ToString(Accuracy))));
					}
					if(Type == TYPE.Binary) {
						foreach(float value in values) {
							((BinaryWriter)Writer).Write(value);
						}
					}
					Lines += 1;
				}
				public void WriteLine(int[] values) {
					if(Writer == null) {return;};
					if(Type == TYPE.Text) {
						if(Lines > 0) {
							((StreamWriter)Writer).Write(((StreamWriter)Writer).NewLine);
						}
						((StreamWriter)Writer).Write(String.Join(Separator, values));
					}
					if(Type == TYPE.Binary) {
						foreach(int value in values) {
							((BinaryWriter)Writer).Write(value);
						}
					}
					Lines += 1;
				}
				public void Close() {
					if(Type == TYPE.Text) {
						((StreamWriter)Writer).Close();
					}
					if(Type == TYPE.Binary) {
						((BinaryWriter)Writer).Close();
					}
					Writer = null;
				}
			}

			public static File CreateFile(string name, TYPE type, string directory="") {
				return new File(name, type, directory);
			}

			public static string GetExportPath() {
				string path = Application.dataPath;
				path = path.Substring(0, path.LastIndexOf("/"));
				path = path.Substring(0, path.LastIndexOf("/"));
				path += "/PyTorch/Dataset";
				return path;
			}

			public Data(string name, bool exportLabels=true, bool exportNormalization=true, bool exportShape=true, TYPE featuresType=TYPE.Binary) {
				Name = name;
				Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
				Features = CreateFile(name, featuresType);
				if(exportLabels) {
					Labels = CreateFile(name+"Labels", TYPE.Text);
				}
				if(exportNormalization) {
					Normalization = CreateFile(name+"Normalization", TYPE.Text);
				}
				if(exportShape) {
					Shape = CreateFile(name+"Shape", TYPE.Text);
				}
				Process = Task.Factory.StartNew(() => WriteData());
			}

			public void Feed(float value, string name, string meanGroup=null, string sigmaGroup=null) {
				Dim += 1;
				if(!Setup) {
					ArrayExtensions.Append(ref Values, value);
					ArrayExtensions.Append(ref Names, name);
					ArrayExtensions.Append(ref MeanGroups, meanGroup == null ? MeanGroups.Length.ToString() : meanGroup);
					ArrayExtensions.Append(ref SigmaGroups, sigmaGroup == null ? SigmaGroups.Length.ToString() : sigmaGroup);
				} else {
					Values[Dim-1] = value;
				}
			}

			public void Feed(bool value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value ? 1f : 0f, name, meanGroup, sigmaGroup);
			}

			public void Feed(float[] values, string name, string meanGroup=null, string sigmaGroup=null) {
				for(int i=0; i<values.Length; i++) {
					string id = (i+1).ToString();
					Feed(values[i], Setup ? null : (name+id), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+id));
				}
			}

			public void Feed(bool[] values, string name, string meanGroup=null, string sigmaGroup=null) {
				for(int i=0; i<values.Length; i++) {
					string id = (i+1).ToString();
					Feed(values[i], Setup ? null : (name+id), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+id));
				}
			}

			public void Feed(float[,] values, string name, string meanGroup=null, string sigmaGroup=null) {
				for(int i=0; i<values.GetLength(0); i++) {
					for(int j=0; j<values.GetLength(1); j++) {
						string id = (i*values.GetLength(1)+j+1).ToString();
						Feed(values[i,j], Setup ? null : (name+id), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+id));
					}
				}
			}

			public void Feed(bool[,] values, string name, string meanGroup=null, string sigmaGroup=null) {
				for(int i=0; i<values.GetLength(0); i++) {
					for(int j=0; j<values.GetLength(1); j++) {
						string id = (i*values.GetLength(1)+j+1).ToString();
						Feed(values[i,j], Setup ? null : (name+id), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+id));
					}
				}
			}

			public void Feed(Vector2 value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value.x, Setup ? null : (name+"X"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"X"));
				Feed(value.y, Setup ? null : (name+"Y"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Y"));
			}

			public void Feed(Vector2[] values, string name, string meanGroup=null, string sigmaGroup=null) {
				for(int i=0; i<values.Length; i++) {
					string id = (i+1).ToString();
					Feed(values[i], Setup ? null : (name+id), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+id));
				}
			}

			public void Feed(Vector3 value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value.x, Setup ? null : (name+"X"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"X"));
				Feed(value.y, Setup ? null : (name+"Y"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Y"));
				Feed(value.z, Setup ? null : (name+"Z"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Z"));
			}

			public void FeedXY(Vector3 value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value.x, Setup ? null : (name+"X"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"X"));
				Feed(value.y, Setup ? null : (name+"Y"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Y"));
			}

			public void FeedXZ(Vector3 value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value.x, Setup ? null : (name+"X"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"X"));
				Feed(value.z, Setup ? null : (name+"Z"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Z"));
			}

			public void FeedYZ(Vector3 value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value.y, Setup ? null : (name+"Y"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Y"));
				Feed(value.z, Setup ? null : (name+"Z"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Z"));
			}

			public void Feed(Quaternion value, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(value.x, Setup ? null : (name+"X"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"X"));
				Feed(value.y, Setup ? null : (name+"Y"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Y"));
				Feed(value.z, Setup ? null : (name+"Z"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"Z"));
				Feed(value.w, Setup ? null : (name+"W"), meanGroup, sigmaGroup==null ? sigmaGroup : (sigmaGroup+"W"));
			}

			public void Feed(Matrix4x4 matrix, string name, string meanGroup=null, string sigmaGroup=null) {
				Feed(matrix.GetPosition(), name + "Position", meanGroup, sigmaGroup);
				Feed(matrix.GetForward(), name + "Forward", meanGroup, sigmaGroup);
				Feed(matrix.GetUp(), name + "Up", meanGroup, sigmaGroup);
			}

			private void WriteData() {
				while(!Finished || Buffer.Count > 0) {
					if(Buffer.Count > 0) {
						float[] item;
						lock(Buffer) {
							item = Buffer.Dequeue();	
						}
						if(Normalization != null) {
							//Update Mean and Std
							for(int i=0; i<item.Length; i++) {
								MeanStatistics[MeanGroups[i]].Add(item[i]);
								SigmaStatistics[SigmaGroups[i]].Add(item[i]);
							}
						}
						//Write to File
						Features.WriteLine(item);
						Samples += 1;
					} else {
						Thread.Sleep(1);
					}
				}
			}

			public void Store() {
				if(!Setup) {
					if(Labels != null) {
						//Write Labels
						for(int i=0; i<Names.Length; i++) {
							Labels.WriteLine("[" + i + "]" + " " + Names[i]);
						}
						Labels.Close();
					}
					if(Normalization != null) {
						//Generate Normalization
						for(int i=0; i<MeanGroups.Length; i++) {
							if(!MeanStatistics.ContainsKey(MeanGroups[i])) {
								MeanStatistics.Add(MeanGroups[i], new RunningStatistics());
							}
							if(!SigmaStatistics.ContainsKey(SigmaGroups[i])) {
								SigmaStatistics.Add(SigmaGroups[i], new RunningStatistics());
							}
						}
					}
					Setup = true;
				}

				//Skip If Nothing Written
				if(Dim == 0) {
					// Debug.LogWarning("Attempting to store feature vector with no values given.");
					return;
				}
				
				//Skip If Dim Mismatch
				if(Dim != Values.Length) {
					// Debug.LogWarning("Writing different number of features than initially registered.");
					return;
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

				//Wait for remaining features
				Task.WaitAll(Process);
				Features.Close();

				if(Normalization != null) {
					//Write Mean and Sigma
					{
						float[] mean = new float[MeanGroups.Length];
						for(int i=0; i<MeanGroups.Length; i++) {
							mean[i] = MeanStatistics[MeanGroups[i]].Mean();
							// if(Normalization.Type == TYPE.Text) {
							// 	mean[i] = mean[i].Round(Digits);
							// }
						}
						Normalization.WriteLine(mean);
					}
					{
						float[] sigma = new float[SigmaGroups.Length];
						for(int i=0; i<MeanGroups.Length; i++) {
							sigma[i] = SigmaStatistics[SigmaGroups[i]].Sigma();
							if(Normalization.Type == TYPE.Text) {
								sigma[i] = sigma[i].Round(Digits);
							}
							if(sigma[i] == 0f) {
								Debug.LogWarning("Standard deviation for feature " + Names[i] + " in file " + Name + " is 0 and will be converted to 1.");
								sigma[i] = 1f;
							}
						}
						Normalization.WriteLine(sigma);
					}
					Normalization.Close();
				}

				if(Shape != null) {
					//Write Shape
					Shape.WriteLine(Samples.ToString());
					Shape.WriteLine(Values.Length.ToString());
					Shape.Close();
				}
			}
		}

	}
}
#endif