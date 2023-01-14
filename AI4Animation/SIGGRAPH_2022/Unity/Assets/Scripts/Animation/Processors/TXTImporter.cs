#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class TXTImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;

		public float Framerate = 30f;
		public Vector3 Delta = Vector3.zero;

		private List<string> Imported;
		private List<string> Skipped;

		[MenuItem ("AI4Animation/Importer/TXT Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(TXTImporter));
			Scroll = Vector3.zero;
		}

		public override string GetID(Item item) {
			return item.ID;
		}

		public override void DerivedRefresh() {
			
		}

		public override void DerivedInspector() {
			EditorGUILayout.LabelField("Source");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
			Source = EditorGUILayout.TextField(Source);
			GUI.skin.button.alignment = TextAnchor.MiddleCenter;
			if(GUILayout.Button("O", GUILayout.Width(20))) {
				Source = EditorUtility.OpenFolderPanel("TXT Importer", Source == string.Empty ? Application.dataPath : Source, "");
				GUIUtility.ExitGUI();
			}
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			Destination = EditorGUILayout.TextField(Destination);
			EditorGUILayout.EndHorizontal();

			Framerate = EditorGUILayout.FloatField("Framerate", Framerate);
			Delta = EditorGUILayout.Vector3Field("Delta", Delta);

			if(Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
				LoadDirectory(Source);
			}
		}

		public override void DerivedInspector(Item item) {
		
		}

		private void SetSource(string source) {
			if(Source != source) {
				Source = source;
				LoadDirectory(null);
			}
		}

		private void LoadDirectory(string directory) {
			if(directory == null) {
				LoadItems(new string[0]);
			} else {
				if(Directory.Exists(directory)) {
					List<string> paths = new List<string>();
					Iterate(directory);
					LoadItems(paths.ToArray());
					void Iterate(string folder) {
						DirectoryInfo info = new DirectoryInfo(folder);
						foreach(FileInfo i in info.GetFiles("*.txt")) {
							paths.Add(i.FullName);
						}
						foreach(DirectoryInfo i in info.GetDirectories()) {
							Iterate(i.FullName);
						}
					}
				} else {
					LoadItems(new string[0]);
				}
			}
		}

        public override bool CanProcess() {
            return true;
        }

		public override void DerivedStart() {
			Imported = new List<string>();
			Skipped = new List<string>();
		}

        Vector3 ToUnity(Vector3 vector) {
            Vector3 result = Quaternion.Euler(Delta) * vector;
            vector.x = -vector.x;
            return result;
        }

        Quaternion ToUnity(Quaternion quaternion) {
            Quaternion result = Quaternion.Euler(Delta) * quaternion;
            return result;
        }

		public override IEnumerator DerivedProcess(Item item) {
			string source = Source;
			string destination = "Assets/" + Destination;
			string target = (destination + item.ID.Remove(0, source.Length)).Replace(".txt", "");

			if(!Directory.Exists(target)) {
				Directory.CreateDirectory(target);

				FileInfo file = new FileInfo(item.ID);

				MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
				asset.name = file.Name;
				AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");

				//Read Data
				string[] lines = FileUtility.ReadAllLines(file.FullName);
				float[][] frames = new float[lines.Length-1][];
				for(int i=0; i<frames.Length; i++) {
					frames[i] = FileUtility.LineToFloat(lines[i], ' ');
				}
				int[] topology = FileUtility.LineToInt(lines.Last(), ' ');

				//Set Framerate
				asset.Framerate = Framerate;

				//Set Frames
				asset.Frames = new Frame[frames.Length];
				for(int i=0; i<frames.Length; i++) {
					float[] values = frames[i];
					int joints = values.Length/9;
					Matrix4x4[] matrices = new Matrix4x4[joints];
					for(int j=0; j<joints; j++) {
						Vector3 p = new Vector3(values[9*j + 0], values[9*j + 1], values[9*j + 2]);
						Vector3 z = new Vector3(values[9*j + 3], values[9*j + 4], values[9*j + 5]);
						Vector3 y = new Vector3(values[9*j + 6], values[9*j + 7], values[9*j + 8]);
						Matrix4x4 m = Matrix4x4.TRS(
							ToUnity(p),
							Quaternion.LookRotation(ToUnity(z), ToUnity(y)),
							Vector3.one
						);
						matrices[j] = m;
					}
					asset.Frames[i] = new Frame(asset, i+1, (float)i / asset.Framerate, matrices);
				}

				//Set Topology
				List<Vector2Int> pairs = new List<Vector2Int>();
				for(int i=0; i<topology.Length; i+=2) {
					pairs.Add(new Vector2Int(topology[i], topology[i+1]));
				}
				asset.Source = new MotionAsset.Hierarchy(pairs.Count+1);
				for(int i=0; i<asset.Source.Bones.Length; i++) {
					asset.Source.Bones[i] = new MotionAsset.Hierarchy.Bone();
					asset.Source.Bones[i].Index = i;
					asset.Source.Bones[i].Name = i.ToString();
				}
				for(int i=0; i<pairs.Count; i++) {
					int bone = pairs[i].y;
					int parent = pairs[i].x;
					asset.Source.FindBone(bone.ToString()).Name = bone.ToString();
					asset.Source.FindBone(bone.ToString()).Parent = asset.Source.FindBone(parent.ToString()).Index;
				}
				
				//Detect Symmetry
				asset.DetectSymmetry();

                //Add Sequence
                asset.AddSequence();

				//Add Scene
				asset.CreateScene();

				//Save
				EditorUtility.SetDirty(asset);
				
				Imported.Add(target);
			} else {
				Skipped.Add(target);
			}

			yield return new WaitForSeconds(0f);
		}
		
		public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
		}

		public override void DerivedFinish() {
			AssetDatabase.Refresh();

			Debug.Log("Imported " + Imported.Count + " assets.");
			Imported.ToArray().Print();

			Debug.Log("Skipped " + Skipped.Count + " assets.");
			Skipped.ToArray().Print();
		}

	}
}
#endif