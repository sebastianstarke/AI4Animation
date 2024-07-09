#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class BVHImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;
		// public bool Flip = true;
		public Axis Axis = Axis.XPositive;

		private List<string> Imported;
		private List<string> Skipped;

		[MenuItem ("AI4Animation/Importer/BVH Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(BVHImporter));
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
				Source = EditorUtility.OpenFolderPanel("BVH Importer", Source == string.Empty ? Application.dataPath : Source, "");
				GUIUtility.ExitGUI();
			}
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			Destination = EditorGUILayout.TextField(Destination);
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.BeginHorizontal();
			// Flip = EditorGUILayout.Toggle("Flip", Flip);
			Axis = (Axis)EditorGUILayout.EnumPopup(Axis);
			EditorGUILayout.EndHorizontal();

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
						foreach(FileInfo i in info.GetFiles("*.bvh")) {
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

		public override IEnumerator DerivedProcess(Item item) {
			string source = Source;
			string destination = "Assets/" + Destination;
			string target = (destination + item.ID.Remove(0, source.Length)).Replace(".bvh", "");
			
			string FilterLine(string line, char[] whitespace) {
				for(int i=0; i<line.Length; i++) {
					if(!whitespace.Contains(line[i])) {
						return line.Substring(i);
					}
				}
				return line;
			}

			if(!Directory.Exists(target)) {
				Directory.CreateDirectory(target);

				FileInfo file = new FileInfo(item.ID);

				MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
				asset.name = file.Name;
				AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");

				string[] lines = System.IO.File.ReadAllLines(file.FullName);
				int index = 0;

				char[] whitespace = new char[] {'\t', ' '};

				//Create Source Data
				List<Vector3> offsets = new List<Vector3>();
				List<int[]> channels = new List<int[]>();
				List<float[]> motions = new List<float[]>();
				asset.Source = new MotionAsset.Hierarchy();
				string name = string.Empty;
				string parent = string.Empty;
				Vector3 offset = Vector3.zero;
				int[] channel = null;
				for(index=0; index<lines.Length; index++) {
					if(lines[index] == "MOTION") {
						break;
					}

					// New Pipeline
					string line = FilterLine(lines[index], whitespace);
					string[] tags = line.Split(whitespace);
					string id = tags[0];
					
					// Debug.Log("Line: " + line + " ID: " + id);
					if(id == "ROOT") {
						parent = "None";
						name = line.Substring(id.Length+1);
					}
					if(id == "JOINT") {
						parent = name;
						name = line.Substring(id.Length+1);
					}
					if(id == "End") {
						parent = name;
						name = parent + line.Substring(id.Length+1);
						string[] entries = FilterLine(lines[index+2], whitespace).Split(whitespace);
						for(int entry=0; entry<entries.Length; entry++) {
							if(entries[entry].Contains("OFFSET")) {
								offset.x = FileUtility.ReadFloat(entries[entry+1]);
								offset.y = FileUtility.ReadFloat(entries[entry+2]);
								offset.z = FileUtility.ReadFloat(entries[entry+3]);
								break;
							}
						}
						asset.Source.AddBone(name, parent);
						offsets.Add(offset);
						channels.Add(new int[0]);
						index += 2;
					}
					if(id == "OFFSET") {
						offset.x = FileUtility.ReadFloat(tags[1]);
						offset.y = FileUtility.ReadFloat(tags[2]);
						offset.z = FileUtility.ReadFloat(tags[3]);
					}
					if(id == "CHANNELS") {
						channel = new int[FileUtility.ReadInt(tags[1])];
						for(int i=0; i<channel.Length; i++) {
							if(tags[2+i] == "Xposition") {
								channel[i] = 1;
							} else if(tags[2+i] == "Yposition") {
								channel[i] = 2;
							} else if(tags[2+i] == "Zposition") {
								channel[i] = 3;
							} else if(tags[2+i] == "Xrotation") {
								channel[i] = 4;
							} else if(tags[2+i] == "Yrotation") {
								channel[i] = 5;
							} else if(tags[2+i] == "Zrotation") {
								channel[i] = 6;
							}
						}
						asset.Source.AddBone(name, parent);
						offsets.Add(offset);
						channels.Add(channel);
					}
					if(id == "}") {
						name = parent;
						MotionAsset.Hierarchy.Bone bone = asset.Source.FindBone(name);
						parent = bone == null || bone.Parent == -1 ? "None" : asset.Source.Bones[bone.Parent].Name;
					}

					// Old Pipeline
					// string[] entries = lines[index].Split(whitespace);
					// for(int entry=0; entry<entries.Length; entry++) {
					// 	if(entries[entry].Contains("ROOT")) {
					// 		parent = "None";
					// 		name = entries[entry+1];
					// 		break;
					// 	} else if(entries[entry].Contains("JOINT")) {
					// 		parent = name;
					// 		name = entries[entry+1];
					// 		break;
					// 	} else if(entries[entry].Contains("End")) {
					// 		parent = name;
					// 		name = name+entries[entry+1];
					// 		string[] subEntries = lines[index+2].Split(whitespace);
					// 		for(int subEntry=0; subEntry<subEntries.Length; subEntry++) {
					// 			if(subEntries[subEntry].Contains("OFFSET")) {
					// 				offset.x = FileUtility.ReadFloat(subEntries[subEntry+1]);
					// 				offset.y = FileUtility.ReadFloat(subEntries[subEntry+2]);
					// 				offset.z = FileUtility.ReadFloat(subEntries[subEntry+3]);
					// 				break;
					// 			}
					// 		}
					// 		asset.Source.AddBone(name, parent);
					// 		offsets.Add(offset);
					// 		channels.Add(new int[0]);
					// 		index += 2;
					// 		break;
					// 	} else if(entries[entry].Contains("OFFSET")) {
					// 		offset.x = FileUtility.ReadFloat(entries[entry+1]);
					// 		offset.y = FileUtility.ReadFloat(entries[entry+2]);
					// 		offset.z = FileUtility.ReadFloat(entries[entry+3]);
					// 		break;
					// 	} else if(entries[entry].Contains("CHANNELS")) {
					// 		channel = new int[FileUtility.ReadInt(entries[entry+1])];
					// 		for(int i=0; i<channel.Length; i++) {
					// 			if(entries[entry+2+i] == "Xposition") {
					// 				channel[i] = 1;
					// 			} else if(entries[entry+2+i] == "Yposition") {
					// 				channel[i] = 2;
					// 			} else if(entries[entry+2+i] == "Zposition") {
					// 				channel[i] = 3;
					// 			} else if(entries[entry+2+i] == "Xrotation") {
					// 				channel[i] = 4;
					// 			} else if(entries[entry+2+i] == "Yrotation") {
					// 				channel[i] = 5;
					// 			} else if(entries[entry+2+i] == "Zrotation") {
					// 				channel[i] = 6;
					// 			}
					// 		}
					// 		asset.Source.AddBone(name, parent);
					// 		offsets.Add(offset);
					// 		channels.Add(channel);
					// 		break;
					// 	} else if(entries[entry].Contains("}")) {
					// 		name = parent;
					// 		parent = name == "None" ? "None" : asset.Source.FindBone(name).Parent;
					// 		break;
					// 	}
					// }
					
				}

				//Set Frames
				index += 1;
				while(lines[index].Length == 0) {
					index += 1;
				}
				int frameCount = FileUtility.ReadInt(lines[index].Substring(8));

				//Set Framerate
				index += 1;
				asset.Framerate = Mathf.RoundToInt(1f / FileUtility.ReadFloat(lines[index].Substring(12)));

				//Compute Frames
				index += 1;
				for(int i=index; i<lines.Length; i++) {
					motions.Add(FileUtility.ReadArray(lines[i]));
				}
				ArrayExtensions.Resize(ref asset.Frames, motions.Count);
				if(frameCount != asset.Frames.Length) {
					Debug.LogWarning("Expected and true number of frames in file " + asset.name + " do not match.");
				}
				for(int k=0; k<asset.GetTotalFrames(); k++) {
					Matrix4x4[] matrices = new Matrix4x4[asset.Source.Bones.Length];
					int idx = 0;
					for(int i=0; i<asset.Source.Bones.Length; i++) {
						MotionAsset.Hierarchy.Bone bone = asset.Source.Bones[i];
						Vector3 position = Vector3.zero;
						Quaternion rotation = Quaternion.identity;
						for(int j=0; j<channels[i].Length; j++) {
							if(channels[i][j] == 1) {
								position.x = motions[k][idx]; idx += 1;
							}
							if(channels[i][j] == 2) {
								position.y = motions[k][idx]; idx += 1;
							}
							if(channels[i][j] == 3) {
								position.z = motions[k][idx]; idx += 1;
							}
							if(channels[i][j] == 4) {
								rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.right); idx += 1;
							}
							if(channels[i][j] == 5) {
								rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.up); idx += 1;
							}
							if(channels[i][j] == 6) {
								rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.forward); idx += 1;
							}
						}

						position = position == Vector3.zero ? offsets[i] : position;
						Matrix4x4 local = Matrix4x4.TRS(position, rotation, Vector3.one);
						// if(Flip) {
						local = local.GetMirror(Axis); //This is due to BVH coordinate system being RH while Unity is LH
						// }
						matrices[i] = bone.Parent == -1 ? local : matrices[asset.Source.Bones[bone.Parent].Index] * local;
					}
					asset.Frames[k] = new Frame(asset, k+1, (float)k / asset.Framerate, matrices);
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