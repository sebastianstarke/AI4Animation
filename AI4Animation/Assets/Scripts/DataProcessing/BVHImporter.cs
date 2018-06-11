using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class BVHImporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Source = string.Empty;
	public string Destination = string.Empty;
	public bool[] Import = new bool[0];
	public FileInfo[] Files = new FileInfo[0];
	public bool Importing = false;
	
	[MenuItem ("Addons/BVH Importer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHImporter));
		Scroll = Vector3.zero;
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
					EditorGUILayout.LabelField("BVH Importer");
				}
		
				if(!Importing) {
					if(Utility.GUIButton("Import Motion Data", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(ImportMotionData());
					}

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Import.Length; i++) {
							Import[i] = true;
						}
					}
					if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Import.Length; i++) {
							Import[i] = false;
						}
					}
					EditorGUILayout.EndHorizontal();
				} else {
					if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) {
						this.StopAllCoroutines();
						Importing = false;
					}
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.LabelField("Source");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					LoadDirectory(EditorGUILayout.TextField(Source));
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						string source = EditorUtility.OpenFilePanel("BVH Importer", Source == string.Empty ? Application.dataPath : Source, "");
						source = source.Substring(0, source.LastIndexOf("/"));
						LoadDirectory(source);
						GUI.SetNextControlName("");
						GUI.FocusControl("");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.LabelField("Destination");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
					Destination = EditorGUILayout.TextField(Destination);
					EditorGUILayout.EndHorizontal();

					for(int i=0; i<Files.Length; i++) {
						if(Import[i]) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Import[i] = EditorGUILayout.Toggle(Import[i], GUILayout.Width(20f));
							EditorGUILayout.LabelField(Files[i].Name);
							EditorGUILayout.EndHorizontal();
						}
					}
				}
		
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory(string source) {
		if(Source != source) {
			Source = source;
			Files = new FileInfo[0];
			Import = new bool[0];
			if(Directory.Exists(Source)) {
				DirectoryInfo info = new DirectoryInfo(Source);
				Files = info.GetFiles("*.bvh");
				Import = new bool[Files.Length];
				for(int i=0; i<Files.Length; i++) {
					Import[i] = true;
				}
			}
		}
	}

	private IEnumerator ImportMotionData() {
		string destination = "Assets/" + Destination;
		if(!AssetDatabase.IsValidFolder(destination)) {
			Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
		} else {
			Importing = true;
			for(int f=0; f<Files.Length; f++) {
				if(Import[f]) {
					MotionData data = ScriptableObject.CreateInstance<MotionData>();
					data.Name = Files[f].FullName.Substring(Files[f].FullName.LastIndexOf("/")+1);
					if(AssetDatabase.LoadAssetAtPath(destination+"/"+data.Name+".asset", typeof(MotionData)) == null) {
						AssetDatabase.CreateAsset(data , destination+"/"+data.Name+".asset");
					} else {
						int i = 1;
						while(AssetDatabase.LoadAssetAtPath(destination+"/"+data.Name+data.Name+" ("+i+").asset", typeof(MotionData)) != null) {
							i += 1;
						}
						AssetDatabase.CreateAsset(data, destination+"/"+data.Name+data.Name+" ("+i+").asset");
					}

					string[] lines = File.ReadAllLines(Files[f].FullName);
					char[] whitespace = new char[] {' '};
					int index = 0;

					//Create Source Data
					List<Vector3> offsets = new List<Vector3>();
					List<int[]> channels = new List<int[]>();
					List<float[]> motions = new List<float[]>();
					data.Source = new MotionData.Hierarchy();
					string name = string.Empty;
					string parent = string.Empty;
					Vector3 offset = Vector3.zero;
					int[] channel = null;
					for(index = 0; index<lines.Length; index++) {
						if(lines[index] == "MOTION") {
							break;
						}
						string[] entries = lines[index].Split(whitespace);
						for(int entry=0; entry<entries.Length; entry++) {
							if(entries[entry].Contains("ROOT")) {
								parent = "None";
								name = entries[entry+1];
								break;
							} else if(entries[entry].Contains("JOINT")) {
								parent = name;
								name = entries[entry+1];
								break;
							} else if(entries[entry].Contains("End")) {
								parent = name;
								name = name+entries[entry+1];
								string[] subEntries = lines[index+2].Split(whitespace);
								for(int subEntry=0; subEntry<subEntries.Length; subEntry++) {
									if(subEntries[subEntry].Contains("OFFSET")) {
										offset.x = Utility.ReadFloat(subEntries[subEntry+1]);
										offset.y = Utility.ReadFloat(subEntries[subEntry+2]);
										offset.z = Utility.ReadFloat(subEntries[subEntry+3]);
										break;
									}
								}
								data.Source.AddBone(name, parent);
								offsets.Add(offset);
								channels.Add(new int[0]);
								index += 2;
								break;
							} else if(entries[entry].Contains("OFFSET")) {
								offset.x = Utility.ReadFloat(entries[entry+1]);
								offset.y = Utility.ReadFloat(entries[entry+2]);
								offset.z = Utility.ReadFloat(entries[entry+3]);
								break;
							} else if(entries[entry].Contains("CHANNELS")) {
								channel = new int[Utility.ReadInt(entries[entry+1])];
								for(int i=0; i<channel.Length; i++) {
									if(entries[entry+2+i] == "Xposition") {
										channel[i] = 1;
									} else if(entries[entry+2+i] == "Yposition") {
										channel[i] = 2;
									} else if(entries[entry+2+i] == "Zposition") {
										channel[i] = 3;
									} else if(entries[entry+2+i] == "Xrotation") {
										channel[i] = 4;
									} else if(entries[entry+2+i] == "Yrotation") {
										channel[i] = 5;
									} else if(entries[entry+2+i] == "Zrotation") {
										channel[i] = 6;
									}
								}
								data.Source.AddBone(name, parent);
								offsets.Add(offset);
								channels.Add(channel);
								break;
							} else if(entries[entry].Contains("}")) {
								name = parent;
								parent = name == "None" ? "None" : data.Source.FindBone(name).Parent;
								break;
							}
						}
					}

					//REMOVE LATER
					//data.Corrections = new Vector3[data.Source.Bones.Length];
					//

					//Set Frames
					index += 1;
					while(lines[index].Length == 0) {
						index += 1;
					}
					ArrayExtensions.Resize(ref data.Frames, Utility.ReadInt(lines[index].Substring(8)));

					//Set Framerate
					index += 1;
					data.Framerate = Mathf.RoundToInt(1f / Utility.ReadFloat(lines[index].Substring(12)));

					//Compute Frames
					index += 1;
					for(int i=index; i<lines.Length; i++) {
						motions.Add(Utility.ReadArray(lines[i]));
					}
					for(int k=0; k<data.GetTotalFrames(); k++) {
						data.Frames[k] = new MotionData.Frame(data, k+1, (float)k / data.Framerate);
						int idx = 0;
						for(int i=0; i<data.Source.Bones.Length; i++) {
							MotionData.Hierarchy.Bone info = data.Source.Bones[i];
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

							position = (position == Vector3.zero ? offsets[i] : position) / 100f; //unit scale
							data.Frames[k].Local[i] = Matrix4x4.TRS(position, rotation, Vector3.one);
							data.Frames[k].World[i] = info.Parent == "None" ? data.Frames[k].Local[i] : data.Frames[k].World[data.Source.FindBone(info.Parent).Index] * data.Frames[k].Local[i];
						}
						/*
						for(int i=0; i<data.Source.Bones.Length; i++) {
							data.Frames[k].Local[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(data.Corrections[i]), Vector3.one);
							data.Frames[k].World[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(data.Corrections[i]), Vector3.one);
						}
						*/
					}

					//Finalise
					data.DetectHeightMapSensor();
					data.DetectDepthMapSensor();
					data.DetectSymmetry();
					data.ComputeStyles();
					data.AddSequence();

					yield return new WaitForSeconds(0f);
				}
			}
			Importing = false;
		}
		yield return new WaitForSeconds(0f);
	}

}

								/*
								string[] presets = new string[4] {"Select preset...", "Dan", "Dog", "Interaction"};
								switch(EditorGUILayout.Popup(0, presets)) {
									case 0:
									break;
									case 1:
									Target.GetData().DepthMapAxis = MotionData.Axis.ZPositive;
									Target.GetData().MirrorAxis = MotionData.Axis.XPositive;
									for(int i=0; i<Target.GetData().Corrections.Length; i++) {
										Target.GetData().SetCorrection(i, Vector3.zero);
									}
									Target.GetData().ClearStyles();
									Target.GetData().AddStyle("Idle");
									Target.GetData().AddStyle("Walk");
									Target.GetData().AddStyle("Run");
									Target.GetData().AddStyle("Jump");
									Target.GetData().AddStyle("Crouch");
									break;

									case 2:
									Target.GetData().DepthMapAxis = MotionData.Axis.XPositive;
									Target.GetData().MirrorAxis = MotionData.Axis.ZPositive;
									for(int i=0; i<Target.GetData().Corrections.Length; i++) {
										if(i==4 || i==5 || i==6 || i==11) {
											Target.GetData().SetCorrection(i, new Vector3(90f, 90f, 90f));
										} else if(i==24) {
											Target.GetData().SetCorrection(i, new Vector3(-45f, 0f, 0f));
										} else {
											Target.GetData().SetCorrection(i, new Vector3(0f, 0f, 0f));
										}
									}
									Target.GetData().ClearStyles();
									Target.GetData().AddStyle("Idle");
									Target.GetData().AddStyle("Walk");
									Target.GetData().AddStyle("Pace");
									Target.GetData().AddStyle("Trot");
									Target.GetData().AddStyle("Canter");
									Target.GetData().AddStyle("Jump");
									Target.GetData().AddStyle("Sit");
									Target.GetData().AddStyle("Stand");
									Target.GetData().AddStyle("Lie");
									break;

									case 3:
									Target.GetData().DepthMapAxis = MotionData.Axis.ZPositive;
									Target.GetData().MirrorAxis = MotionData.Axis.XPositive;							
									for(int i=0; i<Target.GetData().Corrections.Length; i++) {
										Target.GetData().SetCorrection(i, Vector3.zero);
									}
									Target.GetData().ClearStyles();
									Target.GetData().AddStyle("Idle");
									Target.GetData().AddStyle("Walk");
									Target.GetData().AddStyle("Run");
									Target.GetData().AddStyle("Jump");
									Target.GetData().AddStyle("Crouch");
									Target.GetData().AddStyle("Sit");
									break;
								}
								*/