#if UNITY_EDITOR
using System;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class TRCImporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Source = string.Empty;
	public string Destination = string.Empty;
	public string Filter = string.Empty;
	public File[] Files = new File[0];
	public File[] Instances = new File[0];
	public bool Importing = false;
	
	public int Page = 1;
	public const int Items = 25;

	[MenuItem ("AI4Animation/TRC Importer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(TRCImporter));
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
					EditorGUILayout.LabelField("TRC Importer");
				}
		
				if(!Importing) {
					if(Utility.GUIButton("Import Motion Data", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(ImportMotionData());
					}
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

					string filter = EditorGUILayout.TextField("Filter", Filter);
					if(Filter != filter) {
						Filter = filter;
						ApplyFilter();
					}

					if(Utility.GUIButton("Load Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
						LoadDirectory();
					}

					int start = (Page-1)*Items;
					int end = Mathf.Min(start+Items, Instances.Length);
					int pages = Mathf.CeilToInt(Instances.Length/Items)+1;
					Utility.SetGUIColor(UltiDraw.Orange);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White)) {
							Page = Mathf.Max(Page-1, 1);
						}
						EditorGUILayout.LabelField("Page " + Page + "/" + pages);
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White)) {
							Page = Mathf.Min(Page+1, pages);
						}
						EditorGUILayout.EndHorizontal();
					}
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Instances.Length; i++) {
							Instances[i].Import = true;
						}
					}
					if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Instances.Length; i++) {
							Instances[i].Import = false;
						}
					}
					EditorGUILayout.EndHorizontal();
					for(int i=start; i<end; i++) {
						if(Instances[i].Import) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Instances[i].Import = EditorGUILayout.Toggle(Instances[i].Import, GUILayout.Width(20f));
							EditorGUILayout.LabelField(Instances[i].Object.Name);
							EditorGUILayout.EndHorizontal();
						}
					}
				}
		
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory() {
		if(Directory.Exists(Source)) {
			DirectoryInfo info = new DirectoryInfo(Source);
			FileInfo[] items = info.GetFiles("*.trc");
			Files = new File[items.Length];
			for(int i=0; i<items.Length; i++) {
				Files[i] = new File();
				Files[i].Object = items[i];
				Files[i].Import = true;
			}
		} else {
			Files = new File[0];
		}
		ApplyFilter();
		Page = 1;
	}

	private void ApplyFilter() {
		if(Filter == string.Empty) {
			Instances = Files;
		} else {
			List<File> instances = new List<File>();
			for(int i=0; i<Files.Length; i++) {
				if(Files[i].Object.Name.ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Files[i]);
				}
			}
			Instances = instances.ToArray();
		}
	}

	private IEnumerator ImportMotionData() {
		string destination = "Assets/" + Destination;
		if(!AssetDatabase.IsValidFolder(destination)) {
			Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
		} else {
			Importing = true;
			for(int f=0; f<Files.Length; f++) {
				if(Files[f].Import) {
					if(!Directory.Exists(destination+"/"+Files[f].Object.Name) ) {
						AssetDatabase.CreateFolder(destination, Files[f].Object.Name);
						MotionData data = ScriptableObject.CreateInstance<MotionData>();
						data.name = "Data";
						AssetDatabase.CreateAsset(data, destination+"/"+Files[f].Object.Name+"/"+data.name+".asset");

						string[] lines = System.IO.File.ReadAllLines(Files[f].Object.FullName);
						char[] whitespace = new char[]{' '};

						//Get the Order for Channel
						int len_channel = 3;
						int[] channel = new int[len_channel];
						int index = 0;
						string[] entries = System.Text.RegularExpressions.Regex.Replace(lines[index], @"\s+", " ").Split(whitespace);
						string[] entries_channel = entries[2].Split('/');
						for(int i=0; i<entries_channel.Length; i++){
							if (entries_channel[i].Contains("X")){
								channel[i] = 0;
							}
							else if (entries_channel[i].Contains("Y")){
								channel[i] = 1;
							}
							else if (entries_channel[i].Contains("Z")){
								channel[i] = 2;
							}
						}

						//Set Framerate, Number of Joints, Number of Frames
						index += 2;
						entries = System.Text.RegularExpressions.Regex.Replace(lines[index], @"\s+", " ").Split(whitespace);
						data.Framerate = FileUtility.ReadInt(entries[0]);
						ArrayExtensions.Resize(ref data.Frames, FileUtility.ReadInt(entries[2]));
						int num_joint = FileUtility.ReadInt(entries[3]);

						//Record Joint Names/ Build Skeleton
						index += 1;
						data.Source = new MotionData.Hierarchy();
						string Parent = "None";
						entries = System.Text.RegularExpressions.Regex.Replace(lines[index], @"\s+", " ").Split(whitespace);
						for(int i=0; i<num_joint; i++){
							data.Source.AddBone(entries[i+2], Parent);
							//if(i==0){
							//	Parent = entries[i+2];
							//}
						}

						//Set Joint Positions
						index +=2;
						Vector3 position = Vector3.zero;
						Quaternion rotation = Quaternion.identity;
						for(int i=0; i<data.GetTotalFrames(); i++){
							data.Frames[i] = new Frame(data, i+1, (float)i / data.Framerate);
							entries = System.Text.RegularExpressions.Regex.Replace(lines[index+i], @"\s+", " ").Split(whitespace);
							for(int j=0; j<num_joint; j++){
								position[channel[0]] = FileUtility.ReadFloat(entries[j*3+2]);
								position[channel[1]] = FileUtility.ReadFloat(entries[j*3+3]);
								position[channel[2]] = FileUtility.ReadFloat(entries[j*3+4]);
								Matrix4x4 local = Matrix4x4.TRS(position/100f, rotation, Vector3.one);
								data.Frames[i].World[j] =local;
								//Debug.Log("frame"+ i + " " + position[channel[0]]+ " " + position[channel[1]]+ " " + position[channel[2]]);
							}
							
						}

						//If only one frame in the data
						if(data.GetTotalFrames() == 1) {
							Frame reference = data.Frames.First();
							ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt(data.Framerate));
							for(int i=0; i<data.GetTotalFrames(); i++) {
								data.Frames[i] = new Frame(data, i+1, (float)i / data.Framerate);
								data.Frames[i].World = (Matrix4x4[])reference.World.Clone();
							}
						}

						//Detect Symmetry
						data.DetectSymmetry();

						//Add Scene
						data.CreateScene();
						data.AddSequence();

						//Save
						EditorUtility.SetDirty(data);

					} else {
						Debug.Log("File with name " + Files[f].Object.Name + " already exists.");
					}

					yield return new WaitForSeconds(0f);
				}
			}
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
			Importing = false;
		}
		yield return new WaitForSeconds(0f);
	}

	[System.Serializable]
	public class File {
		public FileInfo Object;
		public bool Import;
	}

}
#endif
