#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class FBXImporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Source = string.Empty;
	public string Destination = string.Empty;
	public string Filter = string.Empty;
	public File[] Files = new File[0];
	public File[] Instances = new File[0];
	public bool Importing = false;

	public int Framerate = 60;
	public Actor Character = null;

	public int Page = 1;
	public const int Items = 25;
	
	[MenuItem ("Data Processing/FBX Importer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(FBXImporter));
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
					EditorGUILayout.LabelField("FBX Importer");
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
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
					Source = EditorGUILayout.TextField(Source);
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

					Framerate = EditorGUILayout.IntField("Framerate", Framerate);
					Character = (Actor)EditorGUILayout.ObjectField("Character", Character, typeof(Actor), true);

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
							EditorGUILayout.LabelField(Instances[i].Object.name);
							EditorGUILayout.EndHorizontal();
						}
					}
				}
		
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory() {
		string source = Application.dataPath + "/" + Source;
		if(Directory.Exists(source)) {
			DirectoryInfo info = new DirectoryInfo(source);
			FileInfo[] items = info.GetFiles();
			List<File> files = new List<File>();
			for(int i=0; i<items.Length; i++) {
				string path = items[i].FullName.Substring(items[i].FullName.IndexOf("Assets/"));
				if((AnimationClip)AssetDatabase.LoadAssetAtPath(path, typeof(AnimationClip))) {
					File file = new File();
					file.Object = (GameObject)AssetDatabase.LoadAssetAtPath(path, typeof(GameObject));
					file.Import = false;
					files.Add(file);
				}
			}
			Files = files.ToArray();
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
				if(Files[i].Object.name.ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Files[i]);
				}
			}
			Instances = instances.ToArray();
		}
	}
	
	private IEnumerator ImportMotionData() {
		string destination = "Assets/" + Destination;
		if(Character == null) {
			Debug.Log("No character model assigned.");
		} else if(!AssetDatabase.IsValidFolder(destination)) {
			Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
		} else {
			Importing = true;
			Character.transform.position = Vector3.zero;
			Character.transform.rotation = Quaternion.identity;
			for(int f=0; f<Files.Length; f++) {
				if(Files[f].Import) {
					if(AssetDatabase.LoadAssetAtPath(destination+"/"+Files[f].Object.name+".asset", typeof(MotionData)) == null) {
						AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(AssetDatabase.GetAssetPath(Files[f].Object), typeof(AnimationClip));
						MotionData data = ScriptableObject.CreateInstance<MotionData>();

						//Assign Name
						data.name = Files[f].Object.name;

						AssetDatabase.CreateAsset(data , destination+"/"+data.name+".asset");

						//Create Source Data
						data.Source = new MotionData.Hierarchy();
						for(int i=0; i<Character.Bones.Length; i++) {
							data.Source.AddBone(Character.Bones[i].GetName(), Character.Bones[i].GetParent() == null ? "None" : Character.Bones[i].GetParent().GetName());
						}

						//Set Frames
						ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt((float)Framerate * clip.length));

						//Set Framerate
						data.Framerate = (float)Framerate;

						//Compute Frames
						for(int i=0; i<data.GetTotalFrames(); i++) {
							data.Frames[i] = new Frame(data, i+1, (float)i / data.Framerate);
							clip.SampleAnimation(Character.gameObject, data.Frames[i].Timestamp);
							for(int j=0; j<Character.Bones.Length; j++) {
								data.Frames[i].Local[j] = Character.Bones[j].Transform.GetLocalMatrix();
								data.Frames[i].World[j] = Character.Bones[j].Transform.GetWorldMatrix();
							}
						}

						//Finalise
						data.DetectSymmetry();
						data.AddSequence();

						EditorUtility.SetDirty(data);
					} else {
						Debug.Log("File with name " + Files[f].Object.name + " already exists.");
					}
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
		public GameObject Object;
		public bool Import;
	}

}
#endif