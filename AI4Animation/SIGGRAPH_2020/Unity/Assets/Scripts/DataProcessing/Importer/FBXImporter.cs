#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

public class FBXImporter : EditorWindow {

	[System.Serializable]
	public class Asset {
		public GameObject Object = null;
		public bool Selected = true;
		public bool Imported = false;
	}

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public float Scale = 1f;

	public string Source = string.Empty;
	public string Destination = string.Empty;
	public string Filter = string.Empty;
	public Asset[] Assets = new Asset[0];
	public Asset[] Instances = new Asset[0];
	public bool Importing = false;

	public int Framerate = 60;

	public int Page = 1;
	public const int Items = 25;
	
	[MenuItem ("AI4Animation/Importer/FBX Importer")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(FBXImporter));
		Scroll = Vector3.zero;
	}

	void OnGUI() {
		if(Assets.Length > 0 && (System.Array.Find(Assets, x => x.Object) == null || System.Array.Find(Instances, x => x.Object) == null)) {
			LoadDirectory();
		}

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
					if(Utility.GUIButton("Load Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
						LoadDirectory();
					}
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

					Scale = EditorGUILayout.FloatField("Scale", Scale);

					Framerate = EditorGUILayout.IntField("Framerate", Framerate);

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Instances.Length; i++) {
							Instances[i].Selected = true;
						}
					}
					if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) {
						for(int i=0; i<Instances.Length; i++) {
							Instances[i].Selected = false;
						}
					}
					EditorGUILayout.EndHorizontal();

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
					for(int i=start; i<end; i++) {
						if(Instances[i].Selected) {
							if(Instances[i].Imported) {
								Utility.SetGUIColor(UltiDraw.DarkGreen);
							} else {
								Utility.SetGUIColor(UltiDraw.Mustard);
							}
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Instances[i].Selected = EditorGUILayout.Toggle(Instances[i].Selected, GUILayout.Width(20f));
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
			List<Asset> assets = new List<Asset>();
			LoadDirectory(source, assets);
			Assets = assets.ToArray();
		} else {
			Assets = new Asset[0];
		}
		ApplyFilter();
		Page = 1;
	}

	private void LoadDirectory(string folder, List<Asset> assets) {
		DirectoryInfo info = new DirectoryInfo(folder);
		foreach(FileInfo i in info.GetFiles()) {
			string path = i.FullName.Substring(i.FullName.IndexOf("Assets"));
			if((AnimationClip)AssetDatabase.LoadAssetAtPath(path, typeof(AnimationClip))) {
				Asset asset = new Asset();
				asset.Object = (GameObject)AssetDatabase.LoadAssetAtPath(path, typeof(GameObject));
				asset.Selected = true;
				asset.Imported = false;
				assets.Add(asset);
			}
		}
		foreach(DirectoryInfo i in info.GetDirectories()) {
			LoadDirectory(i.FullName, assets);
		}
	}

	private void ApplyFilter() {
		if(Filter == string.Empty) {
			Instances = Assets;
		} else {
			List<Asset> instances = new List<Asset>();
			for(int i=0; i<Assets.Length; i++) {
				if(Assets[i].Object.name.ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Assets[i]);
				}
			}
			Instances = instances.ToArray();
		}
	}
	
	private IEnumerator ImportMotionData() {
		foreach(Asset a in Assets) {
			a.Imported = false;
		}
		string destination = "Assets/" + Destination;
		if(!AssetDatabase.IsValidFolder(destination)) {
			Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
		} else {
			int added = 0;
			Importing = true;
			foreach(Asset asset in Assets) {
				if(asset.Selected) {
					string assetName = asset.Object.name.Replace(".fbx", "");
					if(!Directory.Exists(destination+"/"+asset.Object.name) ) {
						AssetDatabase.CreateFolder(destination, asset.Object.name);
						MotionData data = ScriptableObject.CreateInstance<MotionData>();
						data.name = assetName;
						AssetDatabase.CreateAsset(data, destination+"/"+asset.Object.name+"/"+data.name+".asset");
						AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(AssetDatabase.GetAssetPath(asset.Object), typeof(AnimationClip));

						//Create Actor
						GameObject instance = Instantiate(asset.Object) as GameObject;
						instance.name = asset.Object.name;
						Actor actor = instance.AddComponent<Actor>();
						string[] names = actor.GetBoneNames();
						ArrayExtensions.RemoveAt(ref names, 0);
						actor.ExtractSkeleton(names);

						//Create Source Data
						data.Source = new MotionData.Hierarchy();
						for(int i=0; i<actor.Bones.Length; i++) {
							data.Source.AddBone(actor.Bones[i].GetName(), actor.Bones[i].GetParent() == null ? "None" : actor.Bones[i].GetParent().GetName());
						}

						//Set Frames
						ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt((float)Framerate * clip.length));

						//Set Framerate
						data.Framerate = (float)Framerate;

						//Compute Frames
						for(int i=0; i<data.GetTotalFrames(); i++) {
							clip.SampleAnimation(instance, (float)i / data.Framerate);
							Matrix4x4[] transformations = new Matrix4x4[actor.Bones.Length];
							for(int j=0; j<transformations.Length; j++) {
								transformations[j] = Matrix4x4.TRS(Scale * actor.Bones[j].Transform.position, actor.Bones[j].Transform.rotation, Vector3.one);
							}
							data.Frames[i] = new Frame(data, i+1, (float)i / data.Framerate, transformations);
						}

						//Remove Actor
						Utility.Destroy(instance);

						//Detect Symmetry
						data.DetectSymmetry();

						//Add Scene
						data.CreateScene();
						data.AddSequence();

						EditorUtility.SetDirty(data);

						added += 1;
					} else {
						Debug.Log("Asset with name " + asset.Object.name + " already exists.");
					}
				}
			}
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
			Importing = false;
			foreach(Asset a in Assets) {
				a.Imported = false;
			}

			Debug.Log("Added " + added + " new assets.");
		}
		
		yield return new WaitForSeconds(0f);
	}

}
#endif