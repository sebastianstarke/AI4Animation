#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class FBXConnectImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;
		public string Name = string.Empty;

		private List<string> Imported;
		private List<string> Skipped;

		private MotionAsset Asset;

		[MenuItem ("AI4Animation/Importer/FBX Connect Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(FBXConnectImporter));
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
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			SetSource(EditorGUILayout.TextField(Source));
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			Destination = EditorGUILayout.TextField(Destination);
			EditorGUILayout.EndHorizontal();

			Name = EditorGUILayout.TextField("Name", Name);

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
				directory = Application.dataPath + "/" + directory;
				if(Directory.Exists(directory)) {
					List<string> paths = new List<string>();
					Iterate(directory);
					LoadItems(paths.ToArray());
					void Iterate(string folder) {
						DirectoryInfo info = new DirectoryInfo(folder);
						foreach(FileInfo i in info.GetFiles()) {
							string path = i.FullName.Substring(i.FullName.IndexOf("Assets"));
							if((AnimationClip)AssetDatabase.LoadAssetAtPath(path, typeof(AnimationClip))) {
								paths.Add(path);
							}
						}
						Resources.UnloadUnusedAssets();
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
			Asset = null;
		}

		public override IEnumerator DerivedProcess(Item item) {
			string source = "Assets/" + Source;
			string destination = "Assets/" + Destination;

			string target = destination + "/" + Name;
			bool created = Asset != null;

			if(!Directory.Exists(target) || created) {
				GameObject go = (GameObject)AssetDatabase.LoadAssetAtPath(item.ID, typeof(GameObject));
				AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(item.ID, typeof(AnimationClip));

				try {
					//Create Model
					GameObject instance = Instantiate(go) as GameObject;
					List<Transform> transforms = new List<Transform>(instance.GetComponentsInChildren<Transform>());
					transforms.RemoveAt(0);

					if(!created) {
						//Create Directory
						Directory.CreateDirectory(target);

						//Create Asset
						Asset = ScriptableObject.CreateInstance<MotionAsset>();
						Asset.name = Name;
						AssetDatabase.CreateAsset(Asset, target+"/"+Asset.name+".asset");

						//Create Source Data
						Asset.Source = new MotionAsset.Hierarchy(transforms.Count);
						for(int i=0; i<Asset.Source.Bones.Length; i++) {
							Asset.Source.SetBone(i, transforms[i].name, transforms[i].parent == instance.transform ? "None" : transforms[i].parent.name);
						}

						//Set Framerate
						Asset.Framerate = clip.frameRate;
					}

					//Frame Pivot and Length
					int pivot = Asset.Frames.Length;
					int length = Mathf.RoundToInt(clip.frameRate * clip.length);

					//Set Frames
					ArrayExtensions.Resize(ref Asset.Frames, pivot + length);

					//Add Sequence
					Asset.AddSequence(pivot+1, pivot+length);

					//Compute Frames
					Matrix4x4[] transformations = new Matrix4x4[Asset.Source.Bones.Length];
					for(int i=0; i<length; i++) {
						clip.SampleAnimation(instance, (float)i / Asset.Framerate);
						for(int j=0; j<transformations.Length; j++) {
							transformations[j] = Matrix4x4.TRS(transforms[j].position, transforms[j].rotation, Vector3.one);
						}
						int idx = pivot+i;
						Asset.Frames[idx] = new Frame(Asset, idx, (float)idx / Asset.Framerate, transformations);
					}

					//Remove Model
					Utility.Destroy(instance);

					if(!created) {
						//Detect Symmetry
						Asset.DetectSymmetry();

						//Add Scene
						Asset.CreateScene();
					}

					//Save
					EditorUtility.SetDirty(Asset);

					Imported.Add(target);
				} catch(System.Exception e) {
					Debug.LogWarning(e.Message);
					// if(Directory.Exists(target)) {
					// 	Directory.Delete(target);
					// }
				}
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
			if(Imported.Count > 0) {
				AssetDatabase.Refresh();
			}

			Debug.Log("Imported " + Imported.Count + " assets.");
			Imported.ToArray().Print();

			Debug.Log("Skipped " + Skipped.Count + " assets.");
			Skipped.ToArray().Print();
		}

	}
}
#endif