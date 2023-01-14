#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class FBXImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;
		public Actor Skeleton = null;

		private List<string> Imported;
		private List<string> Skipped;

		[MenuItem ("AI4Animation/Importer/FBX Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(FBXImporter));
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

			Skeleton = EditorGUILayout.ObjectField("Skeleton", Skeleton, typeof(Actor), true) as Actor;

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
		}

		public override IEnumerator DerivedProcess(Item item) {
			string source = "Assets/" + Source;
			string destination = "Assets/" + Destination;

			string target = (destination + item.ID.Remove(0, source.Length)).Replace(".fbx", "");
			if(!Directory.Exists(target)) {
				GameObject go = (GameObject)AssetDatabase.LoadAssetAtPath(item.ID, typeof(GameObject));
				AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(item.ID, typeof(AnimationClip));

				try {
					//Create Directory
					Directory.CreateDirectory(target);

					//Create Asset
					MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
					asset.name = go.name;
					AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");

					//Set Frames
					ArrayExtensions.Resize(ref asset.Frames, Mathf.RoundToInt(clip.frameRate * clip.length));

					//Set Framerate
					asset.Framerate = clip.frameRate;

					//Create Model
					GameObject instance = Instantiate(go) as GameObject;
					List<Transform> transforms = new List<Transform>(instance.GetComponentsInChildren<Transform>());
					transforms.RemoveAt(0);

					if(Skeleton == null) {
						//Create Source Data
						asset.Source = new MotionAsset.Hierarchy(transforms.Count);
						for(int i=0; i<asset.Source.Bones.Length; i++) {
							asset.Source.SetBone(i, transforms[i].name, transforms[i].parent == instance.transform ? "None" : transforms[i].parent.name);
						}

						//Compute Frames
						Matrix4x4[] transformations = new Matrix4x4[asset.Source.Bones.Length];
						for(int i=0; i<asset.GetTotalFrames(); i++) {
							clip.SampleAnimation(instance, (float)i / asset.Framerate);
							for(int j=0; j<transformations.Length; j++) {
								transformations[j] = transforms[j].GetWorldMatrix();
							}
							asset.Frames[i] = new Frame(asset, i+1, (float)i / asset.Framerate, transformations);
						}
					} else {
						//Create Source Data
						asset.Source = new MotionAsset.Hierarchy(Skeleton.Bones.Length);
						for(int i=0; i<Skeleton.Bones.Length; i++) {
							asset.Source.SetBone(i, Skeleton.Bones[i].GetName(), Skeleton.Bones[i].GetParent() == null ? "None" : Skeleton.Bones[i].GetParent().GetName());
						}

						//Compute Frames
						int[] mapping = new int[Skeleton.Bones.Length];
						for(int i=0; i<Skeleton.Bones.Length; i++) {
							mapping[i] = transforms.FindIndex(x => x.name == Skeleton.Bones[i].GetName());
						}
						Matrix4x4[] transformations = new Matrix4x4[Skeleton.Bones.Length];
						for(int i=0; i<asset.GetTotalFrames(); i++) {
							clip.SampleAnimation(instance, (float)i / asset.Framerate);
							for(int j=0; j<transformations.Length; j++) {
								transformations[j] = transforms[mapping[j]].GetWorldMatrix();
							}
							asset.Frames[i] = new Frame(asset, i+1, (float)i / asset.Framerate, transformations);
						}
					}

					//Remove Instance
					Utility.Destroy(instance);

					//Detect Symmetry
					asset.DetectSymmetry();

					//Add Sequence
					asset.AddSequence();

					//Add Scene
					asset.CreateScene();

					//Save
					EditorUtility.SetDirty(asset);

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