#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class FBXMultiImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;

		public int Framerate = 60;
		public float Scale = 1f;

		[MenuItem ("AI4Animation/Importer/FBX Multi Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(FBXMultiImporter));
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

			Scale = EditorGUILayout.FloatField("Scale", Scale);

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

		}

		private class BoneMapping {
			public Transform Self;
			public Transform Source;
			public BoneMapping(Transform self, Transform source) {
				Self = self;
				Source = source;
			}
		}

		public override IEnumerator DerivedProcess(Item item) {

			string source = "Assets/" + Source;
			string destination = "Assets/" + Destination;

			string target = (destination + item.ID.Remove(0, source.Length)).Replace(".fbx", "");
			if(!Directory.Exists(target)) {
				Directory.CreateDirectory(target);
				GameObject go = (GameObject)AssetDatabase.LoadAssetAtPath(item.ID, typeof(GameObject));
				AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(item.ID, typeof(AnimationClip));

				//Create Actor
				GameObject instance = Instantiate(go) as GameObject;
				instance.name = go.name;
				List<Transform> transforms = new List<Transform>(instance.GetComponentsInChildren<Transform>());
				transforms.RemoveAt(0);

				List<MotionAsset> subAssets = new List<MotionAsset>();
				List<BoneMapping[]> subCharacters = new List<BoneMapping[]>();

				for(int k=0; k<instance.transform.childCount; k++) {
					Transform root = instance.transform.GetChild(k);
					GameObject subInstance = GameObject.Instantiate(root.gameObject);
					subInstance.name = root.name;
					List<Transform> subTransforms = new List<Transform>(subInstance.GetComponentsInChildren<Transform>());
				
					//Create Asset
					int index = k+1;
					Directory.CreateDirectory(target+"/"+index);
					MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
					subAssets.Add(asset);
					asset.name = go.name+"_"+ index;
					AssetDatabase.CreateAsset(asset, target+"/"+index+"/"+asset.name + ".asset");

					//Create Source Data
					asset.Source = new MotionAsset.Hierarchy(subTransforms.Count);
					for(int i=0; i<asset.Source.Bones.Length; i++) {
						asset.Source.SetBone(i, subTransforms[i].name, subTransforms[i].parent == null ? "None" : subTransforms[i].parent.name);
					}

					//Set Frames
					ArrayExtensions.Resize(ref asset.Frames, Mathf.RoundToInt((float)Framerate * clip.length));

					//Set Framerate
					asset.Framerate = (float)Framerate;

					//Cache the Mapping Between Subactor and Actor 
					BoneMapping[] mapping = new BoneMapping[subTransforms.Count];
					for(int i=0; i<mapping.Length; i++) {
						mapping[i] = new BoneMapping(subTransforms[i], transforms.Find(x => x.name == subTransforms[i].name));
					}
					subCharacters.Add(mapping);
				}
				
				for(int i=0; i<subAssets[0].GetTotalFrames(); i++) {
					clip.SampleAnimation(instance, (float)i / subAssets[0].Framerate);
					for(int k=0; k<subCharacters.Count; k++) {
						Matrix4x4[] transformations = new Matrix4x4[subCharacters[k].Length];
						for(int j=0; j<transformations.Length; j++) {
							transformations[j] = Matrix4x4.TRS(
								Scale * subCharacters[k][j].Source.position, 
								subCharacters[k][j].Source.rotation,
								Vector3.one
							);
						}
						subAssets[k].Frames[i] = new Frame(subAssets[k], i+1, (float)i / subAssets[k].Framerate, transformations);
					}
				}
				
				for(int i=0; i<subAssets.Count; i++) {
					//Remove Actor
					Utility.Destroy(subCharacters[i][0].Self.gameObject);

					//Detect Symmetry
					subAssets[i].DetectSymmetry();

					//Add Sequence
					subAssets[i].AddSequence();

					//Add Scene
					subAssets[i].CreateScene();

					//Save
					EditorUtility.SetDirty(subAssets[i]);
				}
					
				Utility.Destroy(instance);

				Debug.Log("Asset at " + target + " successfully imported.");
			} else {
				Debug.Log("Asset at " + target + " already exists.");
			}

			yield return new WaitForSeconds(0f);
		}

		public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
		}

		public override void DerivedFinish() {
			AssetDatabase.Refresh();
		}

	}
}
#endif