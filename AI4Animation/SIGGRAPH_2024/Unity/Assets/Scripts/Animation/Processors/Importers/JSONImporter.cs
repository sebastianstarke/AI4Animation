#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Defective.JSON;

namespace AI4Animation {
	public class JSONImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;
		// public bool Flip = true;
		public Axis Axis = Axis.XPositive;

		private List<string> Imported;
		private List<string> Skipped;

		[MenuItem ("AI4Animation/Importer/JSON Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(JSONImporter));
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
				Source = EditorUtility.OpenFolderPanel("JSON Importer", Source == string.Empty ? Application.dataPath : Source, "");
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
						foreach(FileInfo i in info.GetFiles("*.json")) {
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
			string target = (destination + item.ID.Remove(0, source.Length)).Replace(".json", "");

			if(!Directory.Exists(target)) {
				Directory.CreateDirectory(target);

				FileInfo file = new FileInfo(item.ID);

				MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
				asset.name = file.Name;
				AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");

                string[] lines = System.IO.File.ReadAllLines(file.FullName);
                Debug.Log(lines.Length);
                //JSONObject json = JSONObject.Create(lines[0]);
                //Debug.Log(json.ToString());

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

        [System.Serializable]
        public class RuntimeAnimationJSON{
            /* {
                "SkeletonDefinition": [
                    { "BoneName": "body_world" },
                    { "BoneName": "b_root" },
                ],
                "AnimationFrames": [
                    {
                        "Time": 11.9557175,
                        "PosesLocalSpace": [
                            [ 24.4814, -6.36511, 0.0, 0.0, 0.0, 0.99954, 0.03018, 0.89695 ],
                            [ 0.0, 0.0, 86.91593, -0.49259, 0.55739, -0.44258, 0.5008, 1.0 ]
                    },
                    {
                        "Time": 11.9723389,
                        "PosesLocalSpace": [
                            [ 24.22694, -6.44363, 0.0, 0.0, 0.0, 0.99944, 0.03359, 0.89695 ],
                            [ 0.0, 0.0, 86.95305, -0.49025, 0.55717, -0.44275, 0.50318, 1.0 ]
                    }
                ],
            } */

            public static RuntimeAnimationJSON CreateFromJSON(string jsonString)
            {
                return JsonUtility.FromJson<RuntimeAnimationJSON>(jsonString);
            }
        }

        [System.Serializable]
        private class SkeletonDefinition {

        }

        [System.Serializable]
        private class AnimationFrames {

        }
	}
}
#endif

