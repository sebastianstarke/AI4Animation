#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
    public class MotionAssetCopier : BatchProcessor {

        public string Source = string.Empty;
        public string Destination = string.Empty;

		private List<string> Imported;
		private List<string> Skipped;

        [MenuItem ("AI4Animation/Tools/Motion Asset Copier")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(MotionAssetCopier));
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

			if(Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
				LoadDirectory(Source);
			}
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
							if((MotionAsset)AssetDatabase.LoadAssetAtPath(path, typeof(MotionAsset))) {
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

        public override void DerivedInspector(Item item) {
        
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {

        }

        public override IEnumerator DerivedProcess(Item item) {
            MotionAsset asset = AssetDatabase.LoadAssetAtPath<MotionAsset>(item.ID);
            asset.MakeCopy("Assets" + "/" + Destination + "/" + asset.name);
            yield return new WaitForSeconds(0f);
        }

        public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
        }

        public override void DerivedFinish() {

        }

    }
}
#endif
