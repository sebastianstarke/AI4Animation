#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System;

namespace AI4Animation {
    public class MotionAssetRename : BatchProcessor {

        public string Source = string.Empty;

        public int MaxLength = 10000;
        public string Filter = "";

        [MenuItem ("AI4Animation/Tools/Rename Motion Asset")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(MotionAssetRename));
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
            Filter = EditorGUILayout.TextField("String to be removed: ", Filter);
            MaxLength = EditorGUILayout.IntField("Max Length: ", MaxLength);
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
							if(path.EndsWith(".asset")) {
								paths.Add(path);
							}
						}
						//Resources.UnloadUnusedAssets();
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

        private string FilterName(string fileName){
            string result = fileName;
            if(Filter.Length > 0){
                result = fileName.Replace(Filter, "");
            }
            //trim
            if(result.Length > MaxLength) {
                result = result.Substring(0, MaxLength);
            }
            return result;
        }

        public override IEnumerator DerivedProcess(Item item) {
            MotionAsset asset = AssetDatabase.LoadAssetAtPath<MotionAsset>(item.ID);
            string folder = asset.GetDirectoryPath();
            Iterate(folder);

            void Iterate(string folder) {
                DirectoryInfo info = new DirectoryInfo(folder);
                //rename files
                foreach(FileInfo i in info.GetFiles()) {
                    string path = i.FullName.Substring(i.FullName.IndexOf("Assets"));
                    AssetDatabase.RenameAsset(path, FilterName(i.Name));
                }
                Resources.UnloadUnusedAssets();
                //rename sub directories
                foreach(DirectoryInfo i in info.GetDirectories()) {
                    Iterate(i.FullName);
                }
            }
            //rename folder
            AssetDatabase.RenameAsset(folder, FilterName(asset.name));

            yield return new WaitForSeconds(0f);
        }

        public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
        }

        public override void DerivedFinish() {
            LoadDirectory(Source);
        }

    }
}
#endif
